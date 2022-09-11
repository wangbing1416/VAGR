'''
Description:
version:
Author: chenhao
Date: 2021-06-09 14:17:37
'''
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class GCNBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        self.classifier = nn.Linear(int(opt.bert_dim * 1.0), opt.polarities_dim)

    def forward(self, inputs, mode):
        outputs_dep, outputs_global, adj_dep, pooled_output, aspect_pred, loss_asp = self.gcn_model(inputs, mode)

        final_outputs = torch.cat((outputs_dep, outputs_global), dim=-1)

        feature = self.classifier(final_outputs)
        logits = F.softmax(feature, dim=1)

        return feature, logits, aspect_pred, loss_asp


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers)

    def forward(self, inputs, mode):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, src_mask, aspect_mask = inputs
        outputs_dep, pooled_output, aspect_pred, loss_asp = self.gcn(adj_dep, inputs, mode)

        # avg pooling asp feature
        if mode == 'unlabel':
            aspect_mask = torch.zeros(aspect_mask.size(0), aspect_mask.size(1)).cuda()
            _, indice = torch.topk(aspect_pred, k=2, dim=1)  # assign hard label for aspect prediction
            for i in range(indice.size(0)):
                aspect_mask[i, indice[i].tolist()] = 1
            # aspect_mask = aspect_pred  # assign soft pseudo label
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim // 2)
        outputs_global = torch.mean(outputs_dep, dim=1).squeeze()
        outputs_dep = (outputs_dep * aspect_mask).sum(dim=1) / asp_wn

        return outputs_dep, outputs_global, adj_dep, pooled_output, aspect_pred, loss_asp


class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)

        # self.bert_dense = nn.Linear(self.bert_dim, self.bert_dim)
        # self.bert_activate = nn.Tanh()
        self.aspect_classification = nn.Sequential(
            # nn.Linear(self.bert_dim, self.bert_dim),
            nn.Linear(self.bert_dim, 1)
        )

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.attn = MultiHeadAttention(opt.attention_heads, self.bert_dim)
        self.weight_list = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.bert_dim if j == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

    def forward(self, adj, inputs, mode):
        global selected_pred
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, src_mask, aspect_mask = inputs
        src_mask = src_mask.unsqueeze(-2)
        # aspect mask prediction
        text_masked = text_bert_indices.clone()
        weight = aspect_mask.clone()
        # select aspect index
        # weight = torch.abs(weight - 1)
        if mode == 'train':
            mask_ids = 103
            for i in range(text_masked.size(0)):
                # mask a half aspect
                if aspect_mask[i].sum() > 1:
                    for mask_asp in range(asp_start[i] + 1 + int(aspect_mask[i].sum() / 2), asp_end[i] + 1):
                        text_masked[i, mask_asp] = mask_ids
                # mask other words
                select_length_pred = aspect_mask[i].sum().item() * self.opt.negtive
                select_length_mask = aspect_mask[i].sum().item()
                sentence_length = attention_mask[i].sum().item()
                # can_choice_num = len(list(filter(lambda x: x < sentence_length, (aspect_mask[i] == 0).nonzero().squeeze().tolist())))
                # if can_choice_num < select_length_pred:  # semi-supervised aspect prediction may be large
                #     select_length_pred = can_choice_num
                # if can_choice_num < select_length_mask:
                #     select_length_mask = can_choice_num
                selected_pred = np.random.choice(
                    list(filter(lambda x: x < sentence_length, (aspect_mask[i] == 0).nonzero().squeeze().tolist())),
                    size=int(select_length_pred), replace=False).tolist()
                selected_mask = np.random.choice(selected_pred, size=int(select_length_mask), replace=False)
                text_masked[i, selected_mask] = mask_ids
                # select non-aspect index
                weight[i, selected_pred] = 1

        # bert
        res = self.bert(text_masked, attention_mask=attention_mask, output_hidden_states=False)
        sequence_output, pooled_output = res["last_hidden_state"], res["pooler_output"]
        sequence_output = self.layernorm(sequence_output)

        # aspect prediction
        aspect_pred = self.aspect_classification(sequence_output).squeeze()
        LOSS = nn.BCEWithLogitsLoss(weight=weight.detach())
        # LOSS = nn.BCEWithLogitsLoss()
        loss_asp = LOSS(aspect_pred, aspect_mask.float())

        gcn_inputs = self.bert_drop(sequence_output)
        pooled_output = self.pooled_drop(pooled_output)

        denom_dep = adj.sum(2).unsqueeze(2) + 1
        outputs_dep = None

        outputs_dep = gcn_inputs
        adj = adj.float()

        for l in range(self.layers):
            Ax_dep = adj.bmm(outputs_dep)
            AxW_dep = self.W[l](Ax_dep)
            AxW_dep = AxW_dep / denom_dep
            gAxW_dep = F.relu(AxW_dep)

            outputs_dep = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep

        return outputs_dep, pooled_output, torch.sigmoid(aspect_pred), loss_asp


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)
        return attn