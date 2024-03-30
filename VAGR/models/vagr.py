import sys

sys.path.append(".")

import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tree import head_to_tree, tree_to_adj
# from VAGR.models.TextVAE.text_vae import TextVAE
from .VL import TextVAE


class DualGCNClassifier(nn.Module):
    def __init__(self, embedding_matrix, opt, config):
        super().__init__()
        in_dim = opt.hidden_dim
        self.opt = opt
        self.gcn_model = GCNAbsaModel(embedding_matrix=embedding_matrix, opt=opt, config=config)
        self.classifier = nn.Linear(in_dim * 2, opt.polarities_dim)

    def forward(self, inputs):
        global vaeout, mean, std

        outputs1, outputs2, vaeout, mean, std, adj_ag, adj_dep = self.gcn_model(inputs)
        final_outputs = torch.cat((outputs1, outputs2), dim=-1)
        # final_outputs = outputs1  # gcn
        # final_outputs = outputs2  # vagr
        # penalbi = 0.0 * (outputs1.size(0) / torch.norm(outputs1 - outputs2)).cuda()

        logits = self.classifier(final_outputs)
        # logits = F.softmax(logits)

        adj_ag_T = adj_ag.transpose(1, 2)
        identity = torch.eye(adj_ag.size(1)).cuda()
        identity = identity.unsqueeze(0).expand(adj_ag.size(0), adj_ag.size(1), adj_ag.size(1))
        ortho = adj_ag @ adj_ag_T

        for i in range(ortho.size(0)):
            ortho[i] -= torch.diag(torch.diag(ortho[i]))
            ortho[i] += torch.eye(ortho[i].size(0)).cuda()

        penal1 = (torch.norm(ortho - identity) / adj_ag.size(0)).cuda()
        penal2 = (adj_ag.size(0) / torch.norm(adj_ag - adj_dep)).cuda()
        # penal = self.opt.beta * penal2
        penal = self.opt.alpha * penal1 + self.opt.beta * penal2

        return logits, penal, vaeout, mean, std


class GCNAbsaModel(nn.Module):
    def __init__(self, embedding_matrix, opt, config):
        super().__init__()
        self.opt = opt
        self.config = config
        self.embedding_matrix = embedding_matrix
        self.emb = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=True)
        self.pos_emb = nn.Embedding(opt.pos_size, opt.pos_dim, padding_idx=0) if opt.pos_dim > 0 else None  # POS emb
        self.post_emb = nn.Embedding(opt.post_size, opt.post_dim,
                                     padding_idx=0) if opt.post_dim > 0 else None  # position emb
        embeddings = (self.emb, self.pos_emb, self.post_emb)
        # gcn layer
        self.gcn = GCN(opt, embeddings, opt.hidden_dim, opt.num_layers)
        # self.selfattention = SelfAttention_weight(self.opt.gcn_dropout, config['embed_size'])
        # self.attention = Attention_weight(self.opt.gcn_dropout, config['embed_size'], 50)
        # vae model
        self.vae = TextVAE(
            opt=self.opt,
            emb=self.emb,
            vocab_size=opt.tok_size,
            embed_size=config['embed_size'],
            hidden_size=self.opt.encoder_hidden,
            num_layers=config['num_layers'],
            dropout=self.opt.vae_dropout,
            enc_dec_tying=config['enc_dec_tying'],
            dec_gen_tying=config['dec_gen_tying']
        )
        # self.vae = VAE(output_size=opt.tok_size, dropout=config['dropout'])

    def forward(self, inputs):
        tok, asp, pos, head, deprel, post, mask, l, adj = inputs  # unpack inputs
        maxlen = max(l.data)
        mask = mask[:, :maxlen]
        if self.opt.parseadj:
            adj_dep = adj[:, :maxlen, :maxlen].float()
        else:
            def inputs_to_tree_reps(head, words, l):
                trees = [head_to_tree(head[i], words[i], l[i]) for i in range(len(l))]
                adj = [tree_to_adj(maxlen, tree, directed=self.opt.direct, self_loop=self.opt.loop).reshape(1, maxlen, maxlen) for tree in trees]
                adj = np.concatenate(adj, axis=0)
                adj = torch.from_numpy(adj)
                return adj.cuda()

            adj_dep = inputs_to_tree_reps(head.data, tok.data, l.data)

        h, adj_ag, hidden_lstm = self.gcn(adj_dep, inputs)
        vaeout, output_asp, mean, std, encoding = self.vae(tok, hidden_lstm, asp)
        # vaeout, mean, std, encoding = self.vae(hidden_lstm)
        # vaeout = torch.cat((vaeout, torch.zeros(vaeout.shape[0], asp.shape[1] - vaeout.shape[1], vaeout.shape[2]).cuda()), dim=1)

        # output_asp = output_asp[:, :h.shape[1], :]
        # weight_asp = self.selfattention(output_asp)
        # weight_asp = self.attention(h, output_asp)
        # global attention pooling
        # outputs1 = torch.mean(torch.bmm(weight_asp, h), dim=1)

        # avg pooling asp feature
        asp_wn = mask.sum(dim=1).unsqueeze(-1)  # aspect words num
        mask = mask.unsqueeze(-1).repeat(1, 1, self.opt.hidden_dim)  # mask for h
        outputs1 = (h * mask).sum(dim=1) / asp_wn
        # global pooling
        # outputs1 = torch.mean(h, dim=1)
        outputs2 = encoding.squeeze()

        # normalize
        demo1 = (torch.max(outputs1, dim=1)[0] - torch.min(outputs1, dim=1)[0]).view(-1, 1).repeat(1, outputs1.shape[1])
        demo2 = (torch.max(outputs2, dim=1)[0] - torch.min(outputs2, dim=1)[0]).view(-1, 1).repeat(1, outputs2.shape[1])
        outputs1 = outputs1 / demo1
        outputs2 = outputs2 / demo2

        return outputs1, outputs2, vaeout, mean, std, adj_ag, adj_dep



class SelfAttention_weight(nn.Module):
    def __init__(self, dropout, size):
        super(SelfAttention_weight, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.size = size
        self.query_proj = nn.Linear(size, size)
        self.key_proj = nn.Linear(size, size)

    def forward(self, input):
        query = self.query_proj(input)
        key = self.key_proj(input)
        d = query.shape[-1]
        scores = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(d)
        # scores = self.dropout(scores)
        attention_weights = torch.softmax(scores, dim=2)
        return attention_weights


class Attention_weight(nn.Module):
    def __init__(self, dropout, size_emb, size_hid):
        super(Attention_weight, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.size_emb = size_emb
        self.query_proj = nn.Linear(size_hid, size_hid)
        self.key_proj = nn.Linear(size_emb, size_hid)

    def forward(self, h, output_asp):
        query = self.query_proj(h)
        key = self.key_proj(output_asp)
        d = query.shape[-1]
        scores = torch.bmm(query, key.transpose(1, 2)).transpose(1, 2) / math.sqrt(d)
        # scores = self.dropout(scores)
        attention_weights = torch.softmax(scores, dim=2)
        return attention_weights


class GCN(nn.Module):
    def __init__(self, opt, embeddings, mem_dim, num_layers):
        super(GCN, self).__init__()
        self.opt = opt
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = opt.embed_dim + opt.post_dim + opt.pos_dim
        self.emb, self.pos_emb, self.post_emb = embeddings

        # rnn layer
        input_size = self.in_dim
        self.rnn = nn.LSTM(input_size, opt.rnn_hidden, opt.rnn_layers, batch_first=True,
                           dropout=opt.rnn_dropout, bidirectional=opt.bidirect)
        if opt.bidirect:
            self.in_dim = opt.rnn_hidden * 2
        else:
            self.in_dim = opt.rnn_hidden

        # drop out
        self.rnn_drop = nn.Dropout(opt.rnn_dropout)
        self.in_drop = nn.Dropout(opt.input_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.attention_heads = opt.attention_heads
        self.attn = MultiHeadAttention(self.attention_heads, self.mem_dim * 2)
        self.weight_list = nn.ModuleList()
        for j in range(self.layers):
            input_dim = self.in_dim if j == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))

        self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))

    def encode_with_rnn(self, rnn_inputs, seq_lens, batch_size):
        h0, c0 = rnn_zero_state(batch_size, self.opt.rnn_hidden, self.opt.rnn_layers, self.opt.bidirect)
        rnn_inputs = nn.utils.rnn.pack_padded_sequence(rnn_inputs, seq_lens, batch_first=True, enforce_sorted=False)
        rnn_outputs, (ht, ct) = self.rnn(rnn_inputs, (h0, c0))
        rnn_outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        return rnn_outputs

    def forward(self, adj, inputs):
        tok, asp, pos, head, deprel, post, mask, l, _ = inputs  # unpack inputs
        l = l.cpu()
        src_mask = (tok != 0).unsqueeze(-2)
        maxlen = max(l.data)
        mask_ = (torch.zeros_like(tok) != tok).float().unsqueeze(-1)[:, :maxlen]

        # embedding
        word_embs = self.emb(tok)
        embs = [word_embs]
        if self.opt.pos_dim > 0:
            embs += [self.pos_emb(pos)]
        if self.opt.post_dim > 0:
            embs += [self.post_emb(post)]
        embs = torch.cat(embs, dim=2)
        embs = self.in_drop(embs)

        # rnn layer
        self.rnn.flatten_parameters()
        gcn_inputs = self.rnn_drop(self.encode_with_rnn(embs, l, tok.size()[0]))

        denom_dep = adj.sum(2).unsqueeze(2) + 1
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        outputs_dep = None
        adj_ag = None

        # * Average Multi-head Attention matrixes
        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag += attn_adj_list[i]

        adj_ag = adj_ag / self.attention_heads

        for j in range(adj_ag.size(0)):
            adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
            adj_ag[j] += torch.eye(adj_ag[j].size(0)).cuda()
        adj_ag = mask_ * adj_ag

        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1

        # graph ensemble
        # type 0
        # adj_ensemble = adj
        # adj_ensemble = adj_ag
        # type 1
        # adj_ensemble = (adj + adj_ag) / 2
        # type 2
        for i in range(adj_ag.shape[0]):  # prune
            del_rate = self.opt.prune_rate  # delete rate
            k = int((torch.nonzero(adj_ag[i]).shape[0] - adj_ag[i].shape[0]) * del_rate)
            thre, _ = torch.topk(adj_ag[i].view(1, -1), k + adj_ag[i].shape[0] * adj_ag[i].shape[1] - torch.nonzero(adj_ag[i]).shape[0], largest=False, sorted=True)
            zeros = torch.zeros(adj_ag.shape[1], adj_ag.shape[2]).cuda()
            threnum = thre[0, -1].item()
            adj_ag[i] = torch.where(adj_ag[i] <= threnum, zeros, adj_ag[i])
        alpha = self.opt.ensemble_alpha
        adj_ensemble = alpha * adj + (1 - alpha) * adj_ag
        # type 3
        # for i in range(adj_ag.shape[0]):
        #     del_rate = 0.9  # delete rate
        #     k = int((torch.nonzero(adj_ag[i]).shape[0] - adj_ag[i].shape[0]) * del_rate)
        #     thre, _ = torch.topk(adj_ag[i].view(1, -1), k + adj_ag[i].shape[0] * adj_ag[i].shape[1] - torch.nonzero(adj_ag[i]).shape[0], largest=False, sorted=True)
        #     zeros = torch.zeros(adj_ag.shape[1], adj_ag.shape[2]).cuda()
        #     threnum = thre[0, -1].item()
        #     adj_ag[i] = torch.where(adj_ag[i] <= threnum, zeros, adj_ag[i])
        # adj_ensemble = adj + adj_ag
        # ones = torch.ones(adj_ag.shape[0], adj_ag.shape[1], adj_ag.shape[2]).cuda()
        # adj_ensemble = torch.where(adj_ensemble > 0.001, ones, adj_ensemble)
        # type 4 : attention -> sigmoid
        # ones = torch.ones(adj_ag.shape[0], adj_ag.shape[1], adj_ag.shape[2]).cuda()
        # adj_ag = torch.where(adj_ag > 0.001, ones, adj_ag)

        # alpha = 0.5
        # adj_ensemble = alpha * adj + (1 - alpha) * adj_ag

        denom_ensemble = adj_ensemble.sum(2).unsqueeze(2) + 1
        outputs_ensemble = gcn_inputs

        # outputs_ag = gcn_inputs
        # outputs_dep = gcn_inputs

        for l in range(self.layers):
            Ax_ensemble = adj_ensemble.bmm(outputs_ensemble)
            AxW_ensemble = self.W[l](Ax_ensemble)
            AxW_ensemble = AxW_ensemble / denom_ensemble
            gAxW_ensemble = F.relu(AxW_ensemble)

            # outputs_ensemble = self.gcn_drop(gAxW_ensemble) if l < self.layers - 1 else gAxW_ensemble
            outputs_ensemble = gAxW_ensemble
        return outputs_ensemble, adj_ag, gcn_inputs


def rnn_zero_state(batch_size, hidden_dim, num_layers, bidirectional=True):
    total_layers = num_layers * 2 if bidirectional else num_layers
    state_shape = (total_layers, batch_size, hidden_dim)
    h0 = c0 = Variable(torch.zeros(*state_shape), requires_grad=False)
    return h0.cuda(), c0.cuda()


def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    # p_attn = F.sigmoid(scores)

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
