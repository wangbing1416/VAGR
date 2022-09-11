import torch
import sys
sys.path.append(".")
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .TextVAE.constants import PAD_INDEX
from .TextVAE.multi_layer_gru_cell import MultiLayerGRUCell
from .TextVAE.constants import SOS_INDEX


class TextVAE(nn.Module):

    def __init__(self, opt, emb, vocab_size, embed_size, hidden_size, num_layers, dropout, enc_dec_tying, dec_gen_tying):
        super(TextVAE, self).__init__()
        self.opt = opt
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder = Encoder(
            opt=self.opt,
            emb=emb,
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.decoder = Decoder(
            opt=self.opt,
            vocab_size=vocab_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            weight_tying=dec_gen_tying
        )
        if enc_dec_tying:
            self.decoder.embedding.weight = self.encoder.embedding.weight
        self.mean_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.std_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.novariational_projection = nn.Linear(2 * hidden_size, hidden_size)
        self.projection = nn.Linear(hidden_size, self.opt.hidden_dim)

    def forward(self, src, hidden, trg):
        encoding, linear_hidden, packed_output, mean, std = self.encode(src, hidden)
        # logit, output_asp = self.decoder(encoding, trg)
        logit, output_asp = self.decoder(encoding, trg, linear_hidden)
        coding = self.projection(encoding)
        coding = torch.mean(coding, dim=0)
        return logit, output_asp, mean, std, coding

    def encode(self, src, hidden):
        final_states, linear_hidden, packed_ouput = self.encoder(src, hidden)
        # final_states = packed_ouput[:, -1, :].view(1, final_states.size(1), -1)
        mean = self.mean_projection(final_states)
        std = F.softplus(self.std_projection(final_states))
        sample = torch.randn(size=mean.size(), device=mean.device)
        encoding = mean + std * sample
        # encoding = self.novariational_projection
        return encoding, linear_hidden, packed_ouput, mean, std

    def sample(self, **kwargs):
        assert ('num' in kwargs) ^ ('encoding' in kwargs)
        if 'num' in kwargs:
            encoding = torch.randn(size=(self.num_layers, kwargs['num'], self.hidden_size),
                                   device=self.encoder.embedding.weight.device)
        else:
            encoding = kwargs['encoding']
        logit = self.decoder.decode(encoding, 20)
        output = logit.argmax(dim=-1)
        if 'output_encoding' in kwargs and kwargs['output_encoding']:
            num = encoding.size(1)
            encoding = encoding.transpose(0, 1).reshape(num, -1).cpu().numpy()
            return output, encoding
        else:
            return output


class Encoder(nn.Module):

    def __init__(self, opt, emb, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        # self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.opt = opt
        self.embedding = emb
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.Lin = nn.Linear(in_features=self.opt.rnn_hidden * 2, out_features=hidden_size, bias=True)
        self.dropout = dropout

    def forward(self, src, hidden):
        src_mask = (src != PAD_INDEX)
        src_lens = src_mask.long().sum(dim=1, keepdim=False)
        # src_embedding = self.embedding(src)
        zero = torch.zeros(hidden.shape[0], src.shape[1] - hidden.shape[1], hidden.shape[2]).cuda()
        src_embedding = torch.cat((hidden, zero), dim=1)
        # src_embedding = hidden.clone()
        src_embedding = self.Lin(src_embedding)

        src = F.dropout(src_embedding, p=self.dropout, training=self.training)
        src_lens, sort_index = src_lens.sort(descending=True)
        src = src.index_select(dim=0, index=sort_index)
        # packed_src = pack_padded_sequence(src, src_lens, batch_first=True)
        packed_output, final_states = self.rnn(src)
        # packed_output, final_states = self.rnn(packed_src)  # GRN encoder
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)
        # reorder_index = sort_index.argsort(descending=False)
        # output = output.index_select(dim=0, index=reorder_index)
        # final_states = final_states.index_select(dim=1, index=reorder_index)
        final_states = torch.cat(final_states.chunk(chunks=2, dim=0), dim=2)
        return final_states, src_embedding.mean(dim=1), packed_output

    def sample(self, mean, std):
        assert mean.size() == std.size()
        sample = torch.randn(size=mean.size(), device=mean.device)
        return mean + std * sample


class Decoder(nn.Module):
    def __init__(self, opt, vocab_size, embed_size, hidden_size, num_layers, dropout, weight_tying):
        super(Decoder, self).__init__()
        self.opt = opt
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.rnn_cell = MultiLayerGRUCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(hidden_size, embed_size)
        )
        self.V = torch.nn.Linear(hidden_size, hidden_size)
        self.W = torch.nn.Linear(hidden_size, hidden_size)
        self.dropoutL = torch.nn.Dropout(dropout)
        self.generator = nn.Linear(embed_size, vocab_size)
        if weight_tying:
            self.generator.weight = self.embedding.weight

    def forward(self, hidden, trg, linear_hidden):
        max_len = trg.size(1)
        logit = []
        output_asp = []
        origin_hidden = hidden.clone().sum(0)
        for i in range(max_len):
            hidden, token_logit, output = self.step(hidden, trg[:, i], origin_hidden)
            logit.append(token_logit)
            output_asp.append(output)
        logit = torch.stack(logit, dim=1)
        output_asp = torch.stack(output_asp, dim=1)
        return logit, output_asp

    def step(self, hidden, token, origin_hidden):
        # token_embedding = self.embedding(token.unsqueeze(0)).squeeze(0)
        if self.opt.decoder_type == 'self':
            hidden = self.rnn_cell(origin_hidden, hidden)
        elif self.opt.decoder_type == 'projection':
            hidden = self.rnn_cell(self.V(hidden).sum(0), self.W(hidden))
        elif self.opt.decoder_type == 'double':
            hidden = self.rnn_cell(hidden.sum(0), hidden)
        else :
            ones = torch.zeros(hidden.shape[1], hidden.shape[2]).cuda()
            hidden = self.rnn_cell(ones, hidden)


        # hidden = self.rnn_cell(token_embedding, hidden)
        # hidden = self.rnn_cell(linear_hidden, hidden)
        top_hidden = hidden[-1]
        output = self.output_projection(top_hidden)
        output = self.dropoutL(output)
        token_logit = self.generator(output)
        # token_logit = self.dropoutL(token_logit)
        return hidden, token_logit, output

    def decode(self, hidden, max_len):
        batch_size = hidden.size(1)
        token = torch.tensor([SOS_INDEX] * batch_size, dtype=torch.long, device=hidden.device)
        logit = []
        for i in range(max_len):
            hidden, token_logit = self.step(hidden, token)
            token = token_logit.argmax(dim=-1)
            logit.append(token_logit)
        logit = torch.stack(logit, dim=1)
        return logit