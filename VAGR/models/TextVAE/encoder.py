import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .constants import PAD_INDEX

class Encoder(nn.Module):

    def __init__(self, opt, emb, vocab_size, embed_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        # self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)
        self.opt = opt
        self.embedding = emb
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.Lin = nn.Linear(in_features=self.opt.rnn_hidden * 2, out_features=embed_size, bias=True)
        self.dropout = dropout

    def forward(self, src, hidden):
        src_mask = (src != PAD_INDEX)
        src_lens = src_mask.long().sum(dim=1, keepdim=False)
        # src_embedding = self.embedding(src)
        zero = torch.zeros(hidden.shape[0], src.shape[1] - hidden.shape[1], hidden.shape[2]).cuda()
        src_embedding = torch.cat((hidden, zero), dim=1)
        src_embedding = self.Lin(src_embedding)
        src = F.dropout(src_embedding, p=self.dropout, training=self.training)
        src_lens, sort_index = src_lens.sort(descending=True)
        src = src.index_select(dim=0, index=sort_index)
        # packed_src = pack_padded_sequence(src, src_lens, batch_first=True)
        packed_output, final_states = self.rnn(src)
        # packed_output, final_states = self.rnn(packed_src)  # GRN encoder
        # output, _ = pad_packed_sequence(packed_output, batch_first=True)
        reorder_index = sort_index.argsort(descending=False)
        # output = output.index_select(dim=0, index=reorder_index)
        final_states = final_states.index_select(dim=1, index=reorder_index)
        final_states = torch.cat(final_states.chunk(chunks=2, dim=0), dim=2)
        return final_states, src_embedding.mean(dim=1)

    def sample(self, mean, std):
        assert mean.size() == std.size()
        sample = torch.randn(size=mean.size(), device=mean.device)
        return mean + std * sample