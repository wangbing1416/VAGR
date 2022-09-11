import torch
import sys
sys.path.append(".")
from torch import nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder

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
        self.projection = nn.Linear(hidden_size, 50)

    def forward(self, src, hidden, trg):
        encoding, linear_hidden, mean, std = self.encode(src, hidden)
        # logit, output_asp = self.decoder(encoding, trg)
        logit, output_asp = self.decoder(encoding, trg, linear_hidden)
        coding = self.projection(encoding)
        coding = torch.mean(coding, dim=0)
        return logit, output_asp, mean, std, coding

    def encode(self, src, hidden):
        final_states, linear_hidden = self.encoder(src, hidden)
        mean = self.mean_projection(final_states)
        std = F.softplus(self.std_projection(final_states))
        sample = torch.randn(size=mean.size(), device=mean.device)
        encoding = mean + std * sample

        return encoding, linear_hidden, mean, std

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