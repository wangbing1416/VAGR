import torch
from torch import nn
from .multi_layer_gru_cell import MultiLayerGRUCell
from .constants import SOS_INDEX

class Decoder(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout, weight_tying):
        super(Decoder, self).__init__()
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
        self.dropoutL = torch.nn.Dropout(dropout)
        self.generator = nn.Linear(embed_size, vocab_size)
        if weight_tying:
            self.generator.weight = self.embedding.weight

    def forward(self, hidden, trg, linear_hidden):
        max_len = trg.size(1)
        logit = []
        output_asp = []
        for i in range(max_len):
            hidden, token_logit, output = self.step(hidden, trg[:, i], linear_hidden)
            logit.append(token_logit)
            output_asp.append(output)
        logit = torch.stack(logit, dim=1)
        output_asp = torch.stack(output_asp, dim=1)
        return logit, output_asp

    def step(self, hidden, token, linear_hidden):
        token_embedding = self.embedding(token.unsqueeze(0)).squeeze(0)
        hidden = self.rnn_cell(token_embedding, hidden)
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