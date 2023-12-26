import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F


class OutfitTransformer(nn.Module):
    def __init__(self, embedding_dim=128):
        super(OutfitTransformer, self).__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=16, dim_feedforward=768, batch_first=True)
        encoder_norm = nn.LayerNorm(embedding_dim)
        self.transformer = nn.TransformerEncoder(transformer_layer, 6, encoder_norm)
        self._reset_parameters()

        self.type_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim)
        self._reset_embeddings()

        # fc layer for pretraining
        self.fc_classifier = nn.Sequential(
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, 1)
            )
        
        # fc layer for finetuning
        self.fc_embed = nn.Sequential(
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim)
            )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _reset_embeddings(self) -> None:
        initrange = 0.1
        self.type_embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, task, x, attention_mask):
        if task == 'compatibility':
            compatibility_token = torch.LongTensor([[0] for _ in range(len(x))]).to(x.device)
            compatibility_embed = self.type_embedding(compatibility_token)
            x = torch.cat([compatibility_embed, x], dim=1)

            token_mask = torch.zeros_like(compatibility_token).bool()
            attention_mask = torch.cat([token_mask, attention_mask], dim=1).bool()

            y = self.transformer(x, src_key_padding_mask=attention_mask)
            y = y[:, 0, :]
            y = F.sigmoid(self.fc_classifier(y))
        
        else:
            y = self.transformer(x, src_key_padding_mask=attention_mask)
            y = y[:, 0, :]
            y = self.fc_embed(y)
            #y = F.normalize(y, p=2, dim=1)
        
        return y