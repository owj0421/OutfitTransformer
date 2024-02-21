# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class BaseEncoder(nn.Module):
    def __init__(
            self,
            embed_dim: int = 64,
            do_linear_probing: bool = False,
            normalize: bool = False
            ):
        super().__init__()
        self.model = None
        self.embed_dim = embed_dim
        self.do_linear_probing = do_linear_probing
        self.normalize = normalize

    def freeze_backbone(self):
        pass
        
    def forward(self, input_ids, attention_mask):
        pass


class BertEncoder(BaseEncoder):
    def __init__(
            self,
            huggingface: str = 'sentence-transformers/paraphrase-albert-small-v2',
            embed_dim: int = 64,
            do_linear_probing: bool = True,
            normalize: bool = False
            ):
        super().__init__(embed_dim, do_linear_probing, normalize)
        self.model = AutoModel.from_pretrained(huggingface)

        self.fc = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, embed_dim)
            )
        
        if do_linear_probing:
            self.freeze_backbone()
    
    def freeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = False
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def forward(self, input_ids, attention_mask):
        model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = self.mean_pooling(model_output, attention_mask)
        fc_output = self.fc(sentence_embeddings)
        if self.normalize:
            fc_output = F.normalize(fc_output, p=2, dim=1)
        return fc_output