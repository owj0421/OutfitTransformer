import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet18
from transformers import AutoModel
from torch import Tensor
from typing import List, Tuple
import PIL
    

class ImageEncoder(nn.Module):
    def __init__(
            self,
            embedding_dim=64
            ):
        super().__init__()
        self.model = resnet18(pretrained=True)
        input_features_fc_layer = self.model.fc.in_features
        self.model.fc = nn.Linear(input_features_fc_layer, embedding_dim, bias=False)
        
    def forward(self, images):
        embed = self.model(images)
        embed = F.normalize(embed, p=2, dim=1)
        
        return embed
    

class TextEncoder(nn.Module):
    def __init__(
            self,
            embedding_dim: int = 64,
            huggingface_model: str='sentence-transformers/paraphrase-albert-small-v2'
            ):
        super().__init__()
        self.model = AutoModel.from_pretrained(huggingface_model)
        for param in self.model.parameters():
            param.requires_grad = False

        self.fc_embed = nn.Sequential(
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(768, embedding_dim)
            )

    def _mean_pooling(self, model_output, attention_mask):
        token_embeds = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()

        return torch.sum(token_embeds * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, input_ids, attention_mask):
        model_output = self.model(input_ids, attention_mask)
        model_output = self._mean_pooling(model_output, attention_mask)
        embed = self.fc_embed(model_output)
        embed = F.normalize(embed, p=2, dim=1)

        return embed
    

class ItemEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.txt_encoder = TextEncoder(embedding_dim=embedding_dim // 2)
        self.img_encoder = ImageEncoder(embedding_dim=embedding_dim // 2)

    def forward(self, img, input_ids=None, attention_mask=None):
        img_embed = self.img_encoder(img)
        txt_embed = self.txt_encoder(input_ids, attention_mask)
        embed = torch.concat([img_embed, txt_embed], dim=-1)
        # embed = F.normalize(embed, p=2, dim=1)

        return embed