# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from transformers import AutoModel, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from src.utils.utils import *
from typing import Literal
from src.datasets.processor import FashionInputProcessor


def agg_embeds(image_embeds=None, text_embeds=None, agg_func='concat'):
    embeds = []
    if image_embeds is not None:
        embeds.append(image_embeds)
    if text_embeds is not None:
        embeds.append(text_embeds)
    
    if agg_func == 'concat':
        embeds = torch.cat(embeds, dim=1)
    elif agg_func == 'mean':
        embeds = torch.mean(torch.stack(embeds), dim=0)

    return embeds


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


class OutfitTransformerEmbeddingModel(nn.Module):
    def __init__(
            self,
            input_processor: FashionInputProcessor,
            hidden: int = 128,
            agg_func: Optional[Literal['concat', 'mean']] = 'concat',
            huggingface: Optional[str] = 'sentence-transformers/all-MiniLM-L12-v2',
            normalize: bool = True,
            ):
        super().__init__()

        if hidden % 2 != 0:
            assert("Embedding size should be divisible by 2!")
            
        # Model Settings
        self.input_processor = input_processor
        self.hidden = hidden
        self.encoder_hidden = (hidden//2) if agg_func == 'concat' else hidden
        self.agg_func = agg_func
        self.normalize = normalize
        
        # Image Encoder
        self.image_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.image_encoder.fc = nn.Linear(self.image_encoder.fc.in_features, self.encoder_hidden, bias=False)
        
        # Text Encoder
        self.text_encoder = AutoModel.from_pretrained(huggingface)
        self.text_projection = nn.Parameter(torch.empty(self.text_encoder.config.hidden_size, self.encoder_hidden))
        freeze_model(self.text_encoder)
        nn.init.normal_(self.text_projection, std=self.text_encoder.config.hidden_size ** -0.5)
            
    def encode(self, inputs):
        if 'image_features' in inputs:
            image_embeds = self.image_encoder(inputs['image_features'])
        else:
            image_embeds = None
            
        if 'input_ids' in inputs:
            text_outputs = self.text_encoder(inputs['input_ids'], inputs['attention_mask'])
            text_embeds = mean_pooling(text_outputs, inputs['attention_mask']) @ self.text_projection
        else:
            text_embeds = None
            
        embeds = agg_embeds(image_embeds, text_embeds, self.agg_func)
        if self.normalize:
            embeds = F.normalize(embeds, p=2, dim=1)
        return {'mask': inputs['mask'] if inputs.get('mask') is not None else None, 'embeds': embeds}
        
    def batch_encode(self, inputs):
        inputs = stack_dict(inputs)
        outputs = self.encode(inputs)
        return unstack_dict(outputs)
    
    def forward(self, inputs):
        return self.batch_encode(inputs)


class CLIPEmbeddingModel(nn.Module):
    def __init__(
            self,
            input_processor: FashionInputProcessor,
            hidden: int = 128,
            agg_func: Optional[Literal['concat', 'mean']] = 'concat',
            huggingface: Optional[str] = 'patrickjohncyh/fashion-clip',
            linear_probing: bool = True,
            normalize: bool = True,
            ):
        super().__init__()

        if hidden % 2 != 0:
            assert("Embedding size should be divisible by 2!")

        # Model Settings
        self.input_processor = input_processor
        self.hidden = hidden
        self.encoder_hidden = (hidden//2) if agg_func == 'concat' else hidden
        self.agg_func = agg_func
        self.normalize = normalize

        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(huggingface)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(huggingface)
        if linear_probing:
            freeze_model(self.image_encoder)
            freeze_model(self.text_encoder)
        self.img_ffn = nn.Sequential(
            nn.Linear(self.image_encoder.visual_projection.out_features, self.image_encoder.visual_projection.out_features),
            nn.ReLU(),
            nn.Linear(self.image_encoder.visual_projection.out_features, self.encoder_hidden)
            )
        self.txt_ffn = nn.Sequential(
            nn.Linear(self.image_encoder.visual_projection.out_features, self.image_encoder.visual_projection.out_features),
            nn.ReLU(),
            nn.Linear(self.image_encoder.visual_projection.out_features, self.encoder_hidden)
            )
        
        
    def forward(self, inputs):
        return self.batch_encode(inputs)
            
    def encode(self, inputs):
        if inputs.get('image_features') is not None:
            image_embeds = self.image_encoder(pixel_values=inputs['image_features']).image_embeds
            image_embeds = self.img_ffn(image_embeds)
        else:
            image_embeds = None
            
        if inputs.get('input_ids') is not None:
            text_embeds = self.text_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).text_embeds
            text_embeds = self.txt_ffn(text_embeds)
        else:
            text_embeds = None

        embeds = agg_embeds(image_embeds, text_embeds, self.agg_func)
        
        if self.normalize:
            embeds = F.normalize(embeds, p=2, dim=1)

        return {'mask': inputs.get('mask', None), 'embeds': embeds}
        
    def batch_encode(self, inputs):
        inputs = stack_dict(inputs)
        outputs = self.encode(inputs)

        return unstack_dict(outputs)