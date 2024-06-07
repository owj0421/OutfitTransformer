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


def agg_embeds(img_embeds=None, txt_embeds=None, agg_func='concat'):
    embeds = []
    if img_embeds is not None:
        embeds.append(img_embeds)
    if txt_embeds is not None:
        embeds.append(txt_embeds)
    
    if agg_func == 'concat':
        embeds = torch.cat(embeds, dim=1)
    elif agg_func == 'mean':
        embeds = torch.mean(torch.stack(embeds), dim=0)

    return embeds
    

class OutfitTransformerEmbeddingModel(nn.Module):
    def __init__(
            self,
            input_processor: FashionInputProcessor,
            hidden: int = 128,
            agg_func: Optional[Literal['concat', 'mean']] = 'concat',
            huggingface: Optional[str] = 'sentence-transformers/all-MiniLM-L6-v2',
            normalize: bool = True,
            ):
        super().__init__()

        if hidden % 2 != 0:
            assert("Embedding size should be divisible by 2!")
        # Model Settings
        self.input_processor = input_processor
        self.hidden = hidden
        self.enc_hidden = (hidden//2) if agg_func == 'concat' else hidden
        self.agg_func = agg_func
        self.normalize = normalize
        # Image Encoder
        self.img_enc = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.img_enc.fc = nn.Sequential(
            nn.Linear(self.img_enc.fc.in_features, self.enc_hidden)
            )
        # Text Encoder
        self.txt_enc = AutoModel.from_pretrained(huggingface)
        self.freeze_backbone(self.txt_enc)
        self.txt_enc_classifier = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.txt_enc.config.hidden_size, self.enc_hidden)
            )
        
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def freeze_backbone(self, model):
        for param in model.parameters():
            param.requires_grad = False
        
    def forward(self, inputs):
        return self.batch_encode(inputs)
            
    def encode(self, inputs):
        img_embeds = self.img_enc(
            inputs['image_features']
            )
        txt_embeds = self.txt_enc_classifier(self.mean_pooling(
            self.txt_enc(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']), 
            inputs['attention_mask']
            ))
        embeds = agg_embeds(img_embeds, txt_embeds, self.agg_func)

        if self.normalize:
            embeds = F.normalize(embeds, p=2, dim=1)

        return {'mask': inputs['mask'], 'embeds': embeds}
        
    def batch_encode(self, inputs):
        inputs = stack_dict(inputs)
        outputs = self.encode(inputs)

        return unstack_dict(outputs)
    

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
        
        self.hidden = hidden
        self.enc_hidden = (hidden // 2) if agg_func == 'concat' else hidden
        self.agg_func = agg_func
        self.normalize = normalize

        self.img_enc = CLIPVisionModelWithProjection.from_pretrained(huggingface)
        self.txt_enc = CLIPTextModelWithProjection.from_pretrained(huggingface)
        if linear_probing:
            self.freeze_backbone(self.img_enc)
            self.freeze_backbone(self.txt_enc)
        self.img_ffn = nn.Sequential(
            nn.Linear(self.img_enc.visual_projection.out_features, self.img_enc.visual_projection.out_features),
            nn.ReLU(),
            nn.Linear(self.img_enc.visual_projection.out_features, self.enc_hidden)
            )
        self.txt_ffn = nn.Sequential(
            nn.Linear(self.img_enc.visual_projection.out_features, self.img_enc.visual_projection.out_features),
            nn.ReLU(),
            nn.Linear(self.img_enc.visual_projection.out_features, self.enc_hidden)
            )
        
        # Input Processor
        self.input_processor = input_processor
        
    def freeze_backbone(self, model):
        for param in model.parameters():
            param.requires_grad = False
        
    def forward(self, inputs):
        return self.batch_encode(inputs)
            
    def encode(self, inputs):
        if inputs.get('image_features') is not None:
            img_embeds = self.img_enc(pixel_values=inputs['image_features']).image_embeds
            img_embeds = self.img_ffn(img_embeds)
        else:
            img_embeds = None
            
        if inputs.get('input_ids') is not None:
            txt_embeds = self.txt_enc(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).text_embeds
            txt_embeds = self.txt_ffn(txt_embeds)
        else:
            txt_embeds = None

        embeds = agg_embeds(img_embeds, txt_embeds, self.agg_func)
        if self.normalize:
            embeds = F.normalize(embeds, p=2, dim=1)

        return {'mask': inputs.get('mask', None), 'embeds': embeds}
        
    
    def batch_encode(self, inputs):
        inputs = stack_dict(inputs)
        outputs = self.encode(inputs)

        return unstack_dict(outputs)