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


def agg_embeds(img_embeds, txt_embeds, agg_func):
    if img_embeds is not None:
        if txt_embeds is not None:
            if agg_func == 'concat':
                embeds = torch.cat([img_embeds, txt_embeds], dim=1)
            elif agg_func == 'mean':
                embeds = (img_embeds + txt_embeds) / 2
        else:
            embeds = img_embeds
    else:
        embeds = txt_embeds
    
    return embeds
    

class OutfitTransformerEmbeddingModel(nn.Module):
    def __init__(
            self,
            input_processor: FashionInputProcessor,
            hidden: int = 128,
            agg_func: Optional[Literal['concat', 'mean']] = 'concat',
            huggingface: Optional[str] = 'sentence-transformers/all-MiniLM-L6-v2',
            linear_probing: bool = True,
            normalize: bool = True,
            ):
        super().__init__()

        if hidden % 2 != 0:
            assert("Embedding size should be divisible by 2!")
        
        self.hidden = hidden
        self.encoder_hidden = (hidden // 2) if agg_func == 'concat' else hidden
        self.agg_func = agg_func
        self.normalize = normalize

        self.img_encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.img_encoder.fc = nn.Sequential(
            nn.Linear(self.img_encoder.fc.in_features, self.img_encoder.fc.in_features),
            nn.ReLU(),
            nn.Linear(self.img_encoder.fc.in_features, self.encoder_hidden)
            )

        self.txt_encoder = AutoModel.from_pretrained(huggingface)
        if linear_probing:
            self.freeze_backbone(self.txt_encoder)
        self.ffn = nn.Sequential(
            nn.Linear(self.txt_encoder.config.hidden_size, self.txt_encoder.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.txt_encoder.config.hidden_size, self.encoder_hidden)
            )
        
        # Input Processor
        self.input_processor = input_processor
        
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
        if inputs.get('image_features') is not None:
            img_embeds = self.img_encoder(inputs['image_features'])
        else:
            img_embeds = None
            
        if inputs.get('input_ids') is not None:     
            model_output = self.txt_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            txt_embeds = self.mean_pooling(model_output , inputs['attention_mask'])
            txt_embeds = self.ffn(txt_embeds)
        else:
            txt_embeds = None

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
        self.encoder_hidden = (hidden // 2) if agg_func == 'concat' else hidden
        self.agg_func = agg_func
        self.normalize = normalize

        self.img_encoder = CLIPVisionModelWithProjection.from_pretrained(huggingface)
        self.txt_encoder = CLIPTextModelWithProjection.from_pretrained(huggingface)
        if linear_probing:
            self.freeze_backbone(self.img_encoder)
            self.freeze_backbone(self.txt_encoder)
        self.img_ffn = nn.Sequential(
            nn.Linear(self.img_encoder.visual_projection.out_features, self.img_encoder.visual_projection.out_features),
            nn.ReLU(),
            nn.Linear(self.img_encoder.visual_projection.out_features, self.encoder_hidden)
            )
        self.txt_ffn = nn.Sequential(
            nn.Linear(self.img_encoder.visual_projection.out_features, self.img_encoder.visual_projection.out_features),
            nn.ReLU(),
            nn.Linear(self.img_encoder.visual_projection.out_features, self.encoder_hidden)
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
            img_embeds = self.img_encoder(pixel_values=inputs['image_features']).image_embeds
            img_embeds = self.img_ffn(img_embeds)
        else:
            img_embeds = None
            
        if inputs.get('input_ids') is not None:
            txt_embeds = self.txt_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).text_embeds
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