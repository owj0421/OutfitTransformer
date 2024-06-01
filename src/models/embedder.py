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

class CLIPEmbeddingModel(nn.Module):
    def __init__(
            self,
            input_processor: FashionInputProcessor,
            hidden: int = 128,
            agg_func: Optional[Literal['concat', 'mean']] = 'mean',
            huggingface: Optional[str] = 'patrickjohncyh/fashion-clip',
            fp16: bool = True,
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
        if fp16 and torch.cuda.is_available():
            self.img_encoder = self.img_encoder.half()
            self.txt_encoder = self.txt_encoder.half()
        if linear_probing:
            self.freeze_backbone(self.img_encoder)
            self.freeze_backbone(self.txt_encoder)
        
        self.ffn = nn.Sequential(
            nn.Linear(self.img_encoder.visual_projection.out_features, self.encoder_hidden),
            nn.ReLU(),
            nn.Linear(self.encoder_hidden, self.encoder_hidden)
            )
        
        # Input Processor
        self.input_processor = input_processor
        self.token2id = input_processor.token2id
        self.vocab_size = len(input_processor.id2token)
        
        self.category_embeddings = nn.Embedding(
            num_embeddings=self.vocab_size, 
            embedding_dim=self.hidden, 
            padding_idx=self.token2id['<pad>'],
            max_norm=1
            )
        
    def freeze_backbone(self, model):
        for param in model.parameters():
            param.requires_grad = False
        
    def forward(self, inputs):
        return self.batch_encode(inputs)
            
    def encode(self, inputs):

        if inputs['image_features'] is not None:
            use_image = True
            img_embeds = self.img_encoder(pixel_values=inputs['image_features']).image_embeds
            img_embeds = img_embeds.float()
            
        if inputs['input_ids'] is not None:
            use_text = True
            txt_embeds = self.txt_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).text_embeds
            txt_embeds = txt_embeds.float()

        if use_image:
            if use_text:
                if self.agg_func == 'concat':
                    embeds = torch.cat([img_embeds, txt_embeds], dim=1)
                elif self.agg_func == 'mean':
                    embeds = (img_embeds + txt_embeds) / 2
            else:
                embeds = img_embeds
        else:
            embeds = txt_embeds

        embeds = self.ffn(embeds)

        if self.normalize:
            embeds = F.normalize(embeds, p=2, dim=1)
            
        # if inputs['category_ids'] is not None:
        #     category_embeds = self.category_embeddings(inputs['category_ids'])
        #     embeds = embeds + category_embeds

        return {'mask': inputs.get('mask', None), 'embeds': embeds}
        
    
    def batch_encode(self, inputs):
        inputs = stack_dict(inputs)
        outputs = self.encode(inputs)

        return unstack_dict(outputs)
    

class OutfitTransformerEmbeddingModel(nn.Module):
    def __init__(
            self,
            input_processor: FashionInputProcessor,
            hidden: int = 128,
            agg_func: Optional[Literal['concat', 'mean']] = 'concat',
            huggingface: Optional[str] = 'sentence-transformers/all-MiniLM-L6-v2',
            fp16: bool = True,
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
        self.img_encoder.fc = nn.Linear(self.img_encoder.fc.in_features, self.encoder_hidden)

        self.txt_encoder = AutoModel.from_pretrained(huggingface)
        if fp16 and torch.cuda.is_available():
            self.txt_encoder = self.txt_encoder.half()
        if linear_probing:
            self.freeze_backbone(self.txt_encoder)
        self.ffn = nn.Linear(384, self.encoder_hidden)
        
        # Input Processor
        self.input_processor = input_processor
        self.token2id = input_processor.token2id
        self.vocab_size = len(input_processor.id2token)
        
        self.category_embeddings = nn.Embedding(
            num_embeddings=self.vocab_size, 
            embedding_dim=hidden, 
            padding_idx=self.token2id['<pad>'],
            max_norm=1
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
        if inputs['image_features'] is not None:
            use_image = True
            img_embeds = self.img_encoder(inputs['image_features'])
            img_embeds = img_embeds.float()
            
        if inputs['input_ids'] is not None:
            use_text = True        
            model_output = self.txt_encoder(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            txt_embeds = self.mean_pooling(model_output , inputs['attention_mask']).float()
            txt_embeds = self.ffn(txt_embeds)

        if use_image:
            if use_text:
                if self.agg_func == 'concat':
                    embeds = torch.cat([img_embeds, txt_embeds], dim=1)
                elif self.agg_func == 'mean':
                    embeds = (img_embeds + txt_embeds) / 2
            else:
                embeds = img_embeds
        else:
            embeds = txt_embeds

        if self.normalize:
            embeds = F.normalize(embeds, p=2, dim=1)
            
        # if inputs['category_ids'] is not None:
        #     category_embeds = self.category_embeddings(inputs['category_ids'])
        #     embeds = embeds + category_embeds

        return {'mask': inputs['mask'], 'embeds': embeds}
        
    
    def batch_encode(self, inputs):
        inputs = stack_dict(inputs)
        outputs = self.encode(inputs)

        return unstack_dict(outputs)