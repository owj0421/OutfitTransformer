# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import os
import math
from tqdm import tqdm
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils.utils import *
from src.models.embedder import *
from src.loss.focal_loss import focal_loss
from src.loss.triplet_loss import triplet_loss


class RecommendationModel(nn.Module):
    
    def __init__(
            self,
            embedding_model: Any,# CLIPEmbeddingModel | OutfitTransformerEmbeddingModel,
            ffn_hidden: int = 2024,
            n_layers: int = 6,
            n_heads: int = 16
            ):
        super().__init__()
        self.embedding_model = embedding_model

        # Hyperparameters
        self.hidden = embedding_model.hidden
        self.ffn_hidden = ffn_hidden
        self.n_layers = n_layers
        self.n_heads = n_heads

        # Transformer
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden,
            nhead=self.n_heads,
            dim_feedforward=self.ffn_hidden,
            batch_first=True
            )
        self.transformer=nn.TransformerEncoder(
            encoder_layer=transformer_layer, 
            num_layers=self.n_layers,
            norm=nn.LayerNorm(self.hidden),
            enable_nested_tensor=False
            )
        
        # Task-specific embeddings
        self.task = ['<cp>', '<cir>']
        self.task2id = {task: idx for idx, task in enumerate(self.task)}
        self.task_embeddings = nn.Embedding(
            num_embeddings=len(self.task), 
            embedding_dim=self.hidden,
            max_norm=1 if self.embedding_model.normalize else None
            )
        
        # Task-specific MLP
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.hidden, 1),
            nn.Sigmoid()
            )
    
    def encode(self, inputs):
        return self.embedding_model.encode(inputs)
    
    def batch_encode(self, inputs):
        return self.embedding_model.batch_encode(inputs)
    
    def get_score(
            self, 
            item_embeddings
            ):
        task = '<cp>'

        mask, embeds = item_embeddings.values()
        n_outfit, *_ = embeds.shape
        
        task_id = torch.LongTensor([self.task2id[task] for _ in range(n_outfit)]).to(embeds.device)
        prefix_embed = self.task_embeddings(task_id).unsqueeze(1)
        prefix_mask = torch.zeros((n_outfit, 1), dtype=torch.bool).to(embeds.device)
        
        embeds = torch.cat([prefix_embed, embeds], dim=1)
        mask = torch.cat([prefix_mask, mask], dim=1)
        
        outputs = self.transformer(embeds, src_key_padding_mask=mask)[:, 0, :]
        outputs = self.classifier(outputs)
        
        return outputs
    
    def get_embedding(
            self, 
            item_embeddings, 
            query_inputs = None,
            normalize: bool = True
            ):
        task = '<cir>'
        
        mask, embeds = item_embeddings.values()
        n_outfit, *_ = embeds.shape
        
        if query_inputs is not None:
            task_id = torch.LongTensor([self.task2id[task] for _ in range(n_outfit)]).to(embeds.device)
            task_embedding = self.task_embeddings(task_id).unsqueeze(1)
            query_embedding = self.embedding_model.batch_encode(query_inputs)['embeds']
            
            if self.embedding_model.agg_func == 'mean':
                prefix_embed = task_embedding + query_embedding
            elif self.embedding_model.agg_func == 'concat':
                task_embedding = task_embedding[:, :, :self.embedding_model.encoder_hidden]
                prefix_embed = torch.cat([task_embedding, query_embedding], dim=2)

            prefix_mask = torch.zeros((n_outfit, 1), dtype=torch.bool).to(embeds.device)
            
            embeds = torch.cat([prefix_embed, embeds], dim=1)
            mask = torch.cat([prefix_mask, mask], dim=1)
            
        else:
            task_id = torch.LongTensor([self.task2id[task] for _ in range(n_outfit)]).to(embeds.device)
            prefix_embed = self.task_embeddings(task_id).unsqueeze(1)
            prefix_mask = torch.zeros((n_outfit, 1), dtype=torch.bool).to(embeds.device)
            
            embeds = torch.cat([prefix_embed, embeds], dim=1)
            mask = torch.cat([prefix_mask, mask], dim=1)
            
        outputs = self.transformer(embeds, src_key_padding_mask=mask)[:, 0, :]
        
        if normalize:
            outputs = F.normalize(outputs, p=2, dim=-1)
        
        return outputs