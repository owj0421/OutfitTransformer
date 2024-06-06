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
            n_layers: int = 6,
            n_heads: int = 16
            ):
        super().__init__()
        self.embedding_model = embedding_model

        # Hyperparameters
        self.hidden = embedding_model.hidden
        self.ffn_hidden = 2048 # self.hidden * 4
        self.n_layers = n_layers
        self.n_heads = n_heads

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden,
            nhead=n_heads,
            dim_feedforward=self.ffn_hidden,
            batch_first=True
            )
        self.transformer=nn.TransformerEncoder(
            encoder_layer=encoder_layer, 
            num_layers=n_layers,
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
        self.mlp = nn.ModuleDict({
            '<cp>': nn.Sequential(
                nn.Linear(self.hidden, self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, 1),
                nn.Sigmoid()
                ),
            '<cir>': nn.Sequential(
                nn.Linear(self.hidden, self.hidden),
                nn.Tanh(),
                nn.Linear(self.hidden, self.hidden)
                )
            })
    
    def encode(self, inputs):
        return self.embedding_model.encode(inputs)
    
    def batch_encode(self, inputs):
        return self.embedding_model.batch_encode(inputs)
    
    def calculate_compatibility(self, item_embeddings):
        task = '<cp>'

        mask, embeds = item_embeddings.values()
        n_outfit, *_ = embeds.shape
        
        task_id = torch.LongTensor([self.task2id[task] for _ in range(n_outfit)]).to(embeds.device)
        prefix_embed = self.task_embeddings(task_id).unsqueeze(1)
        prefix_mask = torch.zeros((n_outfit, 1), dtype=torch.bool).to(embeds.device)
        
        embeds = torch.concat([prefix_embed, embeds], dim=1)
        mask = torch.concat([prefix_mask, mask], dim=1)
        
        outputs = self.transformer(embeds, src_key_padding_mask=mask)[:, 0, :]
        outputs = self.mlp[task](outputs)
        
        return outputs
    
    def get_cir_embedding(self, item_embeddings, 
                          query_inputs: Dict[Literal['input_ids', 'attention_mask'], Any]):
        task = '<cir>'
        
        mask, embeds = item_embeddings.values()
        n_outfit, *_ = embeds.shape
        
        task_id = torch.LongTensor([self.task2id[task] for _ in range(n_outfit)]).to(embeds.device)
        if self.embedding_model.agg_func == 'mean':
            prefix_embed = self.task_embeddings(task_id) + self.embedding_model.encode(query_inputs)['embeds']
        elif self.embedding_model.agg_func == 'concat':
            prefix_embed = torch.concat([
                self.task_embeddings(task_id)[:self.embedding_model.encoder_hidden, :],
                self.embedding_model.encode(query_inputs)['embeds']
                ], dim=1)
        prefix_embed = prefix_embed.unsqueeze(1)
        prefix_mask = torch.zeros((n_outfit, 1), dtype=torch.bool).to(embeds.device)
        
        embeds = torch.concat([prefix_embed, embeds], dim=1)
        mask = torch.concat([prefix_mask, mask], dim=1)
        
        outputs = self.transformer(embeds, src_key_padding_mask=mask)[:, 0, :]
        outputs = self.mlp[task](outputs)
        
        return outputs