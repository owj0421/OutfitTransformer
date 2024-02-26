# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
Reference:
    [1] Rohan Sarkar, Navaneeth Bodla, et al. Outfit Transformer : Outfit Representations for Fashion Recommendation. CVPR, 2023. 
    (https://arxiv.org/abs/2204.04812)
"""
import os
import math
import wandb
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

from outfit_transformer.utils.utils import *
from outfit_transformer.models.encoder.builder import *
from outfit_transformer.loss.focal_loss import focal_loss
from outfit_transformer.loss.triplet_loss import triplet_loss


class OutfitTransformer(nn.Module):
    
    def __init__(
            self,
            embedding_dim: int = 128,
            img_backbone: str = 'resnet-18',
            txt_backbone: str = 'bert',
            txt_huggingface: str = 'sentence-transformers/paraphrase-albert-small-v2',
            nhead = 16,
            dim_feedforward = 512, # Not specified; use 4x the built-in size the same as BERT
            num_layers = 6
            ):
        super().__init__()
        self.encode_dim = embedding_dim // 2
        self.embedding_dim = embedding_dim
        #------------------------------------------------------------------------------------------------#
        # Encoder
        self.img_encoder = build_img_encoder(
            backbone = img_backbone, 
            embedding_dim = self.encode_dim
            )
        self.txt_encoder = build_txt_encoder(
            backbone = txt_backbone, 
            embedding_dim = self.encode_dim, 
            huggingface = txt_huggingface, 
            do_linear_probing = True
            )
        #------------------------------------------------------------------------------------------------#
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model = self.embedding_dim,
            nhead = nhead,
            dim_feedforward = dim_feedforward,
            batch_first = True
            )
        self.transformer = nn.TransformerEncoder(
            encoder_layer = encoder_layer, 
            num_layers = num_layers, 
            norm = nn.LayerNorm(self.embedding_dim)
            )
        #------------------------------------------------------------------------------------------------#
        # Embeddings
        # For CP
        self.cp_embedding = nn.Parameter(torch.empty((1, self.embedding_dim), dtype=torch.float32))
        nn.init.xavier_uniform_(self.cp_embedding.data)
        # For CIR
        # This part is different from the original paper. I replaced it with token instead of image.
        self.cir_embedding = nn.Parameter(torch.empty((1, self.encode_dim), dtype=torch.float32))
        nn.init.xavier_uniform_(self.cir_embedding.data)
        #------------------------------------------------------------------------------------------------#
        # FC layers for classification(for pre-training)
        self.fc_classifier = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, 1)
            )
        #------------------------------------------------------------------------------------------------#
        # FC layers for projection(for fine-tuning)
        self.fc_projection = nn.Sequential(
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim)
            )
        #------------------------------------------------------------------------------------------------#


    def encode(self, inputs):
        inputs = stack_dict(inputs)
        img_embedddings = self.img_encoder(inputs['image_features'])
        txt_embedddings = self.txt_encoder(
            input_ids = inputs['input_ids'],
            attention_mask = inputs['attention_mask']
            )
        general_embeddings = torch.concat([img_embedddings, txt_embedddings], dim=-1)
        outputs = {
            'mask': inputs['mask'],
            'embed': general_embeddings
            }
        return unstack_dict(outputs)
    

    def iteration_step(self, batch, task, device):
        if task == 'cp':
            targets = batch['targets'].to(device)
            inputs = {key: value.to(device) for key, value in batch['inputs'].items()}
            x = self.encode(inputs)
            embed = torch.cat([
                self.cp_embedding.unsqueeze(0).expand(len(x['embed']), -1, -1),
                x['embed']
                ], dim=1)
            src_key_padding_mask = torch.cat([
                torch.zeros((len(x['mask']), 1), device=embed.device), 
                x['mask']
                ], dim=1).bool()
            y = self.transformer(embed, src_key_padding_mask=src_key_padding_mask)[:, 0, :]
            y = self.fc_classifier(y)
            logits = F.sigmoid(y)
            loss = focal_loss(logits, targets.to(device))
        elif task == 'cir':
            # Randomly extract the number of items to be used as query and answer.
            n_outfit_per_batch = torch.sum(~batch['outfits']['mask'], dim=1).numpy()
            target_item_idx = torch.LongTensor(np.random.randint(low=0, high=n_outfit_per_batch))
            # Encode
            inputs = {key: value.to(device) for key, value in batch['outfits'].items()}
            x = self.encode(inputs)
            # Extract and copy items to use as positives for triplet loss.
            target_item_embeddings = x['embed'][range(len(target_item_idx)), target_item_idx].clone()
            # Change the front part of embedding at the location corresponding to `target_item_idx` to `cir_embedding`
            # The permutation variable nature of the transformer structure does not interfere with the results.
            x['embed'][range(len(target_item_idx)), target_item_idx, :self.encode_dim] = \
                x['embed'][range(len(target_item_idx)), target_item_idx, :self.encode_dim] * 0
            x['embed'][range(len(target_item_idx)), target_item_idx, :self.encode_dim] = \
                x['embed'][range(len(target_item_idx)), target_item_idx, :self.encode_dim] + self.cir_embedding.expand(len(x['embed']), -1)
            # Take the output of the location corresponding to `target_item_idx`
            # As above reason, results are not interfere with the results.
            y = self.transformer(x['embed'], src_key_padding_mask=x['mask'].bool())[range(len(target_item_idx)), target_item_idx, :]
            y = self.fc_projection(y)
            # Margin is same as paper(2)
            # Use both batch all, hard strategy and added them.
            loss = triplet_loss(y, target_item_embeddings, margin=2, method='both')

        return loss


    def iteration(self, dataloader, epoch, is_train, device,
                  optimizer=None, scheduler=None, use_wandb=False, task='cp'):
        type_str = f'{task} train' if is_train else f'{task} valid'
        epoch_iterator = tqdm(dataloader)
        total_loss = 0.
        for iter, batch in enumerate(epoch_iterator, start=1):
            #--------------------------------------------------------------------------------------------#
            # Compute Loss
            running_loss = self.iteration_step(batch, task, device)
            #--------------------------------------------------------------------------------------------#
            # Backward
            if is_train == True:
                optimizer.zero_grad()
                running_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
                optimizer.step()
                if scheduler:
                    scheduler.step()
            #--------------------------------------------------------------------------------------------#
            # Logging
            total_loss += running_loss.item()
            epoch_iterator.set_description(
                f'[{type_str}] Epoch: {epoch + 1:03} | Loss: {running_loss:.5f}')
            if use_wandb:
                log = {
                    f'{type_str}_loss': running_loss, 
                    f'{type_str}_step': epoch * len(epoch_iterator) + iter
                    }
                if is_train == True:
                    log["learning_rate"] = scheduler.get_last_lr()[0]
                wandb.log(log)
            #--------------------------------------------------------------------------------------------#
        total_loss = total_loss / iter
        print( f'[{type_str} END] Epoch: {epoch + 1:03} | loss: {total_loss:.5f} ' + '\n')
        return total_loss
    
    
    def fit(
            self,
            task: Literal['cp', 'cir'],
            save_dir: str,
            n_epochs: int,
            optimizer: torch.optim.Optimizer,
            train_dataloader: DataLoader,
            valid_dataloader: DataLoader,
            valid_cp_dataloader: Optional[DataLoader] = None,
            valid_fitb_dataloader: Optional[DataLoader] = None,
            valid_cir_dataloader: Optional[DataLoader] = None,
            scheduler: Optional[torch.optim.lr_scheduler.StepLR] = None,
            device: Literal['cuda', 'cpu'] = 'cuda',
            use_wandb: Optional[bool] = False,
            save_every: Optional[int] = 1
            ):
        
        date = datetime.now().strftime('%Y-%m-%d')
        save_dir = os.path.join(save_dir, date)

        device = torch.device(0) if device == 'cuda' else torch.device(1)
        self.to(device)

        best_criterion = -np.inf
        best_model = None

        for epoch in range(n_epochs):
            #--------------------------------------------------------------------------------------------#
            self.train()
            train_loss = self.iteration(
                dataloader = train_dataloader, 
                epoch = epoch, 
                is_train = True, 
                device = device,
                optimizer = optimizer, 
                scheduler = scheduler, 
                use_wandb = use_wandb,
                task = task
                )
            #--------------------------------------------------------------------------------------------#
            self.eval()
            with torch.no_grad():
                valid_loss = self.iteration(
                    dataloader = valid_dataloader, 
                    epoch = epoch, 
                    is_train = False,
                    device = device,
                    use_wandb = use_wandb,
                    task = task
                    )
                #----------------------------------------------------------------------------------------#
                if task == 'cp':
                    cp_score = self.cp_evaluation(
                        dataloader = valid_cp_dataloader,
                        epoch = epoch,
                        is_test = False,
                        device = device,
                        use_wandb = use_wandb
                        )
                    fitb_score = self.fitb_evaluation(
                        dataloader = valid_fitb_dataloader,
                        epoch = epoch,
                        is_test = False,
                        device = device,
                        use_wandb = use_wandb
                        )
                    if cp_score > best_criterion:
                        best_criterion = cp_score
                        best_state = deepcopy(self.state_dict())
                #----------------------------------------------------------------------------------------#
                # Evaluation methods for CIR is not yet implemented
                # elif task == 'cir':
                #     cir_score = self.cir_evaluation(
                #         dataloader = valid_cir_dataloader,
                #         epoch = epoch,
                #         is_test = False,
                #         device = device,
                #         use_wandb = use_wandb
                #         )
                #     if cir_score > best_criterion:
                #         best_criterion = cir_score
                #         best_state = deepcopy(self.state_dict())
                #----------------------------------------------------------------------------------------#
            if epoch % save_every == 0:
                model_name = f'{epoch}_{best_criterion:.3f}'
                self._save(save_dir, model_name)
        self._save(save_dir, 'final', best_state)


    def _save(self, 
              dir, 
              model_name, 
              best_state=None):
        
        def _create_folder(dir):
            try:
                if not os.path.exists(dir):
                    os.makedirs(dir)
            except OSError:
                print('[Error] Creating directory.' + dir)

        _create_folder(dir)
        path = os.path.join(dir, f'{model_name}.pth')
        checkpoint = {'state_dict': best_state if best_state is not None else self.state_dict()}
        torch.save(checkpoint, path)
        print(f'[COMPLETE] Save at {path}')


    def cp_evaluation(self, dataloader, epoch, is_test, device, use_wandb=False):
        type_str = 'cp_test' if is_test else 'cp_eval'
        epoch_iterator = tqdm(dataloader)
        
        total_targets, total_score, total_pred = [], [], []
        for iter, batch in enumerate(epoch_iterator, start=1):
            inputs = {key: value.to(device) for key, value in batch['inputs'].items()}
            logits = self.forward(inputs, 'cp')

            targets = batch['targets'].view(-1).detach().numpy().astype(int)
            run_score = logits.view(-1).cpu().detach().numpy()
            run_pred = (run_score >= 0.5).astype(int)

            total_targets = np.concatenate([total_targets, targets], axis=None)
            total_score = np.concatenate([total_score, run_score], axis=None)
            total_pred = np.concatenate([total_pred, run_pred], axis=None)
            #--------------------------------------------------------------------------------------------#
            # Logging
            run_acc = sum(targets == run_pred) / len(targets)
            epoch_iterator.set_description(f'[{type_str}] Epoch: {epoch + 1:03} | Acc: {run_acc:.5f}')
            if use_wandb:
                log = {
                    f'{type_str}_acc': run_acc, 
                    f'{type_str}_step': epoch * len(epoch_iterator) + iter
                    }
                wandb.log(log)
            #--------------------------------------------------------------------------------------------#
        total_acc = sum(total_targets == total_pred) / len(total_targets)
        total_auc = roc_auc_score(total_targets, total_score)
        print(f'[{type_str} END] Epoch: {epoch + 1:03} | Acc: {total_acc:.5f} | AUC: {total_auc:.5f}\n')
        return total_auc
    

    def fitb_evaluation(self, dataloader, epoch, is_test, device, use_wandb=False):
        type_str = 'fitb_test' if is_test else 'fitb_eval'
        epoch_iterator = tqdm(dataloader)
        
        total_pred = []
        for iter, batch in enumerate(epoch_iterator, start=1):
            questions = {key: value.to(device) for key, value in batch['questions'].items()}
            candidates = {key: value.to(device) for key, value in batch['candidates'].items()}

            n_batch, n_candidate = candidates['mask'].shape

            encoded_question = self.encode(questions)
            encoded_candiates = self.encode(candidates)

            encoded = {
                'mask': torch.cat([
                    encoded_question['mask'].unsqueeze(1).expand(-1, n_candidate, -1), 
                    encoded_candiates['mask'].unsqueeze(2)
                    ], dim=2).flatten(0, 1),
                'embed': torch.cat([
                    encoded_question['embed'].unsqueeze(1).expand(-1, n_candidate, -1, -1), 
                    encoded_candiates['embed'].unsqueeze(2)
                    ], dim=2).flatten(0, 1)
                }

            logits = self.cp_forward(encoded)
            logits = logits.view(n_batch, n_candidate)
            run_pred = logits.argmax(dim=1).cpu().detach().numpy()
            total_pred = np.concatenate([total_pred, run_pred], axis=None)
            #--------------------------------------------------------------------------------------------#
            # Logging
            run_acc = sum(run_pred == 0) / len(run_pred)
            epoch_iterator.set_description(f'[{type_str}] Epoch: {epoch + 1:03} | Acc: {run_acc:.5f}')
            if use_wandb:
                log = {
                    f'{type_str}_acc': run_acc, 
                    f'{type_str}_step': epoch * len(epoch_iterator) + iter
                    }
                wandb.log(log)
            #--------------------------------------------------------------------------------------------#
        total_acc = sum(total_pred == 0) / len(total_pred)
        print(f'[{type_str} END] Epoch: {epoch + 1:03} | Acc: {total_acc:.5f}\n')
        return total_acc