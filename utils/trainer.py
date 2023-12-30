import os
import wandb
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from datetime import datetime
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader
from .loss import *
from .metric import *
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union


def cp_iteration(model, encoder, dataloader, epoch, is_train, device, metric, optimizer=None, scheduler=None, use_wandb=False):
    type_str = 'Train' if is_train else 'Valid'

    epoch_iterator = tqdm(dataloader)
    losses = 0.0
    for iter, batch in enumerate(epoch_iterator, start=1):
        y, set_mask, img, input_ids, attention_mask = batch

        B, T, C, H, W = img.shape
        img = img.view(B * T, C, H, W)
        input_ids = input_ids.view(B * T, -1)
        attention_mask = attention_mask.view(B * T, -1)
        set_input = encoder(img.to(device),
                                    input_ids.to(device), 
                                    attention_mask.to(device))

        set_input = set_input.view(B, T, -1)

        logits = model('cp',
                       set_input, 
                       set_mask.to(device))
                
        loss = focal_loss(logits, y.to(device))
        losses += loss.item()

        y_true = y.view(-1).cpu().detach().type(torch.int).tolist()
        y_score = logits.view(-1).cpu().detach().numpy().tolist()
        y_pred = (logits.view(-1).cpu().detach().numpy() >= 0.5).astype(int).tolist()
        metric.update(y_true, y_pred, y_score)
        running_acc = metric.calc_acc()

        if is_train == True:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(chain(model.parameters(), encoder.parameters()), 5)
            optimizer.step()
            if scheduler:
                scheduler.step()
        
        if use_wandb:
            log = {
                f"{type_str}_loss": loss,
                f"{type_str}_acc": running_acc,
                f"{type_str}_step": epoch * len(epoch_iterator) + iter
                }
            if is_train == True:
                log["learning_rate"] = scheduler.get_last_lr()[0]

            wandb.log(log)
                
        epoch_iterator.set_description(
                f'{type_str} | Epoch: {epoch + 1:03} | Loss: {loss:.5f} | Acc: {running_acc:.5f}'
                )
        
    loss = losses / iter
    acc = metric.calc_acc()
    auc = metric.calc_auc()
    metric.clean()
    print( f'END {type_str} -> | Epoch: {epoch + 1:03} | loss: {loss:.5f} | Acc: {acc:.5f} | Auc: {auc:.5f}')
    return loss if is_train else auc
    

def fitb_iteration(model, encoder, dataloader, epoch, is_train, device, metric, optimizer=None, scheduler=None, use_wandb=False):
    type_str = 'Train' if is_train else 'Valid'

    epoch_iterator = tqdm(dataloader)
    losses = 0.0
    for iter, batch in enumerate(epoch_iterator, start=1):
        question_set_mask, question_img, question_input_ids, question_attention_mask,\
        candidate_img, candidate_input_ids, candidate_attention_mask,\
        ans_idx = batch

        B, T, C, H, W = question_img.shape
        question_img = question_img.view(B * T, C, H, W)
        question_input_ids = question_input_ids.view(B * T, -1)
        question_attention_mask = question_attention_mask.view(B * T, -1)
        question_set = encoder(question_img.to(device),
                                    question_input_ids.to(device),
                                    question_attention_mask.to(device)).view(B, T, -1)

        context_embedding = model('fitb',
                                  question_set, 
                                  question_set_mask.to(device))
            
        B, T, C, H, W = candidate_img.shape
        candidate_img = candidate_img.view(B * T, C, H, W)
        candidate_input_ids = candidate_input_ids.view(B * T, -1)
        candidate_attention_mask = candidate_attention_mask.view(B * T, -1)
        candidate_embedding = encoder(candidate_img.to(device),
                                            candidate_input_ids.to(device),
                                            candidate_attention_mask.to(device)).view(B, T, -1)
            
        B, N, D = candidate_embedding.shape

        loss = triplet_margin_loss_with_multiple_negatives(context_embedding, candidate_embedding[:, 0, :].unsqueeze(1), candidate_embedding[:, 1:, :])

        y_true = ans_idx.view(-1).numpy().tolist()
        y_score = nn.PairwiseDistance()(context_embedding.unsqueeze(1).expand_as(candidate_embedding).contiguous().view(-1, D), candidate_embedding.view(-1, D))
        y_score = y_score.view(B, N)
        y_pred = torch.argmin(y_score, dim=-1).view(-1).cpu().detach().numpy().tolist()
        metric.update(y_true, y_pred)
        running_acc = metric.calc_acc()

        if is_train == True:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(chain(model.parameters(), encoder.parameters()), 5)
            optimizer.step()
            if scheduler:
                scheduler.step()
        
        if use_wandb:
            log = {
                f"{type_str}_loss": loss,
                f"{type_str}_acc": running_acc,
                f"{type_str}_step": epoch * len(epoch_iterator) + iter
                }
            if is_train == True:
                log["learning_rate"] = scheduler.get_last_lr()[0]

            wandb.log(log)
                
        epoch_iterator.set_description(
                f'{type_str} | Epoch: {epoch + 1:03} | Loss: {loss:.5f} | Acc: {running_acc:.5f}'
                )
        
    loss = losses / iter
    acc = metric.calc_acc()
    metric.clean()
    print( f'END {type_str} -> | Epoch: {epoch + 1:03} | loss: {loss:.5f} | Acc: {acc:.5f}')
    return loss if is_train else acc


def triplet_iteration(model, encoder, dataloader, epoch, is_train, device, metric, optimizer=None, scheduler=None, use_wandb=False):
    type_str = 'Train' if is_train else 'Valid'

    epoch_iterator = tqdm(dataloader)
    losses = 0.0
    for iter, batch in enumerate(epoch_iterator, start=1):
        input_set_mask, input_img, input_input_ids, input_attention_mask, \
        pos_img, pos_input_ids, pos_attention_mask, \
        neg_img, neg_input_ids, neg_attention_mask = batch

        B, T, C, H, W = input_img.shape
        input_img = input_img.view(B * T, C, H, W)
        input_input_ids = input_input_ids.view(B * T, -1)
        input_attention_mask = input_attention_mask.view(B * T, -1)
        input_set = encoder(input_img.to(device),
                                    input_input_ids.to(device), 
                                    input_attention_mask.to(device)).view(B, T, -1)

        context_embedding = model('triplet',
                                  input_set, 
                                  input_set_mask.to(device))
            
        B, T, C, H, W = pos_img.shape
        pos_img = pos_img.view(B * T, C, H, W)
        pos_input_ids = pos_input_ids.view(B * T, -1)
        pos_attention_mask = pos_attention_mask.view(B * T, -1)
        pos_embedding = encoder(pos_img.to(device),
                                        pos_input_ids.to(device), 
                                        pos_attention_mask.to(device)).view(B, T, -1)
            
        B, T, C, H, W = neg_img.shape
        neg_img = neg_img.view(B * T, C, H, W)
        neg_input_ids = neg_input_ids.view(B * T, -1)
        neg_attention_mask = neg_attention_mask.view(B * T, -1)
        neg_embedding = encoder(neg_img.to(device),
                                        neg_input_ids.to(device),
                                        neg_attention_mask.to(device)).view(B, T, -1)
            
        loss = triplet_margin_loss_with_multiple_negatives(context_embedding, pos_embedding, neg_embedding)

        if is_train == True:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(chain(model.parameters(), encoder.parameters()), 5)
            optimizer.step()
            if scheduler:
                scheduler.step()
        
        if use_wandb:
            log = {
                f"{type_str}_loss": loss,
                f"{type_str}_step": epoch * len(epoch_iterator) + iter
                }
            if is_train == True:
                log["learning_rate"] = scheduler.get_last_lr()[0]

            wandb.log(log)
                
        epoch_iterator.set_description(
                f'{type_str} | Epoch: {epoch + 1:03} | Loss: {loss:.5f}'
                )
        
    loss = losses / iter
    print( f'END {type_str} -> | Epoch: {epoch + 1:03} | loss: {loss:.5f}')
    return loss


@dataclass
class TrainingArguments:
    train_task: str='cp'
    valid_task: str='fitb'
    train_batch: int=8
    valid_batch: int=32
    n_epochs: int=100
    learning_rate: float=0.01
    save_every: int=1
    work_dir: str=None
    use_wandb: bool=False
    device: str='cuda'


class Trainer:
    def __init__(
            self,
            args: TrainingArguments,
            model: nn.Module,
            encoder: nn.Module,
            train_dataloader: DataLoader,
            valid_dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            metric: MetricCalculator,
            scheduler: Optional[torch.optim.lr_scheduler.StepLR] = None
            ):
        self.device = torch.device(args.device)

        self.model = model
        self.model.to(self.device)
        self.encoder = encoder
        self.encoder.to(self.device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.scheduler = scheduler
        self.metric = metric
        self.args = args
        self.train_task = args.train_task
        self.valid_task = args.valid_task
        self.best_state = {}

    def fit(self):
        best_criterion = -np.inf
        for epoch in range(self.args.n_epochs):
            loss = self._train(epoch, self.train_dataloader)
            criterion = self._validate(epoch, self.valid_dataloader)
            if criterion > best_criterion:
               best_criterion = criterion
               self.best_state['model'] = deepcopy(self.model.state_dict())
               self.best_state['encoder'] = deepcopy(self.encoder.state_dict())

            if epoch % self.args.save_every == 0:
                date = datetime.now().strftime('%Y-%m-%d')
                output_dir = os.path.join(self.args.work_dir, 'checkpoints', self.valid_task, date)
                model_name = f'{epoch}_{best_criterion:.3f}'
                self._save(output_dir, model_name)


    def _train(self, epoch: int, dataloader: DataLoader):
        self.encoder.train()
        self.model.train()
        is_train=True

        if self.train_task=='cp':
            loss = cp_iteration(self.model, self.encoder, dataloader, epoch, is_train, self.device, self.metric, self.optimizer, self.scheduler, self.args.use_wandb)
        elif self.train_task=='fitb':
            loss = fitb_iteration(self.model, self.encoder, dataloader, epoch, is_train, self.device, self.metric, self.optimizer, self.scheduler, self.args.use_wandb)
        else:
            loss = triplet_iteration(self.model, self.encoder, dataloader, epoch, is_train, self.device, self.metric, self.optimizer, self.scheduler, self.args.use_wandb)
        return loss


    @torch.no_grad()
    def _validate(self, epoch: int, dataloader: DataLoader):
        self.encoder.eval()
        self.model.eval()
        is_train=False

        if self.valid_task=='cp':
            criterion = cp_iteration(self.model, self.encoder, dataloader, epoch, is_train, self.device, self.metric, self.optimizer, self.scheduler, self.args.use_wandb)
        elif self.valid_task=='fitb':
            criterion = fitb_iteration(self.model, self.encoder, dataloader, epoch, is_train, self.device, self.metric, self.optimizer, self.scheduler, self.args.use_wandb)
        else:
            # Our goal is maximizing criterion.
            criterion = -triplet_iteration(self.model, self.encoder, dataloader, epoch, is_train, self.device, self.metric, self.optimizer, self.scheduler, self.args.use_wandb)
        return criterion


    def _save(self, dir, model_name, best_model: bool=True):
        def _create_folder(dir):
            try:
                if not os.path.exists(dir):
                    os.makedirs(dir)
            except OSError:
                print('[Error] Creating directory.' + dir)
        _create_folder(dir)

        path = os.path.join(dir, f'{model_name}.pth')
        checkpoint = {
            'model_state_dict': self.best_state['model'] if best_model else self.model.state_dict(),
            'encoder_state_dict': self.best_state['encoder'] if best_model else self.encoder.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
            }
        torch.save(checkpoint, path)
        print(f'[COMPLETE] Save at {path}')


    def load(self, path, load_optim=False):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        if load_optim:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'], strict=False)
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'], strict=False)
        print(f'[COMPLETE] Load from {path}')