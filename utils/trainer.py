import os
import wandb
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from itertools import chain
from datetime import datetime
from dataclasses import dataclass
import torch
from .loss import *


@dataclass
class TrainingArgs():
    task: str='compatibility'
    train_batch: int=8
    valid_batch: int=32
    n_epochs: int=100
    learning_rate: float=0.01
    warmup: bool=True
    warmup_iter: int=1000
    save_every: int=1
    work_dir: str
    

class Trainer:
    def __init__(self, model, encoder, train_dataloader, valid_dataloader, optimizer, scheduler, metric, device, args):
        self.task = args.task

        self.model = model
        self.encoder = encoder
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.scheduler = scheduler
        self.device = device
        self.args = args
        self.loss_func = focal_loss if self.task=='compatibility' else triplet_margin_loss_with_multiple_negatives
        self.model.to(self.device)
        self.encoder.to(self.device)
        
        self.metric = metric

        self.best_model_state = None
        self.best_encoder_state = None
        self.best_optimizer_state = None

    def train(self):
        best_valid_criterion = -np.inf
        for epoch in range(self.args.n_epochs):
            train_loss = self._train(self.train_dataloader, epoch)
            valid_criterion = self._validate(self.valid_dataloader, epoch)

            if valid_criterion >= best_valid_criterion:
               best_valid_criterion = valid_criterion
               self.best_model_state = deepcopy(self.model.state_dict())
               self.best_encoder_state = deepcopy(self.encoder.state_dict())

            if epoch % self.args.save_every == 0:
                date = datetime.now().strftime('%Y-%m-%d')
                model_name = f'{epoch}_{best_valid_criterion:.3f}'
                save_dir = os.path.join(self.args.work_dir, 'checkpoints', self.task, date)
                self.save(save_dir, model_name)

    def _train(self, dataloader, epoch):
        self.encoder.train()
        self.model.train()

        epoch_iterator = tqdm(dataloader)
        losses = 0.0
        for iter, batch in enumerate(epoch_iterator, start=1):
            self.optimizer.zero_grad()

            if self.task=='compatibility':
                y, set_mask, img, input_ids, attention_mask = batch

                B, T, C, H, W = img.shape
                img = img.view(B * T, C, H, W)
                input_ids = input_ids.view(B * T, -1)
                attention_mask = attention_mask.view(B * T, -1)
                set_input = self.encoder(img.to(self.device),
                                         input_ids.to(self.device), 
                                         attention_mask.to(self.device))

                set_input = set_input.view(B, T, -1)

                logits = self.model(self.args.task,
                                    set_input, 
                                    set_mask.to(self.device))
                    
                loss = self.loss_func(logits, y.to(self.device))
                
            else:
                input_set_mask, input_img, input_input_ids, input_attention_mask, \
                pos_img, pos_input_ids, pos_attention_mask, \
                neg_img, neg_input_ids, neg_attention_mask = batch

                B, T, C, H, W = input_img.shape
                input_img = input_img.view(B * T, C, H, W)
                input_input_ids = input_input_ids.view(B * T, -1)
                input_attention_mask = input_attention_mask.view(B * T, -1)
                input_set = self.encoder(input_img.to(self.device),
                                         input_input_ids.to(self.device), 
                                         input_attention_mask.to(self.device)).view(B, T, -1)

                context_embedding = self.model(self.args.task,
                                               input_set, 
                                               input_set_mask.to(self.device))
                
                B, T, C, H, W = pos_img.shape
                pos_img = pos_img.view(B * T, C, H, W)
                pos_input_ids = pos_input_ids.view(B * T, -1)
                pos_attention_mask = pos_attention_mask.view(B * T, -1)
                pos_embedding = self.encoder(pos_img.to(self.device),
                                             pos_input_ids.to(self.device), 
                                             pos_attention_mask.to(self.device)).view(B, T, -1)
                
                B, T, C, H, W = neg_img.shape
                neg_img = neg_img.view(B * T, C, H, W)
                neg_input_ids = neg_input_ids.view(B * T, -1)
                neg_attention_mask = neg_attention_mask.view(B * T, -1)
                neg_embedding = self.encoder(neg_img.to(self.device),
                                               neg_input_ids.to(self.device),
                                               neg_attention_mask.to(self.device)).view(B, T, -1)
                
                loss = self.loss_func(context_embedding, pos_embedding, neg_embedding)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(chain(self.model.parameters(), self.encoder.parameters()), 5)
            self.optimizer.step()
            self.scheduler.step()
            losses += loss.item()

            epoch_iterator.set_description(
                'Train | Epoch: {:03}/{:03} | loss: {:.5f}'.format(epoch + 1, self.args.n_epochs, loss)
                )

            wandb.log({
                "learning_rate": self.scheduler.get_last_lr()[0],
                "train_loss": loss,
                "train_step": epoch * len(epoch_iterator) + iter
                })

        return losses / iter

    @torch.no_grad()
    def _validate(self, dataloader, epoch):
        self.encoder.eval()
        self.model.eval()

        epoch_iterator = tqdm(dataloader)
        losses = 0.0

        for iter, batch in enumerate(epoch_iterator, start=1):
            if self.task=='compatibility':
                y, set_mask, img, input_ids, attention_mask = batch

                B, T, C, H, W = img.shape
                img = img.view(B * T, C, H, W)
                input_ids = input_ids.view(B * T, -1)
                attention_mask = attention_mask.view(B * T, -1)

                set_input = self.encoder(img.to(self.device),
                                         input_ids.to(self.device), 
                                         attention_mask.to(self.device))
                
                set_input = set_input.view(B, T, -1)

                logits = self.model(self.args.task,
                                    set_input, 
                                    set_mask.to(self.device))
                
                loss = self.loss_func(logits, y.to(self.device))
                losses += loss.item()

                y_true = y.view(-1).cpu().detach().type(torch.int).tolist()
                y_score = logits.view(-1).cpu().detach().numpy().tolist()
                y_pred = (logits.view(-1).cpu().detach().numpy() >= 0.5).astype(int).tolist()
                self.metric.update(y_true, y_pred, y_score)

                running_acc = self.metric.calc_acc()
                epoch_iterator.set_description(
                        'Valid | Epoch: {:03}/{:03} | loss: {:.5f} | Acc: {:.2f}'.format(epoch + 1, self.args.n_epochs, loss, running_acc)
                        )

                wandb.log({
                    "valid_loss": loss,
                    "valid_acc": running_acc,
                    "valid_step": epoch * len(epoch_iterator) + iter
                    })
                    
            else:
                question_set_mask, question_img, question_input_ids, question_attention_mask,\
                candidate_img, candidate_input_ids, candidate_attention_mask,\
                ans_idx = batch

                B, T, C, H, W = question_img.shape
                question_img = question_img.view(B * T, C, H, W)
                question_input_ids = question_input_ids.view(B * T, -1)
                question_attention_mask = question_attention_mask.view(B * T, -1)
                question_set = self.encoder(question_img.to(self.device),
                                            question_input_ids.to(self.device),
                                            question_attention_mask.to(self.device)).view(B, T, -1)

                context_embedding = self.model(self.args.task,
                                               question_set, 
                                               question_set_mask.to(self.device))
                
                B, T, C, H, W = candidate_img.shape
                candidate_img = candidate_img.view(B * T, C, H, W)
                candidate_input_ids = candidate_input_ids.view(B * T, -1)
                candidate_attention_mask = candidate_attention_mask.view(B * T, -1)
                candidate_embedding = self.encoder(candidate_img.to(self.device),
                                                   candidate_input_ids.to(self.device),
                                                   candidate_attention_mask.to(self.device)).view(B, T, -1)
                
                B, N, D = candidate_embedding.shape

                prob = nn.PairwiseDistance()(context_embedding.unsqueeze(1).expand_as(candidate_embedding).contiguous().view(-1, D), candidate_embedding.view(-1, D))
                prob = prob.view(B, N)
                pred = torch.argmin(prob, dim=-1)

                y_true = ans_idx.view(-1).numpy().tolist()
                y_pred = pred.view(-1).cpu().detach().numpy().tolist()
                self.metric.update(y_true, y_pred)

                running_acc = self.metric.calc_acc()
                epoch_iterator.set_description(
                        'Valid | Epoch: {:03}/{:03} | Acc: {:.5f}'.format(epoch + 1, self.args.n_epochs, running_acc)
                        )
                            
                wandb.log({
                    "valid acc": running_acc,
                    "valid_step": epoch * len(epoch_iterator) + iter
                    })
        
        if self.task=='compatibility':
            total_auc = self.metric.calc_auc()
            total_acc = self.metric.calc_acc()
            criterion = total_auc
            print(f'CP-> Epoch: {epoch + 1:03}/{self.args.n_epochs:03} | Acc: {total_acc:.2f} | Auc: {total_auc:.2f}')
        else:
            total_acc = self.metric.calc_acc()
            criterion = total_acc
            print(f'FITB-> Epoch: {epoch + 1:03}/{self.args.n_epochs:03} | Acc: {total_acc:.2f}')
            
        self.metric.clean()
        return criterion

    def save(self, dir, model_name, best_model: bool=True):
        def create_folder(dir):
            try:
                if not os.path.exists(dir):
                    os.makedirs(dir)
            except OSError:
                print('[Error] Creating directory. ' + dir)

        create_folder(dir)
        path = os.path.join(dir, f'{model_name}.pth')
        checkpoint = {
            'model_state_dict': self.best_model_state if best_model else self.model.state_dict(),
            'encoder_state_dict': self.best_encoder_state if best_model else self.encoder.state_dict(),
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