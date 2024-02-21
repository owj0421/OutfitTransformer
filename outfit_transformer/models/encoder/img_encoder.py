# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, swin_s, vgg13, efficientnet_b0
from transformers import AutoModel

class BaseEncoder(nn.Module):
    def __init__(
            self,
            embedding_dim: int=64,
            do_linear_probing: bool = False,
            normalize: bool = False
            ):
        super().__init__()
        self.model = None
        self.embedding_dim = embedding_dim
        self.do_linear_probing = do_linear_probing
        self.normalize = normalize

    def freeze_backbone(self):
        for name, param in self.model.layer1.named_parameters():
            if name in ['fc', 'head', 'classifier'] :
                print(name)
                continue
            param.requires_grad = False
        
    def forward(self, x):
        y = self.model(x)
        if self.normalize:
            y = F.normalize(y, p=2, dim=1)
        
        return y
    

class ResNet18Encoder(BaseEncoder):
    def __init__(
            self,
            embedding_dim: int=64,
            do_linear_probing: bool = False,
            normalize: bool = False
            ):
        super().__init__(embedding_dim, do_linear_probing, normalize)
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.embedding_dim, bias=True)
        if do_linear_probing:
            self.freeze_backbone()


class VGG13Encoder(BaseEncoder):
    def __init__(
            self,
            embedding_dim: int=64,
            do_linear_probing: bool = False,
            normalize: bool = False
            ):
        super().__init__(embedding_dim, do_linear_probing, normalize)
        self.model = vgg13(pretrained=True)
        n_inputs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(n_inputs, self.embedding_dim)
        if do_linear_probing:
            self.freeze_backbone()


class EfficientNetB0Encoder(BaseEncoder):
    def __init__(
            self,
            embedding_dim: int=64,
            do_linear_probing: bool = False,
            normalize: bool = False
            ):
        super().__init__(embedding_dim, do_linear_probing, normalize)
        self.model = efficientnet_b0(pretrained=True)
        n_inputs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(n_inputs, self.embedding_dim, bias=True)
        if do_linear_probing:
            self.freeze_backbone()


class SwinTransformerEncoder(BaseEncoder):
    def __init__(
            self,
            embedding_dim: int=64,
            do_linear_probing: bool = False,
            normalize: bool = False
            ):
        super().__init__(embedding_dim, do_linear_probing, normalize)
        self.model = swin_s(pretrained=True)
        n_inputs = self.model.head.in_features
        self.model.head = nn.Sequential(
            nn.Linear(n_inputs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, self.embedding_dim)
            )
        if do_linear_probing:
            self.freeze_backbone()