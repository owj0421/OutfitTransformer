# -*- coding:utf-8 -*-
"""
Author:
    Wonjun Oh, owj0421@naver.com
"""
from outfit_transformer.models.encoder.img_encoder import *
from outfit_transformer.models.encoder.txt_encoder import *


def build_img_encoder(
        backbone = 'resnet-18', 
        embedding_dim = 16,
        do_linear_probing = False,
        normalize = False
        ):
    if backbone == 'resnet-18':
        encoder = ResNet18Encoder(embedding_dim, do_linear_probing, normalize)
    elif backbone == 'swin-transformer':
        encoder = SwinTransformerEncoder(embedding_dim, do_linear_probing, normalize)
    elif backbone == 'vgg-13':
        encoder = VGG13Encoder(embedding_dim, do_linear_probing, normalize)
    elif backbone == 'efficientnet-b0':
        encoder = EfficientNetB0Encoder(embedding_dim, do_linear_probing, normalize)
    else:
        raise ValueError(
            f'{backbone} is not supported.'
            )
    
    return encoder

def build_txt_encoder(
        backbone = 'bert',
        huggingface = 'sentence-transformers/paraphrase-albert-small-v2',
        embedding_dim = 16,
        do_linear_probing = True,
        normalize = False
        ):
    if backbone == 'bert':
        encoder = BertEncoder(huggingface, embedding_dim, do_linear_probing, normalize)
    else:
        raise ValueError(
            f'{backbone} is not supported.'
            )
    
    return encoder