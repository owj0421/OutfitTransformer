import os
import random
import pandas as pd
from tqdm import tqdm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader

from utils.dataset import PolyvoreDataset, DatasetArgs
from model.model import *
from model.encoder import ItemEncoder
from utils.metric import MetricCalculator

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

dataset_args = DatasetArgs(
    work_dir='F:/Projects/outfit-transformer',
    data_dir='F:/Projects/datasets/polyvore_outfits',
    polyvore_split='nondisjoint'
    )

train_transform = A.Compose([
    A.Resize(230, 230),
    A.HorizontalFlip(),
    A.RandomResizedCrop(scale=(0.85, 1.05), height=dataset_args.img_size, width=dataset_args.img_size, always_apply=True, p=1),
    A.Rotate(limit=15, p=0.5),
    A.Normalize(),
    ToTensorV2()
    ])

@torch.no_grad()
def infer():
    THRESHOLD = 0.5
    device = torch.device('cuda')

    checkpoint = torch.load('F:/Projects/outfit-transformer/checkpoints/compatibility/2023-12-27/2_0.923.pth')

    model = OutfitTransformer(embedding_dim=128).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    encoder = ItemEncoder(embedding_dim=128).to(device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)

    model.eval()
    encoder.eval()

    metric = MetricCalculator()

    test_dataset = PolyvoreDataset(dataset_args, task='compatibility', dataset_type='test', train_transform=train_transform)
    test_dataloader = DataLoader(test_dataset, 4, shuffle=True)

    epoch_iterator = tqdm(test_dataloader)
    for batch in epoch_iterator:
        y, set_mask, img, input_ids, attention_mask = batch

        B, T, C, H, W = img.shape
        img = img.view(B * T, C, H, W)
        input_ids = input_ids.view(B * T, -1)
        attention_mask = attention_mask.view(B * T, -1)

        set_input = encoder(img.to(device),
                            input_ids.to(device), 
                            attention_mask.to(device))
                
        set_input = set_input.view(B, T, -1)

        logits = model('compatibility',
                       set_input, 
                       set_mask.to(device))

        y_true = y.view(-1).cpu().detach().type(torch.int).tolist()
        y_score = logits.view(-1).cpu().detach().numpy().tolist()
        # print(y_true)
        # print(y_score)
        y_pred = (logits.view(-1).cpu().detach().numpy() >= THRESHOLD).astype(int).tolist()
        metric.update(y_true, y_pred, y_score)

        running_acc = metric.calc_acc()
        epoch_iterator.set_description(
            'Test | Acc: {:.5f}'.format(running_acc)
             )
        

    total_auc = metric.calc_auc()
    total_acc = metric.calc_acc()
    print(f'CP-> Acc: {total_acc:.2f} | Auc: {total_auc:.2f}')

        # question_ids, candidate_ids, *_ = test_dataset.data[idx]
        # show_images(question_ids, candidate_ids, prob[0].tolist())

if __name__ == '__main__':
    infer()