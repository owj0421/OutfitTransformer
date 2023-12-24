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

import matplotlib.pyplot as plt


dataset_args = DatasetArgs(
        #workdir='/content/drive/MyDrive/outfit-transformer/',
        #datadir='/content/polyvore_outfits/',
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

DIR = 'F:/Projects/datasets/polyvore_outfits/images'
def show_images(query, candidates, probs):
    fig, axes = plt.subplots(2, max(len(query), len(candidates)))
    fig.set_figheight(3 * 3)
    fig.set_figwidth(3 * max(len(query), len(candidates)))
    axes[0][0].axis("off")
    for i, id in enumerate(query, start=0):
        axes[0][i].axis("off")
        axes[0][i].imshow(Image.open(os.path.join(DIR, str(id) + '.jpg')).resize((224, 224)))
    for i, (id, prob) in enumerate(zip(candidates, probs), start=0):
        axes[1][i].axis("off")
        axes[1][i].set_title(f'{prob:.3f}')
        axes[1][i].imshow(Image.open(os.path.join(DIR, str(id) + '.jpg')).resize((224, 224)))
    plt.show()


@torch.no_grad()
def infer():
    device = torch.device('cuda')

    MODEL_PATH = 'F:\Projects\outfit-transformer\model\saved_model'
    MODEL_NAME = 'fine-tune_2023-12-24_3_0.371'

    model = OutfitTransformer(pretrain=False, embedding_dim=128).to(device)
    model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f'{MODEL_NAME}.pth')), strict=False)
    model.eval()

    encoder = ItemEncoder(embedding_dim=128).to(device)
    encoder.load_state_dict(torch.load(os.path.join(MODEL_PATH, f'{MODEL_NAME}_encoder.pth')), strict=False)

    test_dataset = PolyvoreDataset(dataset_args, pretrain=False, dataset_type='test', train_transform=train_transform)
    test_dataloader = DataLoader(test_dataset, 16, shuffle=False)

    correct = 0.
    n_questions = 0.

    epoch_iterator = tqdm(test_dataloader)
    for batch in epoch_iterator:
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

        context_embedding = model(question_set, 
                                question_set_mask.to(device))
                    
        B, T, C, H, W = candidate_img.shape
        candidate_img = candidate_img.view(B * T, C, H, W)
        candidate_input_ids = candidate_input_ids.view(B * T, -1)
        candidate_attention_mask = candidate_attention_mask.view(B * T, -1)
        candidate_embedding = encoder(candidate_img.to(device),
                                    candidate_input_ids.to(device),
                                    candidate_attention_mask.to(device)).view(B, T, -1)
                    
        B, N, D = candidate_embedding.shape
        prob = nn.PairwiseDistance()(context_embedding.unsqueeze(1).expand_as(candidate_embedding).contiguous().view(-1, D), candidate_embedding.view(-1, D))
        prob = prob.view(B, N)
        pred = torch.argmin(prob, dim=-1)

        correct += torch.sum(pred.cpu()==ans_idx.view(-1))
        n_questions += len(ans_idx.view(-1))

        epoch_iterator.set_description(
                        'Test acc: {:.2f}'.format(correct / n_questions)
                        )
        

    print(f'Acc : {100 * correct / n_questions}%')

        # question_ids, candidate_ids, *_ = test_dataset.data[idx]
        # show_images(question_ids, candidate_ids, prob[0].tolist())

if __name__ == '__main__':
    infer()