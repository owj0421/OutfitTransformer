import os
import wandb
import argparse
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torch.utils.data import DataLoader

from utils.dataset import *
from model.model import *
from model.encoder import ItemEncoder
from utils.metric import MetricCalculator
from utils.trainer import *

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Parser
parser = argparse.ArgumentParser(description='Outfit-Transformer Trainer')
parser.add_argument('--test_task', help='cp, fitb', type=str, default='cp')
parser.add_argument('--work_dir', help='Full working directory', type=str, required=True)
parser.add_argument('--data_dir', help='Full dataset directory', type=str, required=True)
parser.add_argument('--test_batch', type=int, default=4)
parser.add_argument('--checkpoint', default=None)
args = parser.parse_args()

# Setup
test_dataset_args = DatasetArguments(
    polyvore_split = 'nondisjoint',
    task_type = args.test_task,
    dataset_type = 'test',
    img_size = 224,
    use_pretrined_tokenizer = True,
    max_token_len = 16,
    custom_transform = None
    )

test_dataset = PolyvoreDataset(args.work_dir, args.data_dir, test_dataset_args)
test_dataloader = DataLoader(test_dataset, args.test_batch, shuffle=False)
print('[COMPLETE] Build Dataset')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

checkpoint = torch.load(args.checkpoint)
print('[COMPLETE] Load checkpoint')

encoder = ItemEncoder(embedding_dim=128).to(device)
encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
print('[COMPLETE] Build and Load Encoder')

model = OutfitTransformer(embedding_dim=128).to(device)
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
print('[COMPLETE] Build and Load Model')

metric = MetricCalculator()

encoder.eval()
model.eval()
with torch.no_grad():
    if args.task=='cp':
        _, acc, auc = cp_iteration(model, encoder, test_dataloader, 0, is_train=False, device=device, metric=metric)
        print(f'TEST | Task: {args.task} | acc: {acc:.5f} | auc: {auc:.5f}')
    elif args.task=='fitb':
        _, acc = fitb_iteration(model, encoder, test_dataloader, 0, is_train=False, device=device, metric=metric)
        print(f'TEST | Task: {args.task} | acc: {acc:.5f}')