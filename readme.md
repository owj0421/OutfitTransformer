# <div align="center"> Outfit-Transformer </div>

<div align="center"> 2023. 12. 26 : CP, FITB is Available </div>

## 🤗 Introduction
Implementation of paper - [Outfit Transformer: Outfit Representations for Fashion Recommendation](https://arxiv.org/abs/2204.04812)

Trained on a single NVIDIA A100 with 80GB memory.

## 🎯 Performance
|Model|CP(AUC)|FITB(Accuracy)|CIR(Recall@10)|
|:-|:-:|:-:|:-:|
|Type-Aware|0.86|57.83|3.50|
|SCE-Net|0.91|59.07|5.10|
|CSA-Net|0.91|63.73|8.27|
|OutfitTransformer(Paper)|0.93|67.10|9.58|
|**OutfitTransformer(Implemented)**|**0.93**|**?**|Not Trained|

## ⚙ Install Dependencies
This code is tested with python 3.9.16, torch 1.12.1
```
python -m pip install -r requirements.txt
```

## 🧱 Train
### Data Preparation
Download the polyvore dataset from [here](https://github.com/xthan/polyvore-dataset)

### Pretraining on CP(Compatibiliby Prediction) task
```
python train.py --train_task cp --valid_task fitb --train_batch 64 --valid_batch 96 --n_epochs 8 --learning_rate 1e-3 --work_dir $WORK_DIR --data_dir $DATA_DIR --wandb_api_key $WANDB_API_KEY
```

### Finetuning on CIR(Complementary Item Retrival) task
```
python train.py --train_task cir --valid_task cir --train_batch 48 --valid_batch 96 --n_epochs 2 --learning_rate 5e-5 --scheduler_step_size 100 --work_dir $WORK_DIR --data_dir $DATA_DIR --wandb_api_key $WANDB_API_KEY --checkpoint $CHECKPOINT
```

## 🔍 Test
```
python test.py --test_task $TASK --work_dir $WORK_DIR --data_dir $DATA_DIR --checkpoint $CHECKPOINT
```

## 🧶 Demo
```
Preparing...
```

## 🔔 Note
- A paper review of implementation can be found at [here](). (Only Available in Korean)
- This is **NON-OFFICIAL** implementation. (The official repo has not been released.)
