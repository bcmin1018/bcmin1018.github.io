---
layout: single
title: "[NLP] Movie_Review_Sentiment_Analysis with bert"
projects: NLP
tag: [pytorch, nlp, deeplearning]
toc: true
---

# 영화 리뷰 감성 분석
Bert를 활용하여 영화 리뷰를 긍정 또는 부정으로 분류하는 프로젝트. Bert 모델은 "bert-base-multilingual-cased" 를 사용했다.

데이터셋 출처: https://dacon.io/competitions/official/235864/data


```python
!pip install transformers
```


```python
import os, sys 
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    

## 1. 라이브러리
```python
import pandas as pd
import numpy as np
import warnings
import random
import time
import datetime
import torch
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm.notebook import tqdm, tqdm_notebook
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

device = torch.device("cuda")
warnings.filterwarnings(action='ignore')
```


```python
DIR = '/content/drive/MyDrive/project/Sentiment_Analysis/영화리뷰/dataset'
TRAIN_SOURCE = os.path.join(DIR, "train.csv")
TEST_SOURCE = os.path.join(DIR, "test.csv")
SAMPLE_SUBMISSION = os.path.join(DIR, "sample_submission.csv")

train = pd.read_csv(TRAIN_SOURCE)
test = pd.read_csv(TEST_SOURCE)
print("train row : {}, train col : {}".format(str(train.shape[0]), str(train.shape[1])))
print("test row : {}, test col : {}".format(str(test.shape[0]), str(test.shape[1])))
```

    train row : 5000, train col : 3
    test row : 5000, test col : 2
    


```python
train_data, valid_data, train_label, valid_label = train_test_split(train['document'], train['label'], random_state = 2022, test_size = 0.2)
print("train row : {}".format(len(train_data)))
print("valid row : {}".format(len(valid_data)))
```

    train row : 4000
    valid row : 1000
    

## Setting
```python
MAX_LEN = 60
BATCH_SIZE = 16
LEARNING_RATE = 1e-6
EPOCH = 30
```

## 3. Bert에 import할 데이터 준비
```python
class TRAINDataset(Dataset):

  def __init__(self, data, label, val=True):
    self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    self.dataset = data
    self.label = label

  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, idx):
    x = self.dataset.iloc[idx]
    y = self.label.iloc[idx]

    inputs = self.tokenizer(
        x,
        truncation=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        add_special_tokens=True,
        max_length=MAX_LEN
    )
    
    input_ids = torch.from_numpy(np.asarray(inputs['input_ids']))
    attention_mask = torch.from_numpy(np.asarray(inputs['attention_mask']))

    return input_ids, attention_mask, y
```


```python
class TESTDataset(Dataset):
  
  def __init__(self, data):
    self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    self.dataset = data
      
  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, idx):
    x = self.dataset.iloc[idx]

    inputs = self.tokenizer(
        x,
        truncation=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        add_special_tokens=True,
        max_length=100
    )
    
    input_ids = torch.from_numpy(np.asarray(inputs['input_ids']))
    attention_mask = torch.from_numpy(np.asarray(inputs['attention_mask']))

    return input_ids, attention_mask
```


```python
# 정확도 계산
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
```

## 4. training
```python
def training(train_dataset,val_dataset):
  best_acc = 0
  model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2).to(device)
  # dataset_train = TRAINDataset(train_dataset)
  # dataset_val = TRAINDataset(val_dataset)
  train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
  valid_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
  optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-2)
  total_steps = len(train_loader) * EPOCH
  print("total_steps {}".format(total_steps))

  # 스케줄러
  scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps = 0,
                                              num_training_steps = total_steps)
  
  train_losses = []
  val_losses = []
  lrs = []

  for e in range(EPOCH):
    train_acc = 0.0
    valid_acc = 0.0
    batch_train_loss = 0.0
    batch_valid_loss = 0.0

    model.train()
    for batch_id, (token_ids, attention_masks, label) in tqdm(enumerate(train_loader), total=len(train_loader)):
      optimizer.zero_grad()
      token_ids = token_ids.to(device)
      attention_masks = attention_masks.to(device)
      label = label.to(device)
      out = model(token_ids, attention_masks)[0]
      train_loss = F.cross_entropy(out, label)
      train_loss.backward()
      optimizer.step()
      lrs.append(optimizer.param_groups[0]['lr'])
      scheduler.step()
      batch_train_loss += train_loss.item()
      train_acc += calc_accuracy(out, label)
    
    train_losses.append(batch_train_loss)
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))

    model.eval()
    with torch.no_grad():
      for batch_id, (token_ids, attention_masks, label) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        token_ids = token_ids.to(device)
        attention_masks = attention_masks.to(device)
        label = label.to(device)
        out = model(token_ids, attention_masks)[0]
        val_loss = F.cross_entropy(out, label)
        batch_valid_loss += val_loss.item()
        valid_acc += calc_accuracy(out, label)

      val_losses.append(batch_valid_loss)
      print("epoch {} valid acc {}".format(e+1, valid_acc / (batch_id+1)))
      torch.save(model, '/content/drive/MyDrive/project/Sentiment_Analysis/영화리뷰/etc/model.pt')

  return lrs, train_losses, val_losses
```


```python
trainDataset = TRAINDataset(train_data, train_label)
validDataset = TRAINDataset(valid_data, valid_label)
```


    Downloading:   0%|          | 0.00/972k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/29.0 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/625 [00:00<?, ?B/s]



```python
train_result = training(trainDataset,validDataset)
```

    Some weights of the model checkpoint at bert-base-multilingual-cased were not used when initializing BertForSequenceClassification: ['cls.seq_relationship.bias', 'cls.predictions.decoder.weight', 'cls.predictions.transform.dense.weight', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.dense.bias', 'cls.seq_relationship.weight', 'cls.predictions.bias', 'cls.predictions.transform.LayerNorm.bias']
    - This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-multilingual-cased and are newly initialized: ['classifier.bias', 'classifier.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    

    total_steps 7500
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 1 train acc 0.575
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 1 valid acc 0.6865079365079365
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 2 train acc 0.70025
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 2 valid acc 0.7559523809523809
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 3 train acc 0.766
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 3 valid acc 0.7767857142857143
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 4 train acc 0.7985
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 4 valid acc 0.7867063492063492
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 5 train acc 0.82325
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 5 valid acc 0.7956349206349206
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 6 train acc 0.842
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 6 valid acc 0.8015873015873016
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 7 train acc 0.85625
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 7 valid acc 0.810515873015873
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 8 train acc 0.8685
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 8 valid acc 0.8204365079365079
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 9 train acc 0.87375
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 9 valid acc 0.8184523809523809
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 10 train acc 0.88425
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 10 valid acc 0.8214285714285714
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 11 train acc 0.89225
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 11 valid acc 0.8244047619047619
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 12 train acc 0.895
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 12 valid acc 0.8343253968253969
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 13 train acc 0.906
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 13 valid acc 0.8333333333333334
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 14 train acc 0.9075
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 14 valid acc 0.8293650793650794
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 15 train acc 0.916
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 15 valid acc 0.8363095238095238
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 16 train acc 0.92675
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 16 valid acc 0.8313492063492064
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 17 train acc 0.92775
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 17 valid acc 0.8313492063492064
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 18 train acc 0.93275
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 18 valid acc 0.8224206349206349
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 19 train acc 0.9295
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 19 valid acc 0.8303571428571429
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 20 train acc 0.93325
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 20 valid acc 0.8273809523809523
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 21 train acc 0.93425
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 21 valid acc 0.8313492063492064
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 22 train acc 0.94175
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 22 valid acc 0.8293650793650794
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 23 train acc 0.9455
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 23 valid acc 0.8303571428571429
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 24 train acc 0.945
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 24 valid acc 0.8273809523809523
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 25 train acc 0.94175
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 25 valid acc 0.8273809523809523
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 26 train acc 0.9475
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 26 valid acc 0.8303571428571429
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 27 train acc 0.95025
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 27 valid acc 0.8313492063492064
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 28 train acc 0.9465
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 28 valid acc 0.8363095238095238
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 29 train acc 0.944
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 29 valid acc 0.8273809523809523
    


      0%|          | 0/250 [00:00<?, ?it/s]


    epoch 30 train acc 0.948
    


      0%|          | 0/63 [00:00<?, ?it/s]


    epoch 30 valid acc 0.8323412698412699
    


```python
lr = train_result[0]
train_loss = train_result[1]
valid_loss = train_result[2]

plt.plot(train_loss, color='red', label = 'train_loss')
plt.plot(valid_loss, color='blue', label = 'valid_loss')
plt.legend(loc='upper right')
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.show()
```


    
![png](Sentiment_Analysis%20%281%29_files/Sentiment_Analysis%20%281%29_12_0.png)
    


## 5. 샘플 댓글로 테스트
```python
def Sentiment_Analysis(text):
  tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
  model = torch.load('/content/drive/MyDrive/project/Sentiment_Analysis/영화리뷰/etc/model.pt')
  inputs = tokenizer(
        text,
        truncation=True,
        return_token_type_ids=False,
        pad_to_max_length=True,
        add_special_tokens=True,
        max_length=200
    )
  input_ids = torch.from_numpy(np.asarray(inputs['input_ids']))
  attention_mask = torch.from_numpy(np.asarray(inputs['attention_mask']))
  
  with torch.no_grad():
    token_ids = input_ids.unsqueeze(0).to(device)
    attention_masks = attention_mask.unsqueeze(0).to(device)
    output=model(token_ids, attention_masks)[0]
    logits = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
    answer= np.argmax(logits, axis=-1)
    if answer == 0:
      answer = "negative"
    else:
      answer = "positive"
  return answer
```


```python
review = "시간 때우기 좋은 영화 지루함"
```

결과가 잘 나오는 것을 확인할 수 있다.
```python
result = Sentiment_Analysis(review)
print(result)
```

    negative
    

## 6. 테스트 데이터 (제출)

```python
testDataset = TESTDataset(test['document'])
```


```python
def test_init(testdata):
  model = torch.load('/content/drive/MyDrive/project/Sentiment_Analysis/영화리뷰/etc/model.pt')
  model.eval()
  test_loader = DataLoader(testdata, batch_size=BATCH_SIZE, shuffle=True)
  pred = []
  print("--evaluation start--")
  with torch.no_grad():
    for batch_id, (token_ids, attention_masks) in tqdm(enumerate(test_loader), total=len(test_loader)):
        token_ids = token_ids.to(device)
        attention_masks = attention_masks.to(device)
        output = model(token_ids, attention_masks)[0]
        logits = torch.nn.functional.softmax(output, dim=1).detach().cpu().numpy()
        answer= np.argmax(logits, axis=-1)
        pred.extend(answer)
  return pred
  print("--evaluation end--")
```


```python
test_result = test_init(testDataset)
```

    --evaluation start--
    


      0%|          | 0/313 [00:00<?, ?it/s]



```python
sample_submission = pd.read_csv('/content/drive/MyDrive/project/Sentiment_Analysis/영화리뷰/dataset/sample_submission.csv')
sample_submission['label'] = test_result
sample_submission.to_csv('/content/drive/MyDrive/project/Sentiment_Analysis/영화리뷰/dataset/bert.csv', index=False)
```
