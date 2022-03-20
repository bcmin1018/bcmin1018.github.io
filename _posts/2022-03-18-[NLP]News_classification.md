---
layout: single
title: "[NLP] NEWS_Classification with roberta-large bert"
projects: NLP
tag: [pytorch, nlp, deeplearning]
toc: true
---

# NEWS 기사 분류
bert 모델로는 "klue/roberta-large"를 사용하였으며 텍스트 전처리는 따로 진행하지 않았다.

사용언어: pytorch

BBC 뉴스 데이터의 출처: https://www.kaggle.com/datasets/shivamkushwaha/bbc-full-text-document-classification

```python
!pip install transformers
```


```python
import os, sys 
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive


## 1. 기본 라이브러리
```python
import pandas as pd
import numpy as np
import datetime
import time
import random
import torch
import tensorflow as tf
import warnings

from torch.nn import functional as F
from tqdm.notebook import tqdm, tqdm_notebook
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AdamW, RobertaForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

device = torch.device("cuda")
warnings.filterwarnings(action='ignore')
```


```python
DIR = '/content/drive/MyDrive/project/Classification/BBC_NEWS/data/'
```

## 2. 데이터 전처리
### 2.1 txt를 dataframe 형태로 변환
데이터가 txt 파일 형태로 저장되어 있어 하나씩 열어서 text 리스트에 append 하는 작업 진행.

```python
directory = []
file = []
title = []
text = []
label = []

for dir, _, filenames in os.walk(DIR):
  if len(filenames) == 1:
    continue
  
  for filename in filenames:
    directory.append(dir)
    file.append(filename)
    label.append(dir.split('/')[-1])
    fullpathfile = os.path.join(dir, filename)

    with open(fullpathfile, 'r', encoding = 'utf8', errors = 'ignore') as infile:
      intext = ''
      firstline = True
      for line in infile:
        if firstline:
          title.append(line.replace('\n', ''))
          firstline = False
        else:
          intext = intext + ' ' + line.replace('\n', '')
      text.append(intext)
```


```python
DATA_COLUMN = 'text'
LABEL_COLUMN = 'label'

df = pd.DataFrame(list(zip(directory, file, title, text, label)), 
               columns =['directory', 'file', 'title', 'text', 'label'])
```


```python
df2 = df[['text', 'label']]
df2
```





  <div id="df-786e0060-17fd-4eeb-b386-d151a7087b45">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>An ex-Russian intelligence officer who riske...</td>
      <td>politics</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Police are investigating a British National ...</td>
      <td>politics</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Before he resigned the position of home secr...</td>
      <td>politics</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Voters' "pent up passion" could confound pre...</td>
      <td>politics</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Teenagers from well-off backgrounds are six ...</td>
      <td>politics</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2220</th>
      <td>South Korea's largest credit card firm has a...</td>
      <td>business</td>
    </tr>
    <tr>
      <th>2221</th>
      <td>UK house prices fell 0.7% in December, accor...</td>
      <td>business</td>
    </tr>
    <tr>
      <th>2222</th>
      <td>The US dollar's slide against the euro and y...</td>
      <td>business</td>
    </tr>
    <tr>
      <th>2223</th>
      <td>Traders at US banking giant Citigroup are fa...</td>
      <td>business</td>
    </tr>
    <tr>
      <th>2224</th>
      <td>The board of Indian conglomerate Reliance ha...</td>
      <td>business</td>
    </tr>
  </tbody>
</table>
<p>2225 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-786e0060-17fd-4eeb-b386-d151a7087b45')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>



### 2.2 LabelEncoder() 함수를 사용하여 5개 레이블에 모두 번호로 레이블링 진행.
```python
le = LabelEncoder()
df2['label'] = le.fit_transform(df2['label'])
```

```python
df2.groupby(by=['label'], as_index=False).count()
```

  <div id="df-c3a2a634-e7a2-4b1f-bcc5-61f271ee0055">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>label</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>510</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>386</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>417</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>511</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>401</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c3a2a634-e7a2-4b1f-bcc5-61f271ee0055')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c3a2a634-e7a2-4b1f-bcc5-61f271ee0055 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c3a2a634-e7a2-4b1f-bcc5-61f271ee0055');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>


## 3. 딥러닝 Setting 값
```python
MAX_LEN = 200
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EPOCH = 5
```

## 4. Bert를 위한 데이터셋
```python
class TRAINDataset(Dataset):

  def __init__(self, data):
    self.dataset = data
    self.tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
  
  def __len__(self):
    return len(self.dataset)
  
  def __getitem__(self, idx):
    row = self.dataset.iloc[idx, 0:2].values
    x = row[0]
    y = row[1]
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
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
```

## 5. training
```python
def training(train_dataset,val_dataset, fold):
  best_acc = 0
  
  model = RobertaForSequenceClassification.from_pretrained("klue/roberta-large", num_labels=5).to(device)
  
  dataset_train = TRAINDataset(train_dataset)
  dataset_val = TRAINDataset(val_dataset)

  train_loader = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
  valid_loader = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

  optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-2)

  total_steps = len(train_loader) * EPOCH

  # 스케줄러
  scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps = 0,
                                              num_training_steps = total_steps)
  # train_losses = []
  # val_losses = []


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
      scheduler.step()
      # batch_train_loss += train_loss.item()
      train_acc += calc_accuracy(out, label)
    
    # train_losses.append(batch_train_loss)
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))

    model.eval()
    with torch.no_grad():
      for batch_id, (token_ids, attention_masks, label) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
        token_ids = token_ids.to(device)
        attention_masks = attention_masks.to(device)
        label = label.to(device)
        out = model(token_ids, attention_masks)[0]
        # val_loss = F.cross_entropy(out, label)
        # batch_valid_loss += val_loss.item()
        valid_acc += calc_accuracy(out, label)

      # val_losses.append(batch_valid_loss)
      print("epoch {} valid acc {}".format(e+1, valid_acc / (batch_id+1)))
      torch.save(model, '/content/drive/MyDrive/project/Classification/BBC_NEWS/etc/model.pt')
```

### 5.1 traing 시작
```python
# 교차검증
def main():
    seed= 2021
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # kfold
    kfold=[]

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
    for train_idx, val_idx in splitter.split(df2.iloc[:, 0],df2.iloc[:, 1]):
        kfold.append((df2.iloc[train_idx,:],df2.iloc[val_idx,:]))

    for fold,(train_datasets, valid_datasets) in enumerate(kfold):
        print(f'fold{fold} 학습중...')
        training(train_dataset=train_datasets,val_dataset=valid_datasets,fold=fold)
```


```python
main()
```

    fold0 학습중...
    


    Downloading:   0%|          | 0.00/547 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/1.25G [00:00<?, ?B/s]


    Some weights of the model checkpoint at klue/roberta-large were not used when initializing RobertaForSequenceClassification: ['lm_head.dense.weight', 'lm_head.dense.bias', 'lm_head.layer_norm.bias', 'lm_head.decoder.bias', 'lm_head.layer_norm.weight', 'lm_head.decoder.weight', 'lm_head.bias']
    - This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
    - This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
    Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at klue/roberta-large and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
    You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
    


    Downloading:   0%|          | 0.00/375 [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/243k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/734k [00:00<?, ?B/s]



    Downloading:   0%|          | 0.00/173 [00:00<?, ?B/s]


    /usr/local/lib/python3.7/dist-packages/transformers/optimization.py:309: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      FutureWarning,
    


      0%|          | 0/223 [00:00<?, ?it/s]


    /usr/local/lib/python3.7/dist-packages/transformers/tokenization_utils_base.py:2277: FutureWarning: The `pad_to_max_length` argument is deprecated and will be removed in a future version, use `padding=True` or `padding='longest'` to pad to the longest sequence in the batch, or use `padding='max_length'` to pad to a max length. In this case, you can give a specific length with `max_length` (e.g. `max_length=45`) or leave max_length to None to pad to the maximal input size of the model (e.g. 512 for Bert).
      FutureWarning,
    

    epoch 1 train acc 0.7662556053811659
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 1 valid acc 0.8937499999999999
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 2 train acc 0.9489910313901345
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 2 valid acc 0.9227678571428571
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 3 train acc 0.9809417040358744
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 3 valid acc 0.9473214285714285
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 4 train acc 0.9932735426008968
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 4 valid acc 0.9540178571428571
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 5 train acc 0.9966367713004485
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 5 valid acc 0.9562499999999999
    fold1 학습중...
    

      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 1 train acc 0.7914798206278026
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 1 valid acc 0.8915178571428571
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 2 train acc 0.929372197309417
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 2 valid acc 0.9495535714285713
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 3 train acc 0.9730941704035875
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 3 valid acc 0.96875
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 4 train acc 0.9871076233183856
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 4 valid acc 0.96875
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 5 train acc 0.9983183856502242
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 5 valid acc 0.9642857142857143
    fold2 학습중...

      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 1 train acc 0.7253363228699552
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 1 valid acc 0.9040178571428571
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 2 train acc 0.9405829596412556
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 2 valid acc 0.9397321428571429
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 3 train acc 0.9848654708520179
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 3 valid acc 0.9263392857142857
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 4 train acc 0.9837443946188341
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 4 valid acc 0.953125
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 5 train acc 0.9960762331838565
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 5 valid acc 0.9508928571428571
    fold3 학습중...

      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 1 train acc 0.7892376681614349
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 1 valid acc 0.9361607142857142
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 2 train acc 0.9512331838565022
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 2 valid acc 0.9620535714285714
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 3 train acc 0.9803811659192825
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 3 valid acc 0.9241071428571429
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 4 train acc 0.9887892376681614
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 4 valid acc 0.9642857142857143
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 5 train acc 0.9977578475336323
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 5 valid acc 0.9598214285714286
    fold4 학습중...

      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 1 train acc 0.8099775784753364
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 1 valid acc 0.9330357142857143
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 2 train acc 0.9501121076233184
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 2 valid acc 0.9508928571428571
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 3 train acc 0.9736547085201793
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 3 valid acc 0.9464285714285714
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 4 train acc 0.9893497757847534
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 4 valid acc 0.9665178571428571
    


      0%|          | 0/223 [00:00<?, ?it/s]


    epoch 5 train acc 0.9988789237668162
    


      0%|          | 0/56 [00:00<?, ?it/s]


    epoch 5 valid acc 0.9665178571428571
    
## 6. 실제 뉴스기사로 모델 test
실제 BBC 뉴스 사이트에서 tech 기사 일부를 발췌하여 input data를 만들었다. 

```python
data = '''A system that searches a database of billions of facial images could help Ukraine uncover Russian infiltrators, fight misinformation and identify the dead, a company has said.
Facial recognition firm Clearview AI has offered its services to Ukraine's government.
The company says it has a searchable database of 10 billion faces sourced from the web.
But the technology has previously attracted fines from data regulators.
"I'm pleased to confirm that Clearview AI has provided its groundbreaking facial recognition technology to Ukrainian officials for their use during the crisis they are facing," chief executive Hoan Ton-That told the BBC in a statement.'''
```


```python
label_dict = {"business" : 0, "entertainment" : 1, "politics" : 2, "sport" : 3, "tech" : 4}
model = torch.load('/content/drive/MyDrive/project/Classification/BBC_NEWS/etc/model.pt')

def news_classification(text):
  tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")
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
  return answer
```


```python
result = news_classification(data)
for key, value in label_dict.items():
  if value == result:
    print("This is a %s news" %key )
```

tech로 잘 분류 되었음을 확인할 수 있었다.

    This is a tech news