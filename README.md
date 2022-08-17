# Text classification with pre-trained BERT-base model using M1 Silicon

## 1. Setup on M1
### Install virtual enviroment with anaconda:
```
conda create -n torch-m1 python=3.9
conda install pytorch torchvision torchaudio -c pytorch
conda install -c conda-forge transformers
conda activate torch-m1
```

## 2. Training model with GPU of M1 Silicon through Metal Performance Shaders (MPS) as backend.
### 2.1 Check MPS is available or not:
```
(torch-m1) ~ % python
>>> import torch
>>> torch.backends.mps.is_available()
```

### 2.2 Using GPU M1:
```
device = torch.device('mps')

Or only using CPU of M1:
#device = torch.device('cpu')
```

### 2.3 Load model pre-trained:
```
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert = BertModel.from_pretrained('bert-base-cased')
```

## 3. Evaluate
### 3.1 Devices:
- Apple M1:
  + CPU/GPU: Macbook Pro M1
  + RAM: 16GB
- Colab :
  + GPU: Tesla P100 16GB
  + RAM: 32GB (high-ram)
#### Parameter:
- max_length: 512
- batch_size: 8

#### Results (per every epoch):

|Device  |GPU    |CPU     |
|--------|-------|--------|
|Apple M1|47m10s | 51m20s |
|Colab Pro|6m40s  |   *    |

(*)take an hour for complete 20% of epoch. 


## 4. Reference
- https://pytorch.org/docs/master/notes/mps.html
- https://towardsdatascience.com/installing-pytorch-on-apple-m1-chip-with-gpu-acceleration-3351dc44d67c
