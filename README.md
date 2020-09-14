# NLP-ATC

NLP-ATC is a extremely lightweight library for **text classification**, it can train and predict nlp text classification model within one line like **sklearn**.

下载网盘数据，打标训练集`data`数据放根目录 

# Features

text classification model:
- fasttext
- cnntext
- albert-classifier
- bert_classifier
- hierarchy_attention_network(han) (implementing...)


# Get Starts

#### how to use
Download the **xatc** module, put it into your working path, then import this module.

#### prepare train data
train data should be prepared as a file like:
```
sentence\tlabel
sentence\tlabel
....
```

#### train model
Also can see in **examples**.
```python

################
# cnntext train
################
from xatc.models.cnntext import train
train(data_path=data_path,
      model_path=model_saved_path,
      train_size=0.9,
      model_params={
          "epochs": 1,
          "batch_size": 64,
          "lr":1e-4,
      })

################
# albert train
################
from xatc.models.albert_classifier import train
train(data_path=data_path,
      model_path=model_saved_path,
      train_size=0.9,
      model_params={
          "epochs": 2,
          "batch_size": 128,
          "hidden_dim": 50,
      })

################
# bert train
################
from xatc.models.bert_classifier import train
train(data_path=data_path,
      model_path=model_saved_path,
      train_size=0.9,
      model_params={
          "epochs": 2,
          "batch_size": 128,
          "hidden_dim": 50,
      })

################
# fasttext train
################
from xatc.models.fasttext import train
train(data_path=data_path,
      model_path=model_saved_path,
      train_size=0.9,
      model_params=None)
```

#### load model to predict 
Different models have the **same** predict functions. Also can see in **examples**.
```python
# cnn 
from xatc.models.cnntext import Predictor
model = Predictor(model_path=model_saved_path)
pred = model.predict(data=[...])

# albert
from xatc.models.albert_classifier import Predictor
model = Predictor(model_path=model_saved_path)
pred = model.predict(data=[...])

# bert
from xatc.models.bert_classifier import Predictor
model = Predictor(model_path=model_saved_path)
pred = model.predict(data=[...])

# fasttext
from xatc.models.fasttext import Predictor
model = Predictor(model_path=model_saved_path)
pred = model.predict(data=[...])
```

# API

#### cnntext train parameters
- finetune: default **True**, if embeddings needs finetune
- label_weights: default **None**, label weights
- dropout: default **0.3**, dropout rate
- embedding_dim: default **200**, the dim of embeddings
- vocab_path: default will load tencent pre_trained vocabulary
- emb_path: default is tencent pre_trained embedding, you can use other pre_trained embeddings
- emb_sep: default is **" "**, the sep of pre_trained embedding file
- kernel_size: default is **[3, 4, 5]**, the kernel size the model will use
- kernel_num: default is **100**, the channal of each kernel
- batch_size: default is **64**
- lr: default is **1e-4**, learning rate
- eps: default is  **1e-8**, the eps of Adam optimizer
- epochs: default is **5**, the epoch of training model
- n_jobs: default is **1**, the num of thread when preprocess and load data, -1 is the max num of thread

#### albert classifier train parameters
- finetune: default **True**, if embeddings needs finetune
- label_weights: default **None**, label weights
- dropout: default **0.3**, dropout rate
- hidden_dim, default is **50**, the hidden layer size after albert
- dropout: default **0.3**, dropout rate
- pre_trained_model_path: default is **albert_tiny_bright**, you can load other pytorch bert pre_trained model
- batch_size: default is **64**
- lr: default is **1e-4**, learning rate
- eps: default is  **1e-8**, the eps of Adam optimizer
- epochs: default is **5**, the epoch of training model
- n_jobs: default is **1**, the num of thread when preprocess and load data, -1 is the max num of thread

#### bert classifier train parameters
- finetune: default **True**, if embeddings needs finetune
- label_weights: default **None**, label weights
- dropout: default **0.3**, dropout rate
- hidden_dim, default is **50**, the hidden layer size after albert
- dropout: default **0.3**, dropout rate
- pre_trained_model_path: default is **bert_base_wwm**, you can load other pytorch bert pre_trained model
- batch_size: default is **64**
- lr: default is **1e-4**, learning rate
- eps: default is  **1e-8**, the eps of Adam optimizer
- epochs: default is **5**, the epoch of training model
- n_jobs: default is **1**, the num of thread when preprocess and load data, -1 is the max num of thread

#### fasttext train parameters (same with facebook fasttext)
- lr: default **1.0**, learning rate
- dim: default **100**, size of word vectors
- ws: **5**, size of the context window
- epoch: default **25**, number of epochs
- minCount: default **1**,  minimal number of word occurences
- minCountLabel: default **0**,  minimal number of label occurences
- minn: default **0**, min length of char ngram
- maxn: default **0**, max length of char ngram
- neg: default **5**, number of negatives sampled
- wordNgrams: **4**, max length of word ngram
- loss: default **'softmax'**, loss function {ns, hs, softmax, ova}
- bucket: default **2000000**, number of buckets
- thread: default **max_cpu-1**, number of threads
- lrUpdateRate: default **100**, change the rate of updates for the learning rate
- t: default **0.0001**, sampling threshold
- label: default **'__label__'**, label prefix
- verbose: default **2**, verbose
- pretrainedVectors: default **''** pre_trained word vectors (.vec file) for supervised learning


# Version
- v1.0.0: text classification model: fasttext, cnntext, albert_classifier
- v1.0.1: implement Predictor for each model to replace predict function
           update readme
- v1.1.1: add model: bert_classificatier

# Futures
- v1.2.1: add model: han



