import joblib
import jieba
import random
from abc import abstractmethod
from pathlib import PurePath
import itertools
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

from .utils import my_timer


class BaseDatasets:
    def __init__(self, path=None, data=None, sep="\t"):
        self._sep = sep
        self.corpus, self.label = self._load_raw_data(path, data)

    def _load_raw_data(self, path, data):
        corpus, label = list(), list()
        if path is not None:
            with open(path) as f:
                for line in f.readlines():
                    line = line.strip().split(self._sep)
                    corpus.append(line[0])
                    if len(line) > 1:
                        label.append(line[-1])
        if data is not None and isinstance(data, str):
            d = data.strip().split(self._sep)
            corpus.append(d[0])
            if len(d) > 1:
                label.append(d[-1])
        if data is not None and isinstance(data, list):
            for line in data:
                line = line.strip().split(self._sep)
                corpus.append(line[0])
                if len(line) > 1:
                    label.append(line[-1])
        return corpus, label

    @staticmethod
    def batch(data, batch_size):
        indices = [i for i in range(len(data))]
        random.seed(2020)
        random.shuffle(indices)
        n_batch = len(data) // batch_size + 1
        for i in range(n_batch):
            batch_indices = indices[0:batch_size]
            indices = indices[batch_size:] + indices[:batch_size]
            yield [data[i] for i in batch_indices]


class DatasetsMixin:
    @abstractmethod
    # @my_timer
    def load(self, **kwargs):
        pass

    @abstractmethod
    # @my_timer
    def loadp(self, **kwargs):
        pass


class FasttextDatasets(BaseDatasets, DatasetsMixin):
    def __init__(self, path, sep="\t", **kwargs):
        super(FasttextDatasets, self).__init__(path=path, sep=sep)

    @staticmethod
    def _cut_words(s):
        return jieba.lcut(s)

    # @my_timer
    def load(self, n_jobs=-1, train_size=None, random_state=2020):
        text = joblib.Parallel(n_jobs=n_jobs) \
            (joblib.delayed(self._cut_words)(c) for c in self.corpus)
        data = [" ".join(t) for t in text]
        if self.label is not None:
            data = [data[i] + "\t__label__" + self.label[i] for i in range(len(data))]
        if train_size is None:
            return data
        return train_test_split(data, train_size=train_size, random_state=random_state)

    # @my_timer
    def loadp(self, n_jobs=-1):
        return self.load(n_jobs=n_jobs)

    @staticmethod
    def save(data, path):
        with open(path, "w", encoding="utf-8") as f:
            f.writelines([s + "\n" for s in data])


class AlbertDatasets(BaseDatasets, DatasetsMixin):
    def __init__(self, path=None, data=None, pre_trained_path=PRE_TRAINED_MODEL_PATH, sep="\t", **kwargs):
        super(AlbertDatasets, self).__init__(path=path, data=data, sep=sep)
        self.pre_trained_path = pre_trained_path
        self.tokenizer = BertTokenizer.from_pretrained(self.pre_trained_path)
        self.max_length = min(max([len(c) for c in self.corpus]), 512)  # bert 最多支持512的长度

        self.label2idx = dict()
        self.num_labels = -1

    def _tokenize(self, s, l=None):
        encoded_dict = self.tokenizer.encode_plus(
            s,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=self.max_length,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt'  # Return pytorch tensors.
        )
        if l is None:
            return [encoded_dict['input_ids'], encoded_dict['attention_mask']]
        label_index = self.label2idx[l]
        label_ids = [1.0 if i == label_index else 0.0 for i in range(len(self.label2idx.keys()))]
        return [encoded_dict['input_ids'], encoded_dict['attention_mask'], label_ids]

    # @my_timer
    def load(self, n_jobs=1, max_length=None, batch_size=None, train_size=None, random_state=2020):
        self.label2idx = {l: i for i, l in enumerate(np.unique(self.label))}
        self.num_labels = len(self.label2idx.keys())
        if max_length is not None:
            self.max_length = max_length

        data = joblib.Parallel(n_jobs)(
            joblib.delayed(self._tokenize)(c, l) for c, l in tqdm(zip(self.corpus, self.label), total=len(self.corpus)))

        if train_size is None:
            tensor_dataset = TensorDataset(torch.cat([d[0] for d in data], dim=0),
                                           torch.cat([d[1] for d in data], dim=0),
                                           torch.tensor([d[2] for d in data]))
            if batch_size is None:
                return tensor_dataset
            return DataLoader(dataset=tensor_dataset, batch_size=batch_size)
        else:
            train, test = train_test_split(data, train_size=train_size, random_state=random_state)
            train_tensor_dataset = TensorDataset(torch.cat([d[0] for d in train], dim=0),
                                                 torch.cat([d[1] for d in train], dim=0),
                                                 torch.tensor([d[2] for d in train]))
            test_tensor_dataset = TensorDataset(torch.cat([d[0] for d in test], dim=0),
                                                torch.cat([d[1] for d in test], dim=0),
                                                torch.tensor([d[2] for d in test]))

            if batch_size is None:
                return [train_tensor_dataset, test_tensor_dataset]
            return [DataLoader(dataset=train_tensor_dataset, batch_size=batch_size),
                    DataLoader(dataset=test_tensor_dataset, batch_size=batch_size)]

    # @my_timer
    def loadp(self, max_length, label2index, n_jobs=1, batch_size=None):
        self.max_length = max_length
        self.label2idx = label2index
        self.num_labels = len(self.label2idx.keys())

        data = joblib.Parallel(n_jobs)(
            joblib.delayed(self._tokenize)(s=c) for c in tqdm(self.corpus))

        tensor_dataset = TensorDataset(torch.cat([d[0] for d in data], dim=0),
                                       torch.cat([d[1] for d in data], dim=0))
        if batch_size is None:
            return tensor_dataset
        return DataLoader(dataset=tensor_dataset, batch_size=batch_size)


BertDatasets = AlbertDatasets


class CNNTextDatasets(BaseDatasets, DatasetsMixin):
    def __init__(self, path=None, data=None, sep="\t", vocab2idx=None,
                 **kwargs):
        super(CNNTextDatasets, self).__init__(path=path, data=data, sep=sep)

        self.vocab2idx = vocab2idx
        self.max_length = min(max([len(c) for c in self.corpus]), 512)  # bert 最多支持512的长度

        self.label2idx = dict()
        self.num_labels = -1

    def _tokenize(self, s, l=None, max_length=None):
        s_l = len(s)
        sent_ids = list()
        for i in range(max_length):
            if i < s_l:
                sent_ids.append(self.vocab2idx.get(s[i], self.vocab2idx["[UNK]"]))
            else:
                sent_ids.append(self.vocab2idx["[PAD]"])
        if l is None:
            return [sent_ids]
        label_index = self.label2idx[l]
        label_ids = [1.0 if i == label_index else 0.0 for i in range(len(self.label2idx.keys()))]
        return [sent_ids, label_ids]

    def load(self, n_jobs=1, max_length=None, batch_size=None, train_size=None, random_state=2020, **kwargs):
        self.label2idx = {l: i for i, l in enumerate(np.unique(self.label))}
        self.num_labels = len(self.label2idx.keys())
        if max_length is not None:
            self.max_length = max_length

        data = joblib.Parallel(n_jobs)(
            joblib.delayed(self._tokenize)(c, l, max_length=self.max_length) for c, l in
            tqdm(zip(self.corpus, self.label), total=len(self.corpus)))

        if train_size is None:
            tensor_dataset = TensorDataset(torch.tensor([d[0] for d in data]),
                                           torch.tensor([d[1] for d in data]))
            if batch_size is None:
                return tensor_dataset
            return DataLoader(dataset=tensor_dataset, batch_size=batch_size)
        else:
            train, test = train_test_split(data, train_size=train_size, random_state=random_state)
            train_tensor_dataset = TensorDataset(torch.tensor([d[0] for d in train]),
                                                 torch.tensor([d[1] for d in train]))
            test_tensor_dataset = TensorDataset(torch.tensor([d[0] for d in test]),
                                                torch.tensor([d[1] for d in test]))

            if batch_size is None:
                return [train_tensor_dataset, test_tensor_dataset]
            return [DataLoader(dataset=train_tensor_dataset, batch_size=batch_size),
                    DataLoader(dataset=test_tensor_dataset, batch_size=batch_size)]

    def loadp(self, max_length, label2index, n_jobs=1, batch_size=None, **kwargs):
        self.max_length = max_length
        self.label2idx = label2index
        self.num_labels = len(self.label2idx.keys())

        data = joblib.Parallel(n_jobs)(
            joblib.delayed(self._tokenize)(s=c, max_length=self.max_length) for c in tqdm(self.corpus))

        tensor_dataset = TensorDataset(torch.tensor([d[0] for d in data]))
        if batch_size is None:
            return tensor_dataset
        return DataLoader(dataset=tensor_dataset, batch_size=batch_size)


class EmbeddingDatasets:
    def __init__(self, embedding_path=None, vocab_path=None, sep=" ", embedding_dim=200):
        self.vocab2index = dict()
        self.embeddings = None
        self.embedding_dim = embedding_dim
        self.vocab_size = None
        self.sep = sep
        self.embedding_path = embedding_path
        self.vocab_path = vocab_path
        self.init()

    def init(self):
        if self.embedding_path is not None:
            self._load_pre_train_embedding(self.embedding_path, self.sep)
        else:
            self._load_vocab(self.vocab_path, self.sep)

    def _load_pre_train_embedding(self, path, sep=" "):
        self.embeddings = list()
        with open(path) as f:
            counter = itertools.count(start=0)
            for rline in tqdm(f.readlines()):
                line = rline.strip().split(sep)
                if len(line) > 30 and len(line[1:]) == self.embedding_dim:
                    self.vocab2index[line[0]] = next(counter)
                    self.embeddings.append([float(i) for i in line[1:]])
        self.embedding_dim = len(self.embeddings[0])
        self.vocab_size = len(self.embeddings)

        if "PAD" not in self.vocab2index.keys() and "[PAD]" not in self.vocab2index.keys():
            self.vocab2index["[PAD]"] = self.vocab_size
            self.vocab_size += 1
            if self.embeddings is not None:
                self.embeddings.append([0.0] * self.embedding_dim)

        if "UNK" not in self.vocab2index.keys() and "[UNK]" not in self.vocab2index.keys():
            self.vocab2index["[UNK]"] = self.vocab_size
            self.vocab_size += 1
            if self.embeddings is not None:
                self.embeddings.append([0.0] * self.embedding_dim)

        self.embeddings = torch.tensor(self.embeddings)

    def _load_vocab(self, path, sep=" "):
        corpus = ""
        with open(path) as f:
            for line in tqdm(f.readlines()):
                corpus += str(line.strip().replace(sep, ""))
        self.vocab2index = dict((v, i) for i, v in enumerate(corpus))
        if "PAD" not in self.vocab2index.keys() and "[PAD]" not in self.vocab2index.keys():
            self.vocab2index["[PAD]"] = len(self.vocab2index.keys())
        if "UNK" not in self.vocab2index.keys() and "[UNK]" not in self.vocab2index.keys():
            self.vocab2index["[UNK]"] = len(self.vocab2index.keys())
        self.vocab_size = len(self.vocab2index.keys())
        self.embeddings = np.random.rand(self.vocab_size, self.embedding_dim)


if __name__ == "__main__":
    # train, test = FasttextDatasets(path="../../data/train_data.txt", sep="\t") \
    #     .load(n_jobs=-1, train_size=0.9)
    #
    #
    # train2, test2 = AlbertDatasets(path="../../data/train_data.txt",
    #                                pre_trained_path="../res/albert_tiny_bright/") \
    #     .load(n_jobs=1, train_size=0.9, batch_size=64)
    #
    # data = AlbertDatasets(path="../../data/train_data.txt",
    #                                pre_trained_path="../res/albert_tiny_bright/") \
    #     .loadp(n_jobs=1, max_length=365, label2index={"1":"2"})

    emb = EmbeddingDatasets("../res/embeddings/tencent_glove.6B.200d.txt")
    print(emb.vocab_size, emb.embedding_dim)
