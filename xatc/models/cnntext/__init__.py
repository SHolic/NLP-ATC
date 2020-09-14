from ...common.utils import batch_print, set_seed
from ...common.loader import CNNTextDatasets, EmbeddingDatasets
from .model import CNNText

from pathlib import PurePath
import torch
from sklearn.metrics import accuracy_score
from transformers import AdamW, get_linear_schedule_with_warmup
import time

EMB_PATH = PurePath(__file__).parent.parent.parent / "res/embeddings/"
VOCAB_PATH = EMB_PATH / "vocab.txt"
EMB_PATH = EMB_PATH / "tencent_glove.6B.200d.txt"

cp = None
pred_model_path = None


def train(data_path,
          model_path=None,
          train_size=0.9,
          model_params=None):
    """
    :param model_params: finetune, label_weights, dropout, pre_trained_model_path,
                         batch_size, lr, eps, epochs
    """
    set_seed(2020)

    # updates params
    if not isinstance(model_params, dict):
        model_params = dict()
    finetune = model_params.get("finetune", True)
    label_weights = model_params.get("label_weights", None)
    dropout = model_params.get("dropout", 0.3)
    embedding_dim = model_params.get("embedding_dim", 200)
    vocab_path = model_params.get("vocab_path", VOCAB_PATH)
    emb_path = model_params.get("emb_path", EMB_PATH)
    emb_sep = model_params.get("emb_sep", " ")
    kernel_size = model_params.get("kernel_size", [3, 4, 5])
    kernel_num = model_params.get("kernel_num", 100)

    batch_size = model_params.get("batch_size", 64)
    lr = model_params.get("lr", 1e-4)
    eps = model_params.get("eps", 1e-8)
    epochs = model_params.get("epochs", 5)
    n_jobs = model_params.get("n_jobs", 1)

    # load training data
    emb_datasets = EmbeddingDatasets(embedding_path=emb_path, vocab_path=vocab_path, embedding_dim=embedding_dim,
                                     sep=emb_sep)

    cnn_datasets = CNNTextDatasets(path=data_path, sep="\t", vocab2idx=emb_datasets.vocab2index)
    train_data, test_data = cnn_datasets.load(n_jobs=n_jobs, train_size=train_size, batch_size=batch_size)

    total_steps = len(train_data) * epochs

    # init the albert model
    model = CNNText(num_labels=cnn_datasets.num_labels,
                    finetune=finetune,
                    label_weights=label_weights,
                    dropout=dropout,
                    embeddings=emb_datasets.embeddings,
                    kernel_size=kernel_size,
                    kernel_num=kernel_num)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, eps=eps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for name, param in model.named_parameters():
        print(name, param.shape, param.device, param.requires_grad)

    for epoch in range(epochs):
        start_time = time.time()
        # training
        model.train()
        train_loss = 0
        avg_train_loss = 0
        for i, train in enumerate(train_data):
            train_input_ids = train[0].to(device)
            train_labels = train[1].to(device)

            logits, loss = model(sent_ids=train_input_ids, labels=train_labels)
            train_loss += loss.item()
            avg_train_loss = train_loss / (i + 1)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            batch_print("[Epoch] \033[34m{:0>3d}\033[0m".format(epoch),
                        "[Batch] \033[34m{:0>5d}\033[0m".format(i),
                        "[lr] \033[34m{:0>.6f}\033[0m".format(scheduler.get_lr()[0]),
                        "[avg train loss] \033[34m{:0>.4f}\033[0m".format(avg_train_loss),
                        "[time] \033[34m{:<.0f}s\033[0m".format(time.time() - start_time),
                        flag="batch")

        # on-time evaluate model
        model.eval()
        test_loss = 0
        avg_test_loss = 0
        pred_labels, test_labels = [], []
        for i, test in enumerate(test_data):
            test_input_ids = test[0].to(device)
            test_label = test[1].to(device)

            with torch.no_grad():
                pred_label, loss = model(sent_ids=test_input_ids,
                                         labels=test_label)
                pred_labels.append(torch.argmax(pred_label.cpu(), -1).float())
                test_labels.append(torch.argmax(test_label.cpu(), -1))
                test_loss += loss
                avg_test_loss = test_loss / (i + 1)

        batch_print("[Epoch] \033[34m{:0>3d}\033[0m".format(epoch),
                    "[lr] \033[34m{:0>.6f}\033[0m".format(scheduler.get_lr()[0]),
                    "[avg train loss] \033[34m{:0>.4f}\033[0m".format(avg_train_loss),
                    "[avg test lost] \033[34m{:>0.4f}\033[0m".format(avg_test_loss),
                    "[time] \033[34m{:<.0f}s\033[0m".format(time.time() - start_time),
                    flag="epoch")

        if epoch == epochs - 1:
            acc = accuracy_score(torch.cat(pred_labels, dim=-1).numpy(), torch.cat(test_labels, dim=-1).numpy())
            print("The model test accuracy is: \033[34m{:.5}\033[0m".format(acc))

    # save model
    if model_path is not None:
        _save_model(path=model_path, model=model, vocab2index=emb_datasets.vocab2index,
                    max_length=cnn_datasets.max_length, label2index=cnn_datasets.label2idx)


def predict(data=None, data_path=None, sep="\t", model_path=None, **kwargs):
    global cp
    global pred_model_path
    if model_path != pred_model_path:
        cp = _load_model(model_path)
    if cp is None:
        cp = _load_model(model_path)
    pred_model_path = model_path

    model = cp['model']
    model.finetune = False
    vocab2index = cp['vocab2index']
    max_length = cp['max_length']
    label2index = cp['label2index']
    index2label = {v: k for k, v in label2index.items()}

    cnn_datasets = CNNTextDatasets(path=data_path, data=data, sep=sep, vocab2idx=vocab2index)
    test_data = cnn_datasets.loadp(max_length=max_length, label2index=label2index, batch_size=64)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()
    pred_labels = []
    for i, test in enumerate(test_data):
        test_input_ids = test[0].to(device)

        with torch.no_grad():
            pred_label, loss = model(sent_ids=test_input_ids)
            pred_labels.append(torch.argmax(pred_label.cpu().detach(), -1).float())

    pred_label = [index2label[index] for index in torch.cat(pred_labels, dim=-1).numpy()]
    return pred_label


class Predictor:
    def __init__(self, model_path, sep="\t"):
        cp = _load_model(model_path)
        self.model = cp["model"]
        self.model.finetune = False
        self.vocab2index = cp['vocab2index']
        self.max_length = cp['max_length']
        self.label2index = cp['label2index']
        self.index2label = {v: k for k, v in self.label2index.items()}
        self.sep = sep

    def predict(self, data=None, data_path=None):
        cnn_datasets = CNNTextDatasets(path=data_path, data=data, sep=self.sep, vocab2idx=self.vocab2index)
        test_data = cnn_datasets.loadp(max_length=self.max_length, label2index=self.label2index, batch_size=64)

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        self.model.eval()
        pred_labels = []
        for i, test in enumerate(test_data):
            test_input_ids = test[0].to(device)

            with torch.no_grad():
                pred_label, loss = self.model(sent_ids=test_input_ids)
                pred_labels.append(torch.argmax(pred_label.cpu().detach(), -1).float())

        pred_label = [self.index2label[index] for index in torch.cat(pred_labels, dim=-1).numpy()]
        return pred_label


def _save_model(path, model, vocab2index, max_length, label2index):
    torch.save({
        'model': model,
        'vocab2index': vocab2index,
        'max_length': max_length,
        'label2index': label2index
    }, path)


def _load_model(path):
    cp = torch.load(path)
    return cp
