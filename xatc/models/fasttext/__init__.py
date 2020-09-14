import os
from sklearn.metrics import accuracy_score

from ...common import FasttextDatasets
from .model import FastText

ft = None
pred_model_path = None


def train(data_path,
          model_path=None,
          train_size=0.9,
          model_params=None):
    temp_train_path = "./fasttext_train.txt"
    ft_datasets = FasttextDatasets(path=data_path, sep="\t")
    train, test = ft_datasets.load(n_jobs=-1, train_size=train_size)

    # Fasttext need train_file with utf-8 encoding when training model
    ft_datasets.save(train, temp_train_path)

    # training
    ft = FastText(**model_params) if isinstance(model_params, dict) is None else FastText()
    ft.train(data_path=temp_train_path)

    # remove the models training file
    os.remove(temp_train_path)

    # prepare data for evaluate the model
    train_x, train_y, test_x, test_y = [], [], [], []
    for t in train:
        t = t.split("\t")
        train_x.append(t[0])
        train_y.append(t[-1].replace("__label__", ""))
    for t in test:
        t = t.split("\t")
        test_x.append(t[0])
        test_y.append(t[-1].replace("__label__", ""))

    # evaluate the model
    train_p, _ = ft.predict(train_x)
    test_p, _ = ft.predict(test_x)
    train_p = [p[0].replace("__label__", "") for p in train_p]
    test_p = [p[0].replace("__label__", "") for p in test_p]

    print("[Fasttext evaluation] Train data accuracy is: {:<3f}".format(accuracy_score(train_y, train_p)))
    print("[Fasttext evaluation] Test data accuracy is:  {:<3f}".format(accuracy_score(test_y, test_p)))

    # save model
    if model_path is not None:
        ft.save(model_path)

    return ft


def predict(data_path=None,
            data=None,
            model_path=None,
            return_type=None):
    # load test data
    test_data = []
    if data_path:
        test_data = FasttextDatasets(path=data_path, sep="\t").loadp(n_jobs=-1)
    if isinstance(data, str):
        test_data.append(data)
    if isinstance(data, list):
        test_data += data

    # load model
    global ft
    global pred_model_path
    if model_path != pred_model_path:
        ft = FastText.load(model_path)
    if ft is None:
        ft = FastText.load(model_path)
    pred_model_path = model_path

    # return predict_label, sentence_embedding, word_embedding according to return_type
    if return_type is None:
        pred, _ = ft.predict(test_data)
        pred = [p[0].replace("__label__", "") for p in pred]
        return pred
    if return_type == "sent_embedding":
        return [ft.model.get_sentence_vector(d) for d in test_data]
    if return_type == "word_embedding":
        return [ft.model.get_word_vector(d) for d in test_data]
    raise ValueError("return_type should be 'None', 'sent_embedding' or 'word_embedding'!")


class Predictor:
    def __init__(self, model_path):
        self.model = FastText.load(model_path)

    def predict(self, data=None, data_path=None, return_type=None):
        # load test data
        test_data = []
        if data_path:
            test_data = FasttextDatasets(path=data_path, sep="\t").loadp(n_jobs=-1)
        if isinstance(data, str):
            test_data.append(data)
        if isinstance(data, list):
            test_data += data

        if return_type is None:
            pred, _ = self.model.predict(test_data)
            pred = [p[0].replace("__label__", "") for p in pred]
            return pred
        if return_type == "sent_embedding":
            return [self.model.model.get_sentence_vector(d) for d in test_data]
        if return_type == "word_embedding":
            return [self.model.model.get_word_vector(d) for d in test_data]
        raise ValueError("return_type should be 'None', 'sent_embedding' or 'word_embedding'!")
