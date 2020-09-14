import multiprocessing as mp
import fasttext

from ...common.utils import my_timer
from ...common.base import BaseModel


class FastText(BaseModel):
    def __init__(self, **params):
        self.params = {
            "lr": 1,  # learning rate [0.1]
            "dim": 100,  # size of word vectors [100]
            "ws": 5,  # size of the context window [5]
            "epoch": 25,  # number of epochs [5]
            "minCount": 1,  # minimal number of word occurences [1]
            "minCountLabel": 0,  # minimal number of label occurences [1]
            "minn": 0,  # min length of char ngram [0]
            "maxn": 0,  # max length of char ngram [0]
            "neg": 5,  # number of negatives sampled [5]
            "wordNgrams": 4,  # max length of word ngram [1]
            "loss": 'softmax',  # loss function {ns, hs, softmax, ova} [softmax]
            "bucket": 2000000,  # number of buckets [2000000]
            "thread": mp.cpu_count() - 1,  # number of threads [number of cpus]
            "lrUpdateRate": 100,  # change the rate of updates for the learning rate [100]
            "t": 0.0001,  # sampling threshold [0.0001]
            "label": '__label__',  # label prefix ['__label__']
            "verbose": 2,  # verbose [2]
            "pretrainedVectors": ''  # pretrained word vectors (.vec file) for supervised learning []
        }
        if params:
            self.params.update(params)
        self.model = None

    @my_timer
    def train(self, data_path, quantize=True):
        self.model = fasttext.train_supervised(input=data_path, **self.params)

        if quantize:
            pass
            # self.model.quantize(data_path, thread=self.params["thread"], verbose=2, retrain=True)
        return self

    @my_timer
    def predict(self, data, k=1):
        return self.model.predict(data, k=k)

    @my_timer
    def save(self, path):
        self.model.save_model(path)

    @staticmethod
    @my_timer
    def load(path):
        ft = FastText()
        ft.model = fasttext.load_model(path)
        return ft
