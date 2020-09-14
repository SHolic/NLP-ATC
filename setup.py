# coding:utf-8

from setuptools import setup

# or
# from distutils.core import setup

setup(
    name='xatc',
    version='1.0',
    description='craiditx automatic text classification wheel~',
    author='baojunshan',
    author_email='baojs@craiditx.com',
    url='https://git.creditx.com/baojs/NLP-ATC',
    packages=['xatc'],
    requires=['joblib~=0.13.2',
              'jieba~=0.39',
              'torch~=1.4.0',
              'numpy~=1.17.0',
              'tqdm~=4.32.2',
              'sklearn~=0.0',
              'scikit-learn~=0.23.1',
              'transformers~=2.10.0',
              'fasttext~=0.9.2']
)
