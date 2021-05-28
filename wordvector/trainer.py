from math import exp
import os.path as osp
import logging
import numpy as np
import gc

import torch
from torch.utils.data import Dataset
import gensim.downloader as api
from gensim.models import Word2Vec 

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

class Word2VecTrainer():
    def __init__(self, dataset: Dataset, pretrained: str=None, unique_name: str="word2vec", **model_args):
        # --------------
        # params
        # --------------
        super(Word2VecTrainer, self).__init__()
        self.pretrained = pretrained
        self.dataset = dataset
        self.unique_name = unique_name
        
        # --------------
        # model
        # --------------
        self.model = Word2Vec(**model_args)
        
        # build model vocab 
        self.model.build_vocab_from_freq(self.dataset.get_counter())
        
        # load pre-trained
        if self.pretrained:
            print(f"Load pre-trained model from {self.pretrained}")
            self.set_lockf(trainable=True)
            
            self.model.wv.intersect_word2vec_format(self.pretrained, lockf=1.0, binary=True)
        
        print(f"Total model vocab: {len(self.model.wv)}")
        gc.collect()
        
        
    def fit(self, **train_cfg):  
        
        # --------------
        # train
        # --------------
        self.model.train(self.dataset, total_examples=len(self.dataset), **train_cfg)        
        return self.model
    
    
    def save_model(self, path: str):
        
        # --------------
        # save model
        # --------------
        save_path = osp.join(path, f"{self.unique_name}.model")
        self.model.save(save_path)
        print(f"save model to {save_path}")
    
    
    def set_lockf(self, trainable=True):
        num_vocab = len(self.model.wv)
        if trainable:
            self.model.wv.vectors_lockf = np.ones(num_vocab, dtype=np.int)
        else:
            self.model.wv.vectors_lockf = np.zeros(num_vocab, dtype=np.int)
        
    

class Glove():
    def __init__(self, dataset: Dataset, pretrained: str=None, unique_name: str="word2vec", embedding_dim: int = 300, 
                 workers:int=4, **args):
        super(Glove, self).__init__()
        
        