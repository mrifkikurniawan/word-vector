from typing import Any, Callable, Dict, List, Optional, Tuple
import os.path as osp
from collections import Counter
import re
import pickle

from torch.utils.data import Dataset
from torchtext.data.utils import get_tokenizer
from datasets import load_dataset

import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

tokenizer = get_tokenizer('basic_english')
        

class Wikipedia(Dataset):
    def __init__(self, cache_dir: str="/media/user/DATA/my_repo/wordvector_trainer/dataset", words_count: str=None, max_corpus:int=100, **dataset_cfg):
        super(Wikipedia, self).__init__()
        
        self.cache_dir = cache_dir
        self.max_corpus = max_corpus
        self.dataset = load_dataset("wikipedia", "20200501.en", cache_dir=self.cache_dir, **dataset_cfg)["train"]
        
        if words_count:
            print(f"load word count from {words_count}")
            self.token_counter = pickle.load(open(words_count, 'rb'))
        else:
            print("Pre-trained word count is not available, count from scratch")
            self.token_counter = self.__count_words()
            f = open(f"{self.cache_dir}/wikipedia_word_count.pkl", 'wb')
            pickle.dump(self.token_counter, f, pickle.HIGHEST_PROTOCOL)
            print(f"Save word count to {self.cache_dir}/wikipedia_word_count.pkl")
            
        
        print(f"Total Dataset Sentence: {len(self.dataset)}")
        print(f"Total token count: {len(self.token_counter)}")
                
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        text = self.dataset[index]['text']
        tokenized, _ = preprocess(text)
                
        return tokenized
            
    def get_counter(self):
        return self.token_counter
    
    def __count_words(self):
        print("Count word in corpus")
        counter = Counter()
        
        for text in self.dataset:
            text = text['text']
            token = tokenizer(text)
            counter.update(token)
        
        return counter
    
    def __iter__(self):
        for line in self.dataset:
            line = line['text']
            tokenized, _ = preprocess(line)
            yield tokenized



class PubMed(Dataset):
    def __init__(self, cache_dir: str="/media/user/DATA/my_repo/wordvector_trainer/dataset/pubmed", words_count: str=None, **dataset_cfg):
        super(PubMed, self).__init__()
        
        self.cache_dir = cache_dir
        self.dataset = load_dataset('text', data_files=osp.join(self.cache_dir, "pubmed_sentence_nltk_80mil.txt"), cache_dir=self.cache_dir, **dataset_cfg)["train"]
        
        if words_count:
            print(f"load word count from {words_count}")
            self.token_counter = pickle.load(open(words_count, 'rb'))
        else:
            print("Pre-trained word count is not available, count from scratch")
            self.token_counter = self.__count_words()
            f = open(f"{self.cache_dir}/pubmed_word_count.pkl", 'wb')
            pickle.dump(self.token_counter, f, pickle.HIGHEST_PROTOCOL)
            print(f"Save word count to {self.cache_dir}/pubmed_word_count.pkl")
            
        print(f"Total Dataset Sentence: {len(self.dataset)}")
        print(f"Total token count: {len(self.token_counter)}")
        
                
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        text = self.dataset[index]['text']
        tokenized, _ = preprocess(text)
        
        return tokenized
            
    def get_counter(self):  
        return self.token_counter
    
    def __count_words(self):
        print("Count word in corpus")
        counter = Counter()
        for text in self.dataset:
            token = tokenizer(text['text'])
            counter.update(token)
    
        return counter
    
    def __iter__(self):
        self.i = 0 
        for line in self.dataset:
            if self.i == self.max_corpus:
                break
            line = line['text']
            tokenized, _ = preprocess(line)
            self.i += 1
            yield tokenized
            
        
        
        
def preprocess(text: str):
    text = text.lower()
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    tokenized = tokenizer(text)
    sentence = ' '.join(tokenized)
    sentence = re.sub(r"\s's\b", "'s", sentence)
    
    return tokenized, sentence

