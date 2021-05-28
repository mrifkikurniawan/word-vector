import os.path as osp
import pickle

from glove import Corpus
from wordvector.datasets import Wikipedia, PubMed

if __name__ == '__main__':
    
    # Vars
    target_path = "/media/user/DATA/my_repo/wordvector_trainer/dataset"
    # corpus = [["my", "name", "is", "rifki"], ["i", "am", "50", "years", "old"]]
    # corpus = [["my name is rifki"], ["i am 50 years old"]]
    
    # Build the corpus dictionary and the cooccurrence matrix.
    print('Pre-processing corpus')
    
    # Create dataset instance
    dataset = Wikipedia(cache_dir="/media/user/DATA/my_repo/wordvector_trainer/dataset", 
                        words_count="/media/user/DATA/my_repo/wordvector_trainer/dataset/wikipedia_word_count.pkl")
    dataset_modelname = "wikipedia_glove_corpus.model"
    word2id_path = open("/media/user/DATA/my_repo/wordvector_trainer/dataset/wikipedia_word_count.pkl", "rb")
    
    # dataset = PubMed(cache_dir="/media/user/DATA/my_repo/wordvector_trainer/dataset/pubmed", 
    #                     words_count="/media/user/DATA/my_repo/wordvector_trainer/dataset/pubmed_word_count.pkl")
    # dataset_modelname = "pubmed_glove_corpus.model"
    # word2id_path = open("/media/user/DATA/my_repo/wordvector_trainer/dataset/pubmed/pubmed_word_count.pkl", "rb")
    
    
    word2id = pickle.load(word2id_path)
    
    corpus_model = Corpus(dictionary=word2id)
    corpus_model.fit(dataset, window=5, ignore_missing=True)
    
    output = osp.join(target_path, dataset_modelname)
    corpus_model.save(output)
    
    print(corpus_model.matrix)
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)