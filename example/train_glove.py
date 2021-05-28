from __future__ import print_function
import argparse
import pprint
import gensim

from glove import Glove
from glove import Corpus

if __name__ == '__main__':
    
    # -----------------
    # WIKIPEDIA
    # -----------------

    # Train the GloVe model and save it to disk.

    # Try to load a corpus from disk.
    save_path = '/media/user/DATA/my_repo/wordvector_trainer/logs/glove_wikipedia/wikipedia_glove.model'
    print('Reading corpus statistics')
    corpus_model = Corpus.load('/media/user/DATA/my_repo/wordvector_trainer/dataset/wikipedia_glove_corpus.model')


    # -----------------
    # PUBMED
    # -----------------
    
    # Train the GloVe model and save it to disk.

    # Try to load a corpus from disk.
    # save_path = '/media/user/DATA/my_repo/wordvector_trainer/logs/glove_pubmed/pubmed_glove.model'
    # print('Reading corpus statistics')
    # corpus_model = Corpus.load('/media/user/DATA/my_repo/wordvector_trainer/dataset/pubmed_glove_corpus.model')
    


    # -----------------
    # Training
    # -----------------
    
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)

    print('Training the GloVe model')

    glove = Glove(dictionary=corpus_model.dictionary,
                  no_components=300, 
                  learning_rate=0.05,
                  alpha=0.75,
                  max_count=100,
                  max_loss=10.0, 
                  random_state=0,
                  pretrained="/media/user/DATA/my_repo/wordvector_trainer/embedding_pretrained/glove.840B.300d.txt")
    
    glove.fit(corpus_model.matrix, 
              epochs=5,
              no_threads=8, 
              verbose=True)
    
    glove.save(save_path)
    glove.save('/media/user/DATA/my_repo/wordvector_trainer/logs/glove_pubmed/pubmed_glove.model')
    