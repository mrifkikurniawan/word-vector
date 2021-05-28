from wordvector.trainer import Word2VecTrainer
from wordvector.datasets import Wikipedia, PubMed
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

if __name__ == '__main__':
    
    # -----------------
    # WIKIPEDIA
    # -----------------
    
    # data = Wikipedia(cache_dir="/media/user/DATA/my_repo/wordvector_trainer/dataset", 
    #                  words_count="/media/user/DATA/my_repo/wordvector_trainer/dataset/wikipedia_word_count.pkl")

    # word2vec = Word2VecTrainer(dataset=data, 
    #                            pretrained="/media/user/DATA/my_repo/wordvector_trainer/embedding_pretrained/GoogleNews-vectors-negative300.bin.gz", 
    #                            unique_name="word2vec_wikipedia", 
    #                            vector_size=300,
    #                            window=5,
    #                            min_count=5,
    #                            workers=5,
    #                            sg=0,
    #                            alpha=0.025,
    #                            min_alpha=0.0001,
    #                            seed=0,
    #                            max_vocab_size=None,
    #                            max_final_vocab=2000000,
    #                            compute_loss=True)


    # -----------------
    # PUBMED
    # -----------------
     
    data = PubMed(cache_dir="/media/user/DATA/my_repo/wordvector_trainer/dataset/pubmed", 
                     words_count="/media/user/DATA/my_repo/wordvector_trainer/dataset/pubmed/pubmed_word_count.pkl"
                     )

    word2vec = Word2VecTrainer(dataset=data, 
                            pretrained="/media/user/DATA/my_repo/wordvector_trainer/embedding_pretrained/GoogleNews-vectors-negative300.bin.gz", 
                            unique_name="word2vec_pubmed", 
                            vector_size=300,
                            window=5,
                            min_count=1,
                            workers=5,
                            sg=0,
                            alpha=0.025,
                            min_alpha=0.0001,
                            seed=0,
                            max_vocab_size=None,
                            max_final_vocab=2500000,
                            compute_loss=True)

    model = word2vec.fit(epochs=3)
    word2vec.save_model("/media/user/DATA/my_repo/wordvector_trainer/logs")