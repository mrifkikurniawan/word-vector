from wordvector.utils import plot_with_matplotlib
from wordvector.utils import cluster_dbscan, reduce_dimensions
from gensim.models import Word2Vec 
from glove import Glove 
import joblib

model = "word2vec"
model = "glove"
save_path = ""
model_path = "/media/user/DATA/my_repo/word-vector/logs/glove_wikipedia/wikipedia_glove.model"

if model == "glove":  
    model = Glove().load(model_path)
    vectors = model.word_vectors

elif model == "word2vec": 
    model = Word2Vec.load(model_path)
    vectors = model.wv.vectors

# clustering
labels = cluster_dbscan(vectors)

# save
joblib.dump(labels, save_path)  