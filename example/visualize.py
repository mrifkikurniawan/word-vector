from wordvector.utils import plot_with_matplotlib
from wordvector.utils import cluster_dbscan, reduce_dimensions
from gensim.models import Word2Vec 
from glove import Glove 
import joblib

model = "word2vec"
model = "glove"
cluster_path = ""
model_path = "/media/user/DATA/my_repo/word-vector/logs/glove_wikipedia/wikipedia_glove.model"


if model == "glove":
    model = Glove().load(model_path)
    vectors = model.word_vectors
    idx2key = model.inverse_dictionary
    
elif model == "word2vec":
    model = Word2Vec.load(model_path)
    vectors = model.wv.vectors
    idx2key = model.wv.index_to_key

# clustering
labels = joblib.load(cluster_path)

# reduce dimensions
x_vals, y_vals = reduce_dimensions(vectors)

plot_with_matplotlib(x_vals, y_vals, idx2key, cluster=labels)