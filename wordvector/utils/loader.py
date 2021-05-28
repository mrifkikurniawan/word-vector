import numpy as np

import torchtext
import torch
import gensim

def load_pretrained_embedding(pretrained_path: str, vocab: torchtext.vocab.Vocab, pretrained: str):
    print("Load pre-trained embedding")
    word2embedding = dict()
    
    # load words vector embedding
    if pretrained == "glove":
        word2embedding = load_glove(pretrained_path=pretrained_path, word2embedding=word2embedding)
    elif pretrained == "word2vec":
        word2embedding = load_word2vec(pretrained_path=pretrained_path, word2embedding=word2embedding)
    
    # create embedding tensor
    dim_embedding = len(word2embedding[0])
    embedding_tensor = torch.zeros((len(vocab), dim_embedding))
    
    # assigne pre-trained vector/random vector to document tokens
    for word in vocab.itos:
        idx = vocab[word]
        unk_tensor = np.random.normal(scale=0.6, size=(dim_embedding,))
        word_tensor = word2embedding.get(word, unk_tensor)
        embedding_tensor[idx, :] = torch.tensor(word_tensor)
    
    print(f"Sucessfully get pretrained embedding size: {embedding_tensor.shape}")
    return embedding_tensor


def load_glove(pretrained_path: str, word2embedding: dict) -> dict:
    with open(pretrained_path, 'rb') as file:
        for line in file:
            line = line.decode().split()
            word = line[0]
            vect = np.array(line[1:]).astype(np.float)
            word2embedding[word] = vect   
    
    return word2embedding 

def load_word2vec(pretrained_path: str, word2embedding: dict) -> dict:
    word2vec_emb = gensim.models.KeyedVectors.load_word2vec_format(pretrained_path, binary=True)
    tokens = word2vec_emb.index_to_key
    
    for token in tokens:
        word2embedding[token] = word2vec_emb[token]
    
    return word2embedding