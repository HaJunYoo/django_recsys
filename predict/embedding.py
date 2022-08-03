import pandas as pd
import numpy as np
import ast
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

word2vec = Word2Vec.load("predict/data/word2vec.model")

# 단어 벡터 평균구하기
def vectors(embedding):
    # vector size 만큼
    tmp = np.zeros(500)
    count = 0

    # embedding = list of list corpus
    for word in embedding:
        try:
            # word에 해당하는 단어를 워드투백에서 찾아 해당 벡터를 리스트에 붙힘
            # 100차원으로 형성됨
            tmp += word2vec.wv[word]
            count += 1
        except:
            pass

    tmp /= count  # 아이템 갯수로 전체 벡터를 mean해줌

    return tmp


def aggregate_vectors(string_list):
    product_vec = []
    for noun in string_list:
        try:
            product_vec.append(word2vec.wv[noun])
        except KeyError:
            continue

    return np.mean(product_vec, axis=0)


def make_list(string):
    try :
        return ast.literal_eval(string)
    except :
        return list()