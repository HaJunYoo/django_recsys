import pandas as pd
import numpy as np
import ast
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

word2vec = Word2Vec.load("predict/data/word2vec.model")

df = pd.read_csv('predict/data/merged_df_최종.csv', index_col='Unnamed: 0')
sim_sorted_ind = np.load('predict/data/sim_sorted_after.npy')


# 단어 벡터 평균구하기
def vectors(embedding):
    # vector size 만큼
    tmp = np.zeros(50)
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
    vector = np.mean(product_vec, axis=0)

    return vector



def make_list(string):
    try :
        return ast.literal_eval(string)
    except :
        return list()



def age_col(age):
    if age <= 18:
        a = "age18"
    elif (age > 18) & (age <= 23):
        a = "age19_23"
    elif (age > 23) & (age <= 28):
        a = "age24_28"
    elif (age > 28) & (age <= 33):
        a = "age29_33"
    elif (age > 33) & (age <= 39):
        a = "age34_39"
    else:
        a = "age40"
    return a

def gender_col(gender):
    if gender == "M":
        a = "man"
    else:
        a = "woman"
    return a

# 키워드 유사도 높은 순으로 상품 리스트 뽑아낸다
def index_out(keyword, df):
    word = df["word"].apply(lambda x: make_list(x)).tolist()

    vec_input = aggregate_vectors(keyword)

    # load
    mean_vector = np.load('predict/data/mean_vector.npy')

    cosine_sim = []
    for idx, vec in enumerate(mean_vector):
        vec1 = vec_input.reshape(1, -1)
        vec2 = vec.reshape(1, -1)
        cos_sim = cosine_similarity(vec1, vec2)[0][0]
        cosine_sim.append((idx, cos_sim))

    # 첫번째 상품부터의 코사인 유사도
    temp_sim = []
    for elem in cosine_sim:
        temp_sim.append(elem[1])

    cosine_sim.sort(key=lambda x: -x[1])

    # 키워드 유사도 높은 순으로 상품 리스트 뽑아낸다
    # sim_sorted_ind = 아이템 * 아이템 유사도 높은 순으로 인덱스가 정렬되어 있는 matrix
    li = sim_sorted_ind[cosine_sim[0][0]]

    return li, temp_sim


def item_filtering(main_category, coordi, sim_df):
    filtered = sim_df[sim_df["main_category"].isin(main_category)].reset_index(drop=True)

    if coordi == [""]:  # coordi 값 입력 안해도 값 나오게
        filtered_both = filtered
    else:
        a = []
        for i in range(filtered.shape[0]):
            inter = list(set(ast.literal_eval(filtered.loc[i, "coordi"])) & set(coordi))
            if len(inter) >= 1:
                a.append(i)
        filtered_index = pd.DataFrame(index=a).index
        filtered_both = filtered.loc[filtered_index,].reset_index(drop=True)

    return filtered_both


def recsys(main_category, coordi, keyword):
    li, temp_sim = index_out(keyword, df)

    # 유사도 열 추가
    df['wv_cosine'] = temp_sim

    temp = df[df['word_string'].str.contains(keyword)]

    name_list = df.loc[li, 'name'].to_list()

    sim_df = temp[temp['name'].isin(name_list)]
    sim_df = sim_df.reset_index(drop=True)

    recsys_df = item_filtering(main_category, coordi, sim_df)

    recsys_df = recsys_df.sort_values(by=["wv_cosine"], ascending=False)

    return recsys_df