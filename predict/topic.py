import pandas as pd
import numpy as np
import ast
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('predict/data/merged_df_review.csv', index_col='Unnamed: 0')


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

# age와 gender가 없을 경우까지 해보자 여기서는  "" 표시가 결측값이라고 생각하고 만듦
def sorted_age_gender(age, gender):
    if age == "":
        if gender == "":
            sorted_li = ["scaled_rating", "year"]
        else:
            sorted_li = ["scaled_rating", gender_col(gender), "year"]
    else:
        if gender == "":
            sorted_li = ["scaled_rating", age_col(age), "year"]
        else:
            sorted_li = ["scaled_rating", age_col(age), gender_col(gender), "year"]
    return sorted_li


def item_filtering(main_category, coordi, sim_df):
    filtered = sim_df[sim_df["main_category"].isin(main_category)].reset_index(drop=True)

    if coordi == [""]:
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


def sim_clothes(main_category, coordi, age, gender, clothe, sim_sorted_ind, top_n):
    # 찾고자 하는 옷들 선택 => 최대 3개의 행
    clothes = df[df['word_string'].str.contains(clothe)]
    # 찾고자 하는 옷의 인덱스 값만 추출하여 새로운 변수에 저장
    clothes_index = clothes.index.values

    # sim_indexes는 이중리스트 상태이므로 이중리스트를 해제한 후 인덱스를 이용해 해당 내용 추출
    similar_indexes = sim_sorted_ind[clothes_index, 1:]

    sim_df = df.iloc[similar_indexes[0]].sort_values(by=sorted_age_gender(age, gender), ascending=False)
    sim_df = sim_df.reset_index(drop=True)

    recsys_df = item_filtering(main_category, coordi, sim_df)[:top_n]

    return recsys_df