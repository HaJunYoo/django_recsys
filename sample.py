from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
import pandas as pd
import numpy as np
import ast
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from predict.models import PredResults

##이미지 featur를 저장한 csv파일을 불러오는 함수
def load_img_feature():
    img_feature = pd.read_csv("predict/data/img_feature.csv", index_col=0)
    return img_feature

#   코사인유사도를 구한 행렬을 역순으로 정렬 -> 유사도가 높은 순의 인덱스로 정렬됨
#   시간 복잡도가 제일 오래 걸림 => 여기서 시간을 제일 많이 소모됨 => O(n*logn)
#   sim_sorted_ind=sim.argsort()[:,::-1]
sim_sorted_ind = np.load('predict/data/sim_sorted.npy')
df = pd.read_csv('predict/data/merged_df_review.csv', index_col='Unnamed: 0')
img_df = load_img_feature()


## 이미지 유사도로 검색 - 총 100개의 유사 제품 리스트 return
# input data: 검색 시 텍스트 유사도 검색 시 가장 관련 있는 제품명
# 유사도 측정 방식: 유클리디안 거리: no1(input data의 feature)과 items(no1을 포함한 모든 아이템의 feature) 사이의 거리 측정
# output data: 이미지 기준 input data와 가장 유사한 제품 100개 리스트 (제품명만)

def img_sim(img_feature, name):
    ## L2 norm 방식으로 유사도 측정
    # input data name의 feature 불러오기
    no1 = img_feature.loc[img_feature['name'] == name, "0":"4095"].values
    items = img_feature.loc[:, "0":"4095"].values

    # 이미지 유사도 거리 계산
    dists = np.linalg.norm(items - no1, axis=1)

    # 유클리디안 거리가 가장 가까운 200개의 상품명 리스트 추출
    idxs = np.argsort(dists)[:250]
    scores = [img_feature.loc[idx, "name"] for idx in idxs]

    return scores


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

    return (tmp)


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

# 상품 중복 제거
def remove_dupe_dicts(l):
  return [dict(t) for t in {tuple(d.items()) for d in l}]


# view 함수
def predict(request):

    if request.POST.get('action') == 'post':

        # Receive data from client(input)
        gender = str(request.POST.get('gender'))
        age = int(request.POST.get('age'))
        main_category = str(request.POST.get('main_category'))
        coordi = str(request.POST.get('coordi'))
        input_text = str(request.POST.get('input_text'))
        top_n = int(request.POST.get('topn'))

        # coordi, category list 화
        main_category = main_category.split(" ")
        coordi = coordi.split(" ")
        print(main_category)
        print(coordi)

        # Make prediction
        try :
            result = sim_clothes(main_category, coordi, age, gender, input_text, sim_sorted_ind, top_n)
            print(result)
            classification = result[['name', 'img', 'review', 'price']]
            name = list(classification['name'])
            img = list(classification['img'])
            review = list(classification['review'])
            price = list(classification['price'])
            print(name)

            records = PredResults.objects.all()
            records.delete()

            for i in range(len(classification)):
                PredResults.objects.create(name=name[i], img=img[i], review=review[i] ,price=price[i])

            return JsonResponse({'name': name, 'img': img}, safe=False)

        except :
            return JsonResponse({'name': "해당되는 추천이 없습니다. 다시 입력해주세요"}, safe=False)

    else :
        return render(request, 'predict.html')


# image classification view 함수
def img_predict(request):

    if request.POST.get('action') == 'post':

        input_text = str(request.POST.get('input_text'))
        top_n = int(request.POST.get('topn'))
        sub_category = str(request.POST.get('sub_category'))

        # Make prediction
        try :
            # 세부 카테고리 => word string => 200개
            category_df = df[df['sub_category'].str.contains(sub_category)]
            text_df = category_df[category_df['word_string'].str.contains(input_text)]
            text_index = text_df.index.values
            similar_indexes = sim_sorted_ind[text_index, 1:200]
            # 텍스트 유사도
            sim_df = df.iloc[similar_indexes[0]]
            name = sim_df['name'].values[0]
            print(name)
            img_score = img_sim(img_df, name)  # 200개 추출
            # print(img_score)
            # 이미지 유사도
            image_df = df[df['name'].isin(img_score)]
            # image_df = image_df[image_df['sub_category'].str.contains(sub_category)]

            temp = pd.merge(image_df, sim_df, how='inner', on='name')
            print(temp.columns)
            temp = temp.sort_values(by='scaled_rating_x',ascending=False)
            temp = temp[:top_n]
            classification = temp[['name', 'img_x', 'review_x', 'price_x']]
            name = list(classification['name'])
            img = list(classification['img_x'])
            review = list(classification['review_x'])
            price = list(classification['price_x'])
            print(name)

            records = PredResults.objects.all()
            records.delete()

            for i in range(len(classification)):
                PredResults.objects.create(name=name[i], img=img[i], review=review[i] ,price=price[i])

            return JsonResponse({'name': name, 'img': img}, safe=False)

        except :

            return JsonResponse({'name': "해당되는 추천이 없습니다. 다시 입력해주세요"}, safe=False)

    else :
        return render(request, 'image_predict.html')


def view_results(request):
    # Submit prediction and show all

    data = PredResults.objects.all()

    return render(request, "results.html", {"dataset" : data})