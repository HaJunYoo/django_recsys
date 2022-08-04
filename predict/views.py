from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
import pandas as pd
import numpy as np
import ast
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from predict.models import PredResults

from predict.embedding import *
from predict.list import *
from predict.topic import *

from timeit import default_timer as timer

from konlpy.tag import Okt

tokenizer = Okt()

##이미지 featur를 저장한 csv파일을 불러오는 함수
def load_img_feature():
    img_feature = pd.read_csv("predict/data/img_feature.csv", index_col=0)
    return img_feature

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

    # 유클리디안 거리가 가장 가까운 1000개의 상품명 리스트 추출
    idxs = np.argsort(dists)[:4000]
    scores = [img_feature.loc[idx, "name"] for idx in idxs]

    return scores

#############################################################
# 데이터 선언

#   코사인유사도를 구한 행렬을 역순으로 정렬 -> 유사도가 높은 순의 인덱스로 정렬됨
#   시간 복잡도가 제일 오래 걸림 => 여기서 시간을 제일 많이 소모됨 => O(n*logn)
#   sim_sorted_ind=sim.argsort()[:,::-1]
sim_sorted_ind = np.load('predict/data/sim_sorted.npy')
df = pd.read_csv('predict/data/merged_df_review.csv', index_col='Unnamed: 0')
img_df = load_img_feature()

################################################################
# embeddings.py에서 함수들 임포트

###################################################

df["tags"] =  df["tags"].apply(lambda x : make_list(x))
df["review_tagged_cleaned"] =  df["review_tagged_cleaned"].apply(lambda x : make_list(x))
df["coordi"] =  df["coordi"].apply(lambda x : make_list(x))

#
# col_dict = {0:"name", 1:"main_category", 2:"sub_category", 3: "brand", 4:"tags", 5:"coordi"}
#
# word = []
# for idx, row in df.iterrows():
#     temp = []
#     for i in range(0, 6):
#         if type(row[col_dict[i]]) is list:
#             #             print(f'col_dict{i} is an list')
#             temp.extend(row[col_dict[i]])
#         else:
#             temp.append(row[col_dict[i]])
#     word.append(temp)
#
#
# mean_vector = np.array([vectors(x) for x in word])
# print(mean_vector.shape)
# print(word2vec.wv['스트릿'])

##############################
# topic modeling
# topic.py

############################
# 상품 중복 제거
def remove_dupe_dicts(l):
  return [dict(t) for t in {tuple(d.items()) for d in l}]


def wordcloud(wc_df):
    #### Wordcloud 만들기
    from wordcloud import WordCloud
    from collections import Counter
    string_list = wc_df['review_tagged_cleaned']
    print(string_list)
    word_list = []
    for words in string_list:
        for word in words:
            if len(word) > 1:
                word_list.append(word)
    # 가장 많이 나온 단어부터 40개를 저장한다.
    counts = Counter(word_list)
    tags = counts.most_common(40)
    font = 'static/fonts/NanumSquareL.otf'

    word_cloud = WordCloud(font_path=font, background_color='black', max_font_size=400,
                           colormap='prism').generate_from_frequencies(dict(tags))
    word_cloud.to_file('static/무신사.png')
    # 사이즈 설정 및 화면에 출력
    ####

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

            try :
                wordcloud(result)
            except :
                pass

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
        main_category = str(request.POST.get('main_category'))

        # Make prediction
        try :
            start = timer()
            # Okt가 시간이 엄청 걸림

            string_list = word_tokenize(input_text) # 리스트 형태
            #
            vec_input = aggregate_vectors(string_list)
            # vec_input = word2vec.wv[input_text]
            print('1단계', vec_input[:10])
            end = timer()
            time = end - start

            mean_vector = np.load('predict/data/mean_vector.npy')

            cosine_sim = []
            for idx, vec in enumerate(mean_vector):
                vec1 = vec_input.reshape(1, -1)
                vec2 = vec.reshape(1, -1)
                cos_sim = cosine_similarity(vec1, vec2)[0][0]
                cosine_sim.append((idx, cos_sim))
            print('1.2단계', cosine_sim[:10])

            temp_sim = []
            for elem in cosine_sim:
                temp_sim.append(elem[1])

            temp_df = df.copy()
            temp_df['wv_cosine'] = temp_sim

            df1 = df[df['main_category'].str.contains(main_category)]
            print('2단계', df1[:10])
            df2 = temp_df.sort_values(by='wv_cosine', ascending=False)
            print('3단계', df2)
            sim_df = pd.merge(df1, df2, how='inner', on='name')
            sim_df = sim_df.sort_values(by='wv_cosine', ascending=False)
            print('4단계', sim_df)
            name = sim_df['name'].values[0]
            print('5단계', name)

            img_score = img_sim(img_df, name)  # 4000개 추출
            print('5.5단계', img_score[:5])
            # 이미지 유사도
            image_df = df[df['name'].isin(img_score)]
            # image_df = image_df[image_df['sub_category'].str.contains(sub_category)]
            print('6단계', image_df)
            temp = pd.merge(image_df, sim_df, how='inner', on='name')
            print('6.5단계', temp)
            print('6.7단계', temp.columns)
            temp = temp.sort_values(by='scaled_rating_x',ascending=False)
            temp = temp[:top_n]

            print('6.8단계', temp)

            temp = temp.sort_values(by='wv_cosine', ascending=False)
            # print(temp.columns)
            print('6.9단계', temp)
            '''
            'name', 'main_category', 'sub_category', 'brand', 'number', 'tags',
           'price', 'rating', 'rating_num', 'season', 'gender', 'like', 'view',
           'sale', 'coordi', 'age18', 'age19_23', 'age24_28', 'age29_33',
           'age34_39', 'age40', 'man', 'woman', 'img', '_1', 'year', 'only_season',
           'scaled_rating', 'review_tagged_cleaned', 'word', 'word_string',
           'review'
            '''
            # print('7단계', type(temp['review_tagged_cleaned'][0]))



            classification = temp[['name', 'img_x', 'review_x', 'price_x']]
            print('8단계', classification)
            name = list(classification['name'])
            img = list(classification['img_x'])
            review = list(classification['review_x'])
            price = list(classification['price_x'])
            print('9단계', name)
            print('9.2단계', img)

            records = PredResults.objects.all()
            records.delete()

            for i in range(len(classification)):
                PredResults.objects.create(name=name[i], img=img[i], review=review[i] ,price=price[i])


            #### Wordcloud 만들기
            try : wordcloud(temp)
            except : pass

            return JsonResponse({'name': name, 'time' : time}, safe=False)

        except :
            return JsonResponse({'name': "해당되는 추천이 없습니다. 다시 입력해주세요"}, safe=False)

    else :
        return render(request, 'image_predict.html', {'list': category_list})



def view_results(request):
    # Submit prediction and show all

    data = PredResults.objects.all()

    return render(request, "results.html", {"dataset" : data})

def view_wordcloud(request):
    return render(request, "wordcloud.html")