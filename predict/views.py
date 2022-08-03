from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
import pandas as pd
import numpy as np
import ast
from nltk.tokenize import word_tokenize

from predict.models import PredResults

#   코사인유사도를 구한 행렬을 역순으로 정렬 -> 유사도가 높은 순의 인덱스로 정렬됨
#   시간 복잡도가 제일 오래 걸림 => 여기서 시간을 제일 많이 소모됨 => O(n*logn)
#   sim_sorted_ind=sim.argsort()[:,::-1]
sim_sorted_ind = np.load('predict/data/sim_sorted.npy')
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


def view_results(request):
    # Submit prediction and show all

    data = PredResults.objects.all()

    return render(request, "results.html", {"dataset" : data})