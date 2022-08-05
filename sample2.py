# image classification view 함수
def img_predict(request):
    if request.POST.get('action') == 'post':

        input_text = str(request.POST.get('input_text'))
        top_n = int(request.POST.get('topn'))
        main_category = str(request.POST.get('main_category'))

        # Make prediction
        try:
            start = timer()
            # Okt가 시간이 엄청 걸림

            string_list = word_tokenize(input_text)  # 리스트 형태
            #
            vec_input = aggregate_vectors(string_list)
            # vec_input = word2vec.wv[input_text]
            print('1단계', vec_input[:10])

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

            name1 = sim_df['name'].values[0]
            name2 = sim_df['name'].values[1]
            name3 = sim_df['name'].values[2]

            print('5단계', name1)
            # 상위 3개의 그림과 비교
            img_score1 = img_sim(img_df, name1)  # 4000개 추출
            img_score2 = img_sim(img_df, name2)
            img_score3 = img_sim(img_df, name3)

            # 3개 간 교집합
            img_score = list(set(img_score1) & set(img_score2) & set(img_score3))

            print('5.5단계', img_score[:5])
            # 이미지 유사도
            image_df = df[df['name'].isin(img_score)]
            # image_df = image_df[image_df['sub_category'].str.contains(sub_category)]
            print('6단계', image_df)
            temp = pd.merge(image_df, sim_df, how='inner', on='name')
            print('6.5단계', temp)
            print('6.7단계', temp.columns)
            temp = temp.sort_values(by='scaled_rating_x', ascending=False)
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
                PredResults.objects.create(name=name[i], img=img[i], review=review[i], price=price[i])

            #### Wordcloud 만들기
            try:
                wordcloud(temp)
            except:
                pass

            end = timer()
            time = end - start

            return JsonResponse({'name': name, 'time': time}, safe=False)

        except:
            return JsonResponse({'name': "해당되는 추천이 없습니다. 다시 입력해주세요"}, safe=False)

    else:
        return render(request, 'image_predict.html', {'list': category_list})