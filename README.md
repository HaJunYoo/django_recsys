## Django Musinsa Recommendation 

### 개요
- word2vec을 이용한 컨텐츠 based 추천
- 총 상품 9500개
- 리뷰 정보와 컨텐츠 정보들을 가지고 word2vec 임베딩 수행

- 토픽 
  - 코사인 유사도 => Topn df 만듦
  - rating 정렬
  
- word2vec + 이미지  
 - 둘다 겹치는 거 상위 200개 inner join
 - 코사인 유사도 정렬 => topn => DF 만듦
 - rating 정렬

---------



### Stack
  - Django
  - Bootstrap
  - word2vec => (npy 파일 이용)
  - pandas
  - sqlite

### 구현
<img src="./image1.png" width="320" height="400">

<img src="./image2.png" width="300" height="400">


------

### Word2vec 학습

![스크린샷 2022-08-05 오후 5.01.38.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/699b5947-50da-4a6b-8778-a7f38163193b/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2022-08-05_%EC%98%A4%ED%9B%84_5.01.38.png)

- 학습할 코퍼스 만들기
    - 리뷰 데이터
    - 아이템 정보들(상품명, 카테고리, 브랜드명, 상품 이름을 구성하는 단어들)
    - 코디, 태그
- 데이터 사이즈가 크지 않기 때문에 Vocab 사이즈가 작아 Out of Vocabulary (OOV) 문제가 발생
    - 이를 방지하기 위해 위와 같이 구성

- 데이터 사이즈가 크지 않기 때문에 word2vec 은닉층 사이즈는 50으로 설정
- skip gram을 채택, window 사이즈가 작을 수록 미세하게 우세하다는 논문을 참고 (window = 5)
    
    [](http://journal.dcs.or.kr/xml/19540/19540.pdf)
    
- 중심 단어 이웃에 포함되지 않는 단어 (negative sample)은 20개로 설정
    - 통상적으로 주변 단어 + 10~20개를 선택한다고 한다.
    - 아래의 공식을 적용하면 조금 더 적합한 숫자를 이끌어낼 수 있다.
        
        ![스크린샷 2022-08-05 오후 5.02.34.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5c574046-ceba-4dea-9a9a-2f3d8ab87e88/%EC%8A%A4%ED%81%AC%EB%A6%B0%EC%83%B7_2022-08-05_%EC%98%A4%ED%9B%84_5.02.34.png)
        

### 토픽 모델링

아이템을 추천해주는 함수 구성

유사도 = Cosine Similarity

모든 벡터는 word2vec 모델을 통해 산출된 수치 벡터

- **아이템 유사도 벡터 * 아이템 유사도 벡터**
    
    cosine_similarity(전체 아이템 열 벡터, 전체 아이템 열 벡터)
    
    ⇒ item마다 모든 item들에 관한 유사도 벡터 생성
    
    (8745 * 8745)
    

- **키워드(Input string) 벡터 * 아이템 유사도 벡터**
    
    (8745, )
    

 

1. 키워드를 인풋으로 넣었을 때 해당 키워드와 아이템들 간 유사도를 계산한 후 
    
    키워드와 유사성이 높은 순서대로 정렬
    

1. 아이템간 유사도를 이용하여 키워드와 유사도가 제일 높은 아이템과 가장 유사한 아이템들을 가져온다.

### Word2vec + CNN(VGG16) 추천

토픽 모델링과 동일한 word2vec 모델 사용

1. 키워드를 인풋으로 넣었을 때 해당 키워드와 아이템들 간 유사도를 계산한 후 
    
    키워드와 유사성이 높은 순서대로 정렬
    

1. 키워드와 유사성이 높은 상품 상위 3가지를 가져옵니다.
2. 아이템마다 vgg16으로 feature를 추출한 정보를 바탕으로 이미지 벡터 유사도가 높은 상품 2000개씩 가져와 교집합에 속하는 상품들을 추출합니다
3. 해당 상품들에 대한 정보를 가져와 키워드와의 코사인 유사도를 기준으로 내림차순 정렬합니다. 
4. 보고 싶은 아이템의 개수만큼 슬라이싱합니다. (Top-n)
5. 슬라이싱한 데이터를 보완한 가중 평점으로 내림차순 정렬하여 사용자에게  보여줍니다.