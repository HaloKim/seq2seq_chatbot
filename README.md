# 소개
케라스를 사용하여 Seq2Seq 챗봇 구현

# 설명
- \dataset : 데이터셋
- Seq2Seq Chatbot.ipynb : 실행 파일

# Customize
우종하 님의 seq2seq 챗봇 소스를 이용하여 Train과 Test 코드를 분리하여 작성하였습니다.
큰 용량의 .csv파일을 이용시 메모리가 부족했기 때문에 모델의 학습부분에 체크포인트를 설정했습니다.

모델은 케라스 .h5로 내보내고 딕셔너리는 .pkl파일로 두가지가 나옵니다.
word_to_index(w2i.pkl) / index_to_word(i2w.pkl) 이는 테스트 코드에서 사전생성된 딕셔너리를 이용하기 위함입니다.

# Seq2Seq 모델의 구조
<img src = "/image/image01.png">

<img src = "/image/image02.png">

<img src = "/image/image03.png">

<img src = "/image/image04.png">

<img src = "/image/image05.png">

<img src = "/image/image06.png">
