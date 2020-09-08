import data_loader
import pos_tag
import build_models
from keras import models
import train
from keras.models import load_model
import pickle

# --------------------------------------------
# 데이터로드
# --------------------------------------------
path = r"C:\Users\user\Documents\GitHub\seq2seq_chatbot\dataset.csv"

d_loader = data_loader.DataLoader()

question, answer = d_loader.set_path(path)

# 데이터의 일부만 학습에 사용
# question = question[:10]
# answer = answer[:10]

# 형태소분석 수행
p_tag = pos_tag.Postag
question = p_tag.postag(question, d_loader.RE_FILTER)
answer = p_tag.postag(answer, d_loader.RE_FILTER)

# 질문과 대답 문장들을 하나로 합침
sentences = []
sentences.extend(question)
sentences.extend(answer)

# 단어 딕셔너리 생성
words = d_loader.add_tag(sentences)
d_loader.create_word_dic(words)
word_to_index, index_to_word = d_loader.return_data()

# --------------------------------------------
# 전처리
# --------------------------------------------
# 인코더 입력 인덱스 변환
x_encoder = d_loader.convert_text_to_index(question, word_to_index, d_loader.ENCODER_INPUT)

# 첫 번째 인코더 입력 출력 (12시 땡)
# print(x_encoder[0])

# 디코더 입력 인덱스 변환
x_decoder = d_loader.convert_text_to_index(answer, word_to_index, d_loader.DECODER_INPUT)

# 첫 번째 디코더 입력 출력 (START 하루 가 또 가네요)
# print(x_decoder[0])

# 디코더 목표 인덱스 변환
y_decoder = d_loader.convert_text_to_index(answer, word_to_index, d_loader.DECODER_TARGET)

# 첫 번째 디코더 목표 출력 (하루 가 또 가네요 END)
# print(y_decoder[0])

# 디코더 목표 설정
y_decoder = d_loader.one_hot(y_decoder, words)

# 첫 번째 디코더 목표 출력
# print(y_decoder[0])

# 모델로드
build = build_models.BuildModel(words, d_loader.embedding_dim, d_loader.lstm_hidden_dim)

build.train_model()

# --------------------------------------------
# 훈련 모델 정의
# --------------------------------------------

# 입력과 출력으로 함수형 API 모델 생성
model = models.Model([build.encoder_inputs, build.decoder_inputs], build.decoder_outputs)

# 학습 방법 설정
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

encoder_model, decoder_model = build.predict_model()

# --------------------------------------------
# 학습
# --------------------------------------------

# batchsize = 64
Train = train.Train(64, model, x_encoder, x_decoder, y_decoder, index_to_word)
i,j = Train.return_data()

# 인덱스를 문장으로 변환
sentence = d_loader.convert_index_to_text(i, j)
print(sentence)

# 학습 모델 테스트
check = "프로젝트"
input_seq = d_loader.make_predict_input(check, word_to_index)
sentence = d_loader.generate_text(input_seq, encoder_model, decoder_model)

# 데이터 저장
model.save('seq2seq_model.h5')
with open('w2i.pkl', 'wb') as f:
    pickle.dump(word_to_index, f)
with open('i2w.pkl', 'wb') as f:
    pickle.dump(index_to_word, f)
