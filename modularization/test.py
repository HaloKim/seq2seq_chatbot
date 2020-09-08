import data_loader
import pos_tag
import test_models

# --------------------------------------------
# 데이터로드
# --------------------------------------------
d_loader = data_loader.DataLoader()


# 형태소분석 수행
p_tag = pos_tag.Postag

# 단어 딕셔너리 생성
word_to_index, index_to_word = d_loader.set_word_dic()

# 첫 번째 디코더 목표 출력
# print(y_decoder[0])

# 모델로드
build = test_models.BuildModel(2345, d_loader.embedding_dim, d_loader.lstm_hidden_dim)

model = build.train_model()
model.load_weights("seq2seq_model.h5")

encoder_model, decoder_model = build.predict_model()

# 학습 모델 테스트
check = "프로젝트"
input_seq = d_loader.make_predict_input(check, word_to_index)
sentence = d_loader.generate_text(input_seq, encoder_model, decoder_model)
print(sentence)
