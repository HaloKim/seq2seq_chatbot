from keras import layers
from keras import models


class BuildModel:

    def __init__(self, words, embedding_dim, lstm_hidden_dim):

        self.words = words
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        # 입력 문장의 인덱스 시퀀스를 입력으로 받음
        self.encoder_inputs = layers.Input(shape=(None,))
        self.encoder_states = 0

        # 임베딩 레이어
        self.decoder_embedding = layers.Embedding(len(self.words), self.embedding_dim)

        # 목표 문장의 인덱스 시퀀스를 입력으로 받음
        self.decoder_inputs = layers.Input(shape=(None,))

        # 인코더와 달리 return_sequences 를 True 로 설정하여 모든 타임 스텝 출력값 리턴
        # 모든 타임 스텝의 출력값들을 다음 레이어의 Dense()로 처리하기 위함
        self.decoder_lstm = layers.LSTM(self.lstm_hidden_dim, dropout=0.1, recurrent_dropout=0.5, return_state=True,
                                        return_sequences=True)

        # 단어의 개수만큼 노드의 개수를 설정하여 원핫 형식으로 각 단어 인덱스를 출력
        self.decoder_dense = layers.Dense(len(self.words), activation='softmax')

        self.decoder_outputs = 0

    def train_model(self):
        # --------------------------------------------
        # 훈련 모델 인코더 정의
        # --------------------------------------------

        # 임베딩 레이어
        encoder_outputs = layers.Embedding(len(self.words), self.embedding_dim)(self.encoder_inputs)

        # return_state 가 True 면 상태값 리턴
        # LSTM 은 state_h(hidden state)와 state_c(cell state) 2개의 상태 존재
        encoder_outputs, state_h, state_c = layers.LSTM(self.lstm_hidden_dim,
                                                        dropout=0.1,
                                                        recurrent_dropout=0.5,
                                                        return_state=True)(encoder_outputs)

        # 히든 상태와 셀 상태를 하나로 묶음
        self.encoder_states = [state_h, state_c]

        # --------------------------------------------
        # 훈련 모델 디코더 정의
        # --------------------------------------------

        # 임베딩 레이어
        self.decoder_outputs = self.decoder_embedding(self.decoder_inputs)

        # initial_state 를 인코더의 상태로 초기화
        self.decoder_outputs, _, _ = self.decoder_lstm(self.decoder_outputs, initial_state=self.encoder_states)

        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)

    def predict_model(self):
        # --------------------------------------------
        #  예측 모델 인코더 정의
        # --------------------------------------------

        # 훈련 모델의 인코더 상태를 사용하여 예측 모델 인코더 설정
        encoder_model = models.Model(self.encoder_inputs, self.encoder_states)

        # --------------------------------------------
        # 예측 모델 디코더 정의
        # --------------------------------------------

        # 예측시에는 훈련시와 달리 타임 스텝을 한 단계씩 수행
        # 매번 이전 디코더 상태를 입력으로 받아서 새로 설정
        decoder_state_input_h = layers.Input(shape=(self.lstm_hidden_dim,))
        decoder_state_input_c = layers.Input(shape=(self.lstm_hidden_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

        # 임베딩 레이어
        self.decoder_outputs = self.decoder_embedding(self.decoder_inputs)

        # LSTM 레이어
        self.decoder_outputs, state_h, state_c = self.decoder_lstm(self.decoder_outputs,
                                                                   initial_state=decoder_states_inputs)

        # 히든 상태와 셀 상태를 하나로 묶음
        decoder_states = [state_h, state_c]

        # Dense 레이어를 통해 원핫 형식으로 각 단어 인덱스를 출력
        self.decoder_outputs = self.decoder_dense(self.decoder_outputs)

        # 예측 모델 디코더 설정
        decoder_model = models.Model([self.decoder_inputs] + decoder_states_inputs,
                                     [self.decoder_outputs] + decoder_states)

        return encoder_model, decoder_model
