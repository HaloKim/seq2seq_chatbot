import pandas as pd
import re
import pickle
import numpy as np
import pos_tag


class DataLoader:

    def __init__(self):

        # 태그 단어
        self.PAD = "<PADDING>"   # 패딩
        self.STA = "<START>"     # 시작
        self.END = "<END>"       # 끝
        self.OOV = "<OOV>"       # 없는 단어(Out of Vocabulary)

        # 태그 인덱스
        self.PAD_INDEX = 0
        self.STA_INDEX = 1
        self.END_INDEX = 2
        self.OOV_INDEX = 3

        # 데이터 타입
        self.ENCODER_INPUT = 0
        self.DECODER_INPUT = 1
        self.DECODER_TARGET = 2

        # 한 문장에서 단어 시퀀스의 최대 개수
        self.max_sequences = 30

        # 임베딩 벡터 차원
        self.embedding_dim = 100

        # LSTM 히든레이어 차원
        self.lstm_hidden_dim = 128

        # 정규 표현식 필터
        self.RE_FILTER = re.compile("[.,!?\"':;~()]")

        self.word_to_index = {}
        self.index_to_word = {}

        self.sentences = []

    @staticmethod
    def set_path(path):

        # 챗봇 데이터 로드
        chatbot_data = pd.read_csv(path, encoding='utf-8')
        question, answer = list(chatbot_data['Q']), list(chatbot_data['A'])
    
        return question, answer

    def add_tag(self, sentences):

        words = []

        # 단어들의 배열 생성
        for sentence in sentences:
            for word in sentence.split():
                words.append(word)

        # 길이가 0인 단어는 삭제
        words = [word for word in words if len(word) > 0]

        # 중복된 단어 삭제
        words = list(set(words))

        # 제일 앞에 태그 단어 삽입
        words[:0] = [self.PAD, self.STA, self.END, self.OOV]

        return words

    def create_word_dic(self, words):
        # 단어와 인덱스의 딕셔너리 생성
        self.word_to_index = {word: index for index, word in enumerate(words)}
        self.index_to_word = {index: word for index, word in enumerate(words)}

        # 워드 딕셔너리 저장
        with open('w2i.pkl', 'wb') as f:
            pickle.dump(self.word_to_index, f)
        with open('i2w.pkl', 'wb') as f:
            pickle.dump(self.index_to_word, f)

    def return_data(self):
        return self.word_to_index, self.index_to_word

    # ==========
    # 전처리
    # ==========

    # 문장을 인덱스로 변환
    def convert_text_to_index(self, sentences, vocabulary, type):

        sentences_index = []

        # 모든 문장에 대해서 반복
        for sentence in sentences:
            sentence_index = []

            # 디코더 입력일 경우 맨 앞에 START 태그 추가
            if type == self.DECODER_INPUT:
                sentence_index.extend([vocabulary[self.STA]])

            # 문장의 단어들을 띄어쓰기로 분리
            for word in sentence.split():
                if vocabulary.get(word) is not None:
                    # 사전에 있는 단어면 해당 인덱스를 추가
                    sentence_index.extend([vocabulary[word]])
                else:
                    # 사전에 없는 단어면 OOV 인덱스를 추가
                    sentence_index.extend([vocabulary[self.OOV]])

            # 최대 길이 검사
            if type == self.DECODER_TARGET:
                # 디코더 목표일 경우 맨 뒤에 END 태그 추가
                if len(sentence_index) >= self.max_sequences:
                    sentence_index = sentence_index[:self.max_sequences - 1] + [vocabulary[self.END]]
                else:
                    sentence_index += [vocabulary[self.END]]
            else:
                if len(sentence_index) > self.max_sequences:
                    sentence_index = sentence_index[:self.max_sequences]

            # 최대 길이에 없는 공간은 패딩 인덱스로 채움
            sentence_index += (self.max_sequences - len(sentence_index)) * [vocabulary[self.PAD]]

            # 문장의 인덱스 배열을 추가
            sentences_index.append(sentence_index)

        return np.asarray(sentences_index)

    def one_hot(self, y_decoder, words):
        # 원핫인코딩 초기화
        one_hot_data = np.zeros((len(y_decoder), self.max_sequences, len(words)))

        # 디코더 목표를 원핫인코딩으로 변환
        # 학습시 입력은 인덱스이지만, 출력은 원핫인코딩 형식임
        for i, sequence in enumerate(y_decoder):
            for j, index in enumerate(sequence):
                one_hot_data[i, j, index] = 1

        return one_hot_data

    # 인덱스를 문장으로 변환
    def convert_index_to_text(self, indexs, vocabulary):

        sentence = ''

        # 모든 문장에 대해서 반복
        for index in indexs:
            if index == self.END_INDEX:
                # 종료 인덱스면 중지
                break
            if vocabulary.get(index) is not None:
                # 사전에 있는 인덱스면 해당 단어를 추가
                sentence += vocabulary[index]
            else:
                # 사전에 없는 인덱스면 OOV 단어를 추가
                sentence.extend([vocabulary[self.OOV_INDEX]])

            # 빈칸 추가
            sentence += ' '

        return sentence

    # 예측을 위한 입력 생성
    def make_predict_input(self, sentence, word_to_index):
        self.sentences.append(sentence)
        self.sentences = pos_tag.Postag.postag(self.sentences, self.RE_FILTER)
        input_seq = self.convert_text_to_index(self.sentences, word_to_index, self.ENCODER_INPUT)

        return input_seq

    # 텍스트 생성
    def generate_text(self, input_seq, encoder_model, decoder_model):

        # 입력을 인코더에 넣어 마지막 상태 구함
        states = encoder_model.predict(input_seq)

        # 목표 시퀀스 초기화
        target_seq = np.zeros((1, 1))

        # 목표 시퀀스의 첫 번째에 <START> 태그 추가
        target_seq[0, 0] = self.STA_INDEX

        # 인덱스 초기화
        indexs = []

        # 디코더 타임 스텝 반복
        while 1:
            # 디코더로 현재 타임 스텝 출력 구함
            # 처음에는 인코더 상태를, 다음부터 이전 디코더 상태로 초기화
            decoder_outputs, state_h, state_c = decoder_model.predict(
                [target_seq] + states)

            # 결과의 원핫인코딩 형식을 인덱스로 변환
            index = np.argmax(decoder_outputs[0, 0, :])
            indexs.append(index)

            # 종료 검사
            if index == self.END_INDEX or len(indexs) >= self.max_sequences:
                break

            # 목표 시퀀스를 바로 이전의 출력으로 설정
            target_seq = np.zeros((1, 1))
            target_seq[0, 0] = index

            # 디코더의 이전 상태를 다음 디코더 예측에 사용
            states = [state_h, state_c]

        # 인덱스를 문장으로 변환
        sentence = self.convert_index_to_text(indexs, self.index_to_word)

        return sentence
