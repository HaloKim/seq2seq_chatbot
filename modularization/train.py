from keras.callbacks import ModelCheckpoint
import numpy as np


class Train:

    def return_data(self):
        # 인덱스를 문장으로 변환
        return self.indexs, self.index_to_word

    def __init__(self, batch_size, model, x_encoder, x_decoder, y_decoder, index_to_word):

        self.index_to_word = index_to_word

        # 에폭 반복
        for epoch in range(10):
            print('Total Epoch :', epoch + 1)

            # 체크포인트
            filename = 'checkpoint-epoch-{}-batch-{}-trial-001.h5'.format(epoch + 1, batch_size)
            checkpoint = ModelCheckpoint(filename,  # file명을 지정합니다
                                         monitor='val_loss',  # val_loss 값이 개선되었을때 호출됩니다
                                         verbose=1,  # 로그를 출력합니다
                                         save_best_only=True,  # 가장 best 값만 저장합니다
                                         mode='auto',  # auto는 알아서 best를 찾습니다. min/max
                                         patience=1,  # epoch 10 동안 개선되지 않으면 callback이 호출됩니다
                                         )

            # 훈련 시작
            history = model.fit([x_encoder, x_decoder],
                                y_decoder,
                                epochs=100,
                                batch_size=batch_size,
                                verbose=1,
                                callbacks=[checkpoint])

            # 정확도와 손실 출력
            print('accuracy :', history.history['accuracy'][-1])
            print('loss :', history.history['loss'][-1])

            # 문장 예측 테스트
            input_encoder = x_encoder[2].reshape(1, x_encoder[2].shape[0])
            input_decoder = x_decoder[2].reshape(1, x_decoder[2].shape[0])
            results = model.predict([input_encoder, input_decoder])

            # 결과의 원핫인코딩 형식을 인덱스로 변환
            # 1축을 기준으로 가장 높은 값의 위치를 구함
            self.indexs = np.argmax(results[0], 1)

            print(self.indexs)
