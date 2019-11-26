import os
from flyai.model.base import Base
from keras.engine.saving import load_model
import numpy as np
from path import MODEL_PATH

KERAS_MODEL_NAME = "model.h5"


class Model(Base):
    def __init__(self, data):
        self.model_path = MODEL_PATH

        self.dataset = data
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)


    def predict(self, **data):
        '''
        使用模型
        :param path: 模型所在的路径
        :param name: 模型的名字
        :param data: 模型的输入参数
        :return:
        '''
        model = load_model(os.path.join(MODEL_PATH, KERAS_MODEL_NAME))
        predict = model.predict(self.dataset.predict_data(**data))
        # index = np.argmax(predict[0])
        # return self.label_list[index]
        predict = self.dataset.to_categorys(predict)
        return predict


    def predict_all(self, datas):
        model = load_model(os.path.join(MODEL_PATH, KERAS_MODEL_NAME))
        labels = []
        for data in datas:
            predict = model.predict(self.dataset.predict_data(**data))
            # index = np.argmax(predict[0])
            # labels.append(self.label_list[index])
            predict = self.dataset.to_categorys(predict)
            labels.append(predict)
        return labels
    def save_model(self, model, path, name=KERAS_MODEL_NAME, overwrite=False):
        super().save_model(model, path, name, overwrite)
        print(os.path.join(path, name))
        model.save(os.path.join(path, name))




