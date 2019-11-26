# -*- coding: utf-8 -*
import numpy
from flyai.processor.base import Base
from flyai.processor.download import check_download
from path import DATA_PATH
from PIL import Image
from flyai.processor.base import Base
import cv2
from path import DATA_PATH
import os
import numpy as np
import config


'''
把样例项目中的processor.py件复制过来替换即可
'''

size=config.size
class Processor(Base):

    def __init__(self):
        self.img_shape = [size, size, 3]

    def input_x(self, image_path):
        img = cv2.imread(os.path.join(DATA_PATH, image_path))
        img = cv2.resize(img, (self.img_shape[0], self.img_shape[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        return img

    '''
    参数为csv中作为输入y的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。
    该方法字段与app.yaml中的output:->columns:对应
    '''

    def output_x(self, image_path):
        path = check_download(image_path, DATA_PATH)
        image = Image.open(path)
        image = image.convert("RGB")
        image = image.resize((64, 64), Image.BILINEAR)
        x_data = numpy.array(image)
        x_data = x_data.astype(numpy.float32)
        x_data = numpy.multiply(x_data, 1.0 / 255.0)
        return x_data

    '''
    参数为csv中作为输入x的一条数据，该方法会被dataset.next_train_batch()
    和dataset.next_validation_batch()多次调用。评估的时候会调用该方法做数据处理
    该方法字段与app.yaml中的input:->columns:对应
    '''

    def input_y(self, labels):
        one_hot_label = numpy.zeros([10])  ##生成全0矩阵
        one_hot_label[labels] = 1  ##相应标签位置置
        return one_hot_label

    '''
    输出的结果，会被dataset.to_categorys(data)调用
    '''

    def output_y(self, data):
        return numpy.argmax(data)