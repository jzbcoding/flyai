# -*- coding: utf-8 -*
import argparse
import json
import numpy
import os
from flyai.dataset import Dataset
from keras.preprocessing.image import ImageDataGenerator
import keras
from model import Model
from path import DATA_PATH, MODEL_PATH
import config

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)
KERAS_MODEL_NAME = "model.h5"
# 超参
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()
# 数据获取辅助类
dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
# # 模型操作辅助类
modelpp = Model(dataset)

train_size = dataset.get_train_length()
val_size = dataset.get_validation_length()
print("train size:" + str(train_size))
print("test size:" + str(val_size))
steps_per_epoch = int((train_size - 1) / args.BATCH) + 1
print("steps_per_epoch:", steps_per_epoch)
def get_train_generator():
    while 1:
        yield dataset.next_train_batch()
train_generator=get_train_generator()
val_data=dataset.get_all_validation_data()

callbacks = []
ModelCheckpoint = keras.callbacks.ModelCheckpoint(os.path.join(MODEL_PATH, KERAS_MODEL_NAME), monitor='val_acc',
                                                  verbose=1,
                                                  save_best_only=True, save_weights_only=False, mode='auto', period=1)
EarlyStopping = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=config.es_patience,
                                              verbose=1, mode='auto', baseline=None, restore_best_weights=False)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1,
                                              patience=config.lr_patience, min_lr=0.000001)
callbacks.append(EarlyStopping)
callbacks.append(reduce_lr)
callbacks.append(ModelCheckpoint)
from net import get_model

model, base_model = get_model()
optimizer = keras.optimizers.adam(lr=0.001, decay=0)
if config.num_class == 2:
    loss_name = "binary_crossentropy"
else:
    loss_name = "categorical_crossentropy"
if not config.onehot:
    loss_name = "sparse_" + loss_name

model.compile(optimizer=optimizer,
              loss=loss_name,
              metrics=["acc"])

print(model.summary())
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch,
                    epochs=args.EPOCHS,
                    verbose=1, callbacks=callbacks,
                    validation_data=val_data, validation_steps=None,
                    )
