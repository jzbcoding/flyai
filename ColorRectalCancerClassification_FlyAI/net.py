import keras
from keras import layers,Sequential
from keras import layers
from keras.layers import *
from keras import Model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet201,DenseNet121
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from config import size
import os
import config
model_name=config.model_name
MODEL_URL={
        "vgg16":"https://www.flyai.com/m/v0.1|vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "vgg19": "https://www.flyai.com/m/v0.1|vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "xception": "https://www.flyai.com/m/v0.4|xception_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "resnet50": "https://www.flyai.com/m/v0.2|resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "densenet121":"https://www.flyai.com/m/v0.8|densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5",
        "densenet201": "https://www.flyai.com/m/v0.8|densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5"

    }
# 必须使用该方法下载模型，然后加载
from flyai.utils import remote_helper
path = remote_helper.get_remote_date(MODEL_URL[model_name])
for rt, dirs, files in os.walk('./'):
    for f in files:
        if model_name in f:
            weight_path = os.path.join(rt, f)
            print(weight_path)

if model_name=="vgg16":
    base_model=VGG16(include_top=False, weights=weight_path, input_shape=(size, size, 3))
elif model_name=="vgg19":
    base_model=VGG19(include_top=False, weights=weight_path, input_shape=(size, size, 3))
elif model_name=="xception":
    base_model=Xception(include_top=False, weights=weight_path, input_shape=(size, size, 3))
elif model_name=="resnet50":
    base_model=ResNet50(include_top=False, weights=weight_path, input_shape=(size, size, 3))
elif model_name=="densenet121":
    base_model=DenseNet121(include_top=False, weights=weight_path, input_shape=(size, size, 3))
elif model_name=="densenet201":
    base_model=DenseNet201(include_top=False, weights=weight_path, input_shape=(size, size, 3))
else:
    base_model=VGG16(include_top=False, weights=weight_path, input_shape=(size, size, 3))

def get_model():
    model = Sequential()
    feature=base_model
    model.add(feature)
    model.add(layers.Dropout(0.2))
    model.add(layers.GlobalAveragePooling2D(name='avg_pool'))
    # model.add(layers.Flatten())
    # model.add(layers.Dropout(0.2))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(256, activation='relu'))
    # model.add(layers.Dense(256))
    model.add(layers.Dropout(0.5))
    if config.num_class==2:
        model.add(layers.Dense(1, activation='sigmoid'))
    else:
        model.add(layers.Dense(config.num_class, activation='softmax'))
    return model,feature
