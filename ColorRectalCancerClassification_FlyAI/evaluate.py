import sys

import json
import random

import requests
from flyai.dataset import Dataset
from flyai.processor.download import check_download
from flyai.source.base import DATA_PATH
from model import Model


def get_json(url, is_log=False):
    try:
        response = requests.get(url=url)
        if is_log:
            print("server code ", response, url)
        return response.json()
    except:
        return None


'''
不需要修改

'''
if "https" in sys.argv[1]:
    data_id = sys.argv[1].split('/')[4]
else:
    data_id = sys.argv[1]
data_path = get_json("https://www.flyai.com/get_evaluate_command?data_id=" + data_id)

dataset = Dataset()
check_download(data_path['command'], DATA_PATH, is_print=False)
x_test, y_test = dataset.evaluate_data_no_processor("validation.csv")

randnum = random.randint(0, 100)
random.seed(randnum)
random.shuffle(x_test)

random.seed(randnum)
random.shuffle(y_test)

model = Model(dataset)
labels = model.predict_all(x_test)
eval = 0
'''
if不需要修改

'''
if len(y_test) != len(labels):
    result = dict()
    result['score'] = 0
    result['label'] = "评估违规"
    result['info'] = ""
    print(json.dumps(result))
else:
    '''
    在下面实现不同的评估算法
    '''
    for index in range(len(labels)):
        label = labels[index]
        test = y_test[index]
        if 'labels' in str(label):
            label = label['labels']
        if label == test['labels']:
            eval = eval + 1
    eval = eval / len(labels)
    result = dict()
    result['score'] = round(eval * 100, 2)
    result['label'] = "分数为准确率"
    result['info'] = ""
    print(json.dumps(result))
