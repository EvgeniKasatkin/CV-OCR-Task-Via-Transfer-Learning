from IPython.core.interactiveshell import InteractiveShell
from torchvision import transforms, datasets, models
import torch
from torch import optim, cuda
from torch.utils.data import DataLoader, sampler
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
import pandas as pd
import os
from PIL import Image
from torchsummary import summary
from timeit import default_timer as timer
import traceback
import sys


def process_image(image_path):

    image = Image.open(image_path)
    img = image.resize((256, 256))
    width = 256
    height = 256
    new_width = 224
    new_height = 224
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))
    img = np.array(img).transpose((2, 0, 1)) / 256

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img = img - means
    img = img / stds
    img_tensor = torch.Tensor(img)
    return img_tensor

def predict(image_path, model, topk=5):

    real_class = image_path.split('/')[-2]
    img_tensor = process_image(image_path)
    img_tensor = img_tensor.view(1, 3, 224, 224)
    with torch.no_grad():
        model.eval()
        out = model(img_tensor)
        ps = torch.exp(out)
        topk, topclass = ps.topk(topk, dim=1)
        top_classes = [
            model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
        ]
        top_p = topk.cpu().numpy()[0]
        return img_tensor.cpu().squeeze(), top_p, top_classes, real_class

def probability_choice(x):
    val = list(x.values())
    keys = list(list(x.keys()))
    if keys[0] == 'term_1_yes' and val[0] > 0.9:
        try:
            prob_pr1 = x['term_1_no']
            prob_pr2 = x['term_1_no2']
            if prob_pr1 < 0.08 or prob_pr2 < 0.05:
                return 'term_1_yes'
            else:
                return 'Hand Check'
        except:
            return 'term_1_yes'

    elif (keys[0] == 'term_1_no') or (keys[0] == 'term_1_no2') or (keys[0] == 'term_2_no1') or (
            keys[0] == 'term_4_no') or (keys[0] == 'term_3_no'):
        return keys[0]

    elif keys[0] == 'term_2_yes' and val[0] > 0.9:
        return keys[0]

    elif keys[0] == 'term_3_yes' and val[0] > 0.9:
        return keys[0]

    elif keys[0] == 'term_4_yes' and val[0] > 0.9:
        return keys[0]

    elif keys[0] == 'term_1_yes' and val[0] < 0.9 and (keys[1] == 'term_1_no' or keys[1] == 'term_1_no2'):
        prob = val[1]
        if prob >= 0.25:
            return keys[1]

    elif prob <= 0.25 and prob >= 0.07:
        return 'Hand Check'

    else:
        return keys[0]
    return keys[0]


def check_exist(data_exist):
    try:
        if data_exist['no'] >= 0.2:
            return 'no'
        elif data_exist['no'] < 0.2 and data_exist['no'] > 0.1:
            return 'undefined'
        return 'yes'
    except:
        return 'undefined'


def check_promo(data_promo):
    try:
        if data_promo['yes'] >= data_promo['no'] and data_promo['yes'] > data_promo['undefined']:
            return 'yes'
        elif data_promo['undefined'] >= data_promo['no'] and data_promo['undefined'] > data_promo['yes']:
            return 'undefined'
        elif data_promo['no'] >= data_promo['yes'] and data_promo['no'] > data_promo['undefined']:
            return 'no'
        return 'undefined'
    except:
        return 'undefined'


def output_assortment_flg(pred_assortment):
    if pred_assortment[-3:] == 'yes':
        return 'yes'
    return 'no'

def predicion():

    list_of_photos = os.listdir('/recognition/input_photos/')
    try:
        list_of_files = list_of_photos
        req_id = str(list_of_files[0])
        req_id = req_id[0:req_id.find('__')]
        list_of_photos = list_of_files
        list_exist, list_assortment, list_promo, list_ex, list_as, list_pr, pred_list = [], [], [], [], [], [], []
        i = 0
        for ind_ in range(len(list_of_photos)):
            predict_assortment = predict('/recognition/input_photos/' + list_of_photos[ind_],
                                         model_assortment, topk=5)
            list_of_pred, list_of_values = [], []
            for ind_ in range(0, 5):
                list_of_pred.append(predict_assortment[1][ind_])
            for ind_ in range(0, 5):
                list_of_values.append(str(predict_assortment[2][ind_]))
            data_assortment = dict(zip(list_of_values, list_of_pred))
            list_as.append(data_assortment)
            predict_promo = predict('/recognition/input_photos/' + list_of_photos[ind_], model_promo,
                                    topk=5)
            list_of_pred, list_of_values = [], []
            for ind_ in range(0, 5):
                list_of_pred.append(predict_promo[1][ind_])
            for ind_ in range(0, 5):
                list_of_values.append(str(predict_promo[2][ind_]))
            data_promo = dict(zip(list_of_values, list_of_pred))
            list_pr.append(data_promo)
            predict_exist = predict('/recognition/input_photos/' + list_of_photos[ind_], model_exist,
                                    topk=5)
            list_of_pred, list_of_values = [], []
            for ind_ in range(0, 5):
                list_of_pred.append(predict_exist[1][ind_])
            for ind_ in range(0, 5):
                list_of_values.append(str(predict_exist[2][ind_]))
            data_exist = dict(zip(list_of_values, list_of_pred))
            list_ex.append(data_exist)

        df_predict = pd.DataFrame({'file': list_of_photos,
                            'ass': list_as,
                            'promo': list_pr,
                            'exist': list_ex
                            })
        df_predict['predict_assortment'] = df_predict['ass'].apply(lambda x: probability_vybor(x))
        df_predict['predict_exist'] = df_predict['exist'].apply(lambda x: check_exist(x))
        df_predict['predict_promo'] = df_predict['promo'].apply(lambda x: check_promo(x))
        df_predict['predict_assortment'] = df_predict['predict_assortment'].apply(lambda x: output_assortment_flg(x))

        file_string_df = df_predict[['file', 'predict_assortment', 'predict_exist', 'predict_promo']]
        return {
            'request_id': req_id,
            'result': 'success',
            'facts': [
                {'code': 'promo', 'value': file_string_df['predict_promo'][0]},
                {'code': 'assortment', 'value': file_string_df['predict_assortment'][0]},
                {'code': 'in_stock', 'value': file_string_df['predict_exist'][0]}
            ]
              }
    except Exception as e:
        return {
            'request_id': req_id,
            'result': 'failure',
            'error_code': str(traceback.format_stack()),
            'error_message': str(traceback.format_exc())
        }