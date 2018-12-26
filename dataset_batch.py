#!/usr/bin/env python

import pandas as pd
import numpy as np
import math
import glob
import cv2
from matplotlib import pyplot as plt

def load_train_data(train_img_folder = './data/data/rez224/', path = './data/ntu_final_2018/train.csv',
    min_cnt = 0, max_cnt = 10001):

    train_df = pd.read_csv(path)
    img_idx, labels = train_df['Image Index'], train_df['Labels']
    label_img, labels_res = [], []

    print("Loading Training Images and Labels")
    five_peercent_len = ((max_cnt-min_cnt+1) // 100 * 5) or 1
    skip_cnt = 0
    for img_id, label in zip(img_idx, labels):
        cnt = skip_cnt+len(label_img)
        if cnt < min_cnt:
            skip_cnt += 1
            continue
        if len(label_img) % five_peercent_len == 0:
            print(len(label_img) / five_peercent_len * 5, "% done")
        if cnt > max_cnt or cnt >= 10002:
            break

        img = cv2.imread(train_img_folder + img_id)
        res = img / 255.0

        label_img.append(res)
        label = [int(x) for x in label.split(" ")]
        labels_res.append(label)

    print("Done Loading Training Images and Labels from "+str(min_cnt)+" to "+str(max_cnt))
    label_img, labels_res = np.array(label_img), np.array(labels_res)
    print(label_img.shape, labels_res.shape)
    return label_img, labels_res

def load_test_data(test_img_folder = './data/data/images/', path = './data/ntu_final_2018/test.csv',
    reshaped_size = (150, 150), min_cnt = 0, max_cnt = 33651):

    test_df = pd.read_csv(path)
    img_idx = test_df['Image Index']
    img_result = []

    print("Loading Testing Images")
    five_peercent_len = ((max_cnt-min_cnt+1) // 100 * 5) or 1
    skip_cnt = 0
    for img_id in img_idx:
        cnt = skip_cnt+len(img_result)
        if cnt < min_cnt:
            skip_cnt += 1
            continue
        if len(img_result) % five_peercent_len == 0:
            print(len(img_result) / five_peercent_len * 5, "% done")
        if cnt > max_cnt or cnt >= 33652:
            break

        img = cv2.imread(test_img_folder + img_id)
        res = cv2.resize(img, dsize = reshaped_size) / 255.0
        res = res.reshape((reshaped_size[0], reshaped_size[1], 3))

        img_result.append(res)

    print("Done Loading Testing Images from "+str(min_cnt)+" to "+str(max_cnt))
    img_result = np.array(img_result)
    print(img_result.shape)
    return img_result, np.array(img_idx[min_cnt:max_cnt+1])

def generate_batch(x, y, batch_size):
    pass

def show_img(arr):
    plt.imshow(arr, cmap=plt.get_cmap('gray'))

if __name__ == '__main__':
    load_train_data()
    pass