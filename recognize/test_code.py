# -*- coding: utf-8 -*-
"""
File Name：     test_code
Description :
date：          2024/1/17 017
"""
import cv2
import joblib
import numpy as np
from skimage.feature import hog

z_mapping = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L',
             'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
zh_mapping = ['云', '京', '冀', '内', '吉', '宁', '川', '新', '晋', '桂', '沪', '津', '浙', '渝', '湘', '琼', '甘',
              '皖', '粤', '苏', '蒙', '藏', '豫', '贵', '赣', '辽', '鄂', '闽', '陕', '青', '鲁', '黑']


def infer(img, model_path=r'../recognize/model/LPDR_0_Z.pkl'):
    """
    :param img: cv2类型img
    :param model_path: 预训练模型路径
    :return: 识别结果
    """
    classfier = joblib.load(model_path)  # 读取预训练模型
    if img.ndim == 3:  # 因图片通道不全为单通道，针对此类数据进行转换
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (50, 100))  # 根据训练代码重设大小
    # img = cv2.GaussianBlur(img, (5, 5), 0.5)
    # cv2.imshow('a', img)
    # cv2.waitKey(0)
    img = np.array(img)  # 转换为numpy数组
    img = [hog(img, orientations=8, pixels_per_cell=(16, 16),
               cells_per_block=(1, 1), visualize=False)]  # 提取特征
    img = np.array(img)  # 转换为特征数组
    x = str(classfier.predict(img)[0])  # 使用预训练模型进行识别
    if 'ZH' in model_path:
        return zh_mapping[int(x)]  # 返回识别结果
    else:
        return z_mapping[int(x)]


if __name__ == '__main__':
    print(infer(img=cv2.imread(r'../static/exa_cor.png'), model_path=r'../recognize/model/LPDR_ZH.pkl'))
