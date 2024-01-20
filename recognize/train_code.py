# -*- coding: utf-8 -*-
"""
File Name：     train_code
Description :
date：          2024/1/17 017
"""

import cv2
import joblib
import numpy as np
from io import BytesIO
from sklearn.svm import SVC
from skimage.feature import hog
from sklearn.datasets import load_files
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


def get_hog_features(X) -> np.array:
    """
    :param X: 图片数据二进制流文件
    :return: hog特征
    """
    hog_features = []
    for image in X:
        # X为包含图片的数据集列表内数据类型为二进制流
        image = BytesIO(image)  # 转换为IO流
        image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), -1)  # 转换为cv2类型图片
        image = cv2.resize(image, (50, 100))  # 统一图片大小，可提高训练准确率
        if image.ndim == 3:  # 因图片通道不全为单通道，针对此类数据进行转换
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.array(image)  # 转换为numpy数组
        fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
                 cells_per_block=(1, 1), visualize=False)  # 提取hog特征
        hog_features.append(fd)
    hog_features = np.array(hog_features)  # 整体转换为sklearn可训练特征数组
    return hog_features


def main(data_path):
    # <<<读取数据<<<
    data = load_files(rf'database/{data_path}')  # 读取文件夹内数据
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3)  # 划分测试集训练集 3：7
    # >>>结束>>>

    # <<<提取sklearn可训练特征<<<
    X_train_hog = get_hog_features(X_train)
    X_test_hog = get_hog_features(X_test)
    # >>>结束>>>

    # 分类器任选其一
    # <<<构建MLP分类器<<<
    mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(50, 50), random_state=1,
                        max_iter=10000, verbose=100, learning_rate_init=.1, tol=1e-9)
    # >>>结束>>>

    # <<<构建SVC分类器<<<
    # mlp = SVC(max_iter=100000)
    # >>>结束>>>

    # <<<开始训练<<<
    mlp.fit(X_train_hog, y_train)
    # >>>结束>>>

    # <<<获取训练结果<<<
    accuracy = mlp.score(X_test_hog, y_test)
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    # >>>结束>>>

    # <<<保存训练模型<<<
    joblib.dump(mlp, rf'./model/LPDR_{data_path}.pkl')
    # >>>结束>>>


if __name__ == '__main__':
    # main参数可选0_Z 或 ZH
    main('ZH')
