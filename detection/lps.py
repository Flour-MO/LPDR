# -*- coding: utf-8 -*-
"""
File Name：     main
Description :
date：          2024/1/19 018
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


def Exp_images(img2, s):
    # 灰度
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    # cv2.waitKey(0)
    # 二值化
    ret, thresh = cv2.threshold(gray, s, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)
    # cv2.waitKey(0)
    # 反色
    thresh = cv2.bitwise_not(thresh)
    # cv2.imshow("thresh0", thresh)
    # cv2.waitKey(0)

    # 腐蚀 膨胀运算
    kernel = np.ones((2, 2), np.uint8)

    # thresh = cv2.dilate(thresh, kernel, iterations=2)
    # cv2.imshow("thresh1", thresh)
    # cv2.waitKey(0)
    #
    thresh = cv2.erode(thresh, kernel, iterations=4)
    # cv2.imshow("thresh2", thresh)
    # cv2.waitKey(0)

    # 小轮廓去除
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    draw_img1 = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
    res = cv2.drawContours(draw_img1, contours, -1, (0, 0, 255), 2)
    # cv2.imshow("res3", res)
    # cv2.waitKey(0)
    fill = []
    for contour in contours:
        area = cv2.contourArea((contour))
        if area < 200:
            fill.append(contour)
            # print(area)

    thresh0 = cv2.fillPoly(thresh, fill, (255, 255, 255))
    thresh0 = cv2.bitwise_not(thresh0)

    # cv2.imshow("thresh0", thresh0)
    # cv2.waitKey(0)
    return thresh0


def White_Statistic(image):
    ptx = []  # 每行白色像素个数
    pty = []  # 每列白色像素个数
    height, width = image.shape
    # 逐行遍历
    for i in range(height):
        num = 0
        for j in range(width):
            if image[i][j] == 255:
                num = num + 1
        ptx.append(num)

    # 逐列遍历
    for i in range(width):
        num = 0
        for j in range(height):
            if image[j][i] == 255:
                num = num + 1
        pty.append(num)

    return ptx, pty


def Draw_Hist(ptx, pty):
    # 依次得到各行、列
    rows, cols = len(ptx), len(pty)
    row = [i for i in range(rows)]
    col = [j for j in range(cols)]
    # 横向直方图
    plt.barh(row, ptx, color='black', height=1)
    #       纵    横
    plt.show()
    # 纵向直方图
    plt.bar(col, pty, color='black', width=1)
    #       横    纵
    plt.show()


def Get_Lps(image):
    # 统计各行各列白色像素个数（为了得到直方图横纵坐标）-
    ptx, pty = White_Statistic(image)

    # 绘制直方图（横、纵）
    # Draw_Hist(ptx, pty)

    pty.append(0)
    flag = False
    S = E = None
    lp_ls = []
    for k, j in enumerate(pty):
        if j != 0 and not flag:
            S = k + 1
            flag = True
        elif j == 0 and flag:
            E = k - 1
            if S and E:
                lp_ls.append((S, E))
                S = E = None
            flag = False
    return lp_ls


def Get_Lp_Images(image):
    img2 = cv2.resize(image, (220, 70))
    # cv2.imshow("img2", img2)
    # cv2.waitKey(0)

    lp_ls = None
    for x in range(1, 8):
        thresh0 = Exp_images(img2, x * 25)
        lp_ls = Get_Lps(thresh0)
        if len(lp_ls) == 7:
            break
    if lp_ls:
        lp_imgs = []
        for _ in lp_ls:
            lp_imgs.append(img2[0:70, _[0]:_[1]])
        return lp_imgs


if __name__ == '__main__':
    # 读取文件
    img2 = cv2.imread("../static/exa_lps.png")
    lp_imgs = Get_Lp_Images(img2)
    cv2.imshow('ima', img2)
    [cv2.imshow(str(l), _) for l, _ in enumerate(lp_imgs)]
    cv2.waitKey(0)
