# -*- coding: utf-8 -*-
"""
File Name：     main
Description :
date：          2024/1/19 018
"""
import uuid

import cv2
import numpy as np
from matplotlib import pyplot as plt


# 二-7-3、纵向分割：分割字符
def Cut_Y(pty, cols, source):
    lps_img = []
    # print(pty)
    WIDTH = 32  # 经过测试，一个字符宽度约为32
    w = w1 = w2 = 0  # 前谷 字符开始 字符结束
    begin = False  # 字符开始标记
    last = 10  # 上一次的值
    con = 0  # 计数

    # 纵向切割（正式切割字符）
    for j in range(int(cols)):
        # 0、极大值判断
        if pty[j] == max(pty):
            if j < 1:  # 左边（跳过）
                w2 = j
                if begin == True:
                    begin = False
                continue

            elif j > 220:  # 右边（直接收尾）
                if begin == True:
                    begin = False
                w2 = j
                # w1 = w1 - 10 if w2 - w1 < 20 else w1
                b_copy = source[:, w1:w2]
                lps_img.append(b_copy)
                # cv2.imshow(str(uuid.uuid4()), b_copy)
                con += 1
                break

        # 1、前谷（前面的波谷）
        if pty[j] < 12 and begin == False:  # 前谷判断：像素数量<12
            last = pty[j]
            w = j

        # 2、字符开始（上升）
        elif last < 12 and pty[j] > 1:
            last = pty[j]
            w1 = j
            begin = True

        # 3、字符结束
        elif pty[j] < 1 and begin:
            begin = False
            last = pty[j]
            w2 = j
            width = w2 - w1
            # 3-1、分割并显示（排除过小情况）
            if 10 < width < WIDTH + 3:  # 要排除掉干扰，又不能过滤掉字符”1“
                # w1 = w1 - 10 if w2 - w1 < 20 else w1
                b_copy = source[:, w1:w2]
                lps_img.append(b_copy)
                # cv2.imshow(str(uuid.uuid4()), b_copy)
                # cv2.waitKey(0)
                con += 1
            # 3-2、从多个贴合字符中提取单个字符
            elif width >= WIDTH + 3:
                # 统计贴合字符个数
                num = int(width / WIDTH + 0.5)  # 四舍五入
                for k in range(num):
                    # w1和w2坐标向后移（用w3、w4代替w1和w2）
                    w3 = w1 + k * WIDTH
                    w4 = w1 + (k + 1) * WIDTH
                    b_copy = source[:, w3:w4]
                    lps_img.append(b_copy)
                    # cv2.imshow(str(uuid.uuid4()), b_copy)
                    con += 1

        # 4、分割尾部噪声（距离过远默认没有字符了）
        elif begin is False and (j - w2) > 30:
            break

    # 最后检查收尾情况
    if begin:
        w2 = 220
        # w1 = w1 - 10 if w2 - w1 < 20 else w1
        b_copy = source[:, w1:w2]
        lps_img.append(b_copy)
        # cv2.imshow(str(uuid.uuid4()), b_copy)
        # cv2.waitKey(0)
    return lps_img


def Exp_images(img2, s):
    # 灰度
    gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("gray", gray)
    # cv2.waitKey(0)
    # 二值化
    ret, thresh = cv2.threshold(gray, s, 255, cv2.THRESH_BINARY)
    # ret, thresh = cv2.threshold(gray, s, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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


def Get_Lps(image, source):
    # 统计各行各列白色像素个数（为了得到直方图横纵坐标）-
    ptx, pty = White_Statistic(image)

    lp_ls = Cut_Y(pty, len(pty), source)
    # 绘制直方图（横、纵）
    # Draw_Hist(ptx, pty)
    return lp_ls


def Get_Lp_Images(image):
    img2 = cv2.resize(image, (220, 70))
    # cv2.imshow("img2", img2)
    # cv2.waitKey(0)

    for x in range(1, 31)[::-1]:
        # for k in range(10):
        #     img3 = img2[:, (k * 2):220 - (2 * k)]
        #     cv2.imshow('a', img3)
        #     cv2.waitKey(0)
        thresh0 = Exp_images(img2, x * 10)
        lp_ls = Get_Lps(thresh0, img2)
        # print(len(lp_ls))
        if len(lp_ls) == 7:
            # break
            check = [x.shape[1:2][0] for x in lp_ls]
            if 0 not in check:
                return lp_ls


if __name__ == '__main__':
    # 读取文件
    img2 = cv2.imread("../static/exa_lps.png")
    lp_imgs = Get_Lp_Images(img2)
    cv2.imshow('ima', img2)
    [cv2.imshow(str(l), _) for l, _ in enumerate(lp_imgs)]
    cv2.waitKey(0)
