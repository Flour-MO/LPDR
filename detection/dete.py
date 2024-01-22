# -*- coding: utf-8 -*-
"""
File Name：     main
Description :
date：          2024/1/18 018
"""
import cv2
import numpy as np
from scipy import stats
from detection.lps import Get_Lp_Images


class Det:
    def __init__(self, path):
        self.img, self.img_Gas, self.img_B, self.img_G, self.img_R, self.img_gray, self.img_HSV = self.imgProcess(path)
        self.img_bin = self.preIdentification(self.img_gray, self.img_HSV, self.img_B, self.img_R)
        self.lp_img = self.fixPosition(self.img, self.img_bin)
        self.lps = Get_Lp_Images(self.lp_img)

    def point_limit(self, point):
        if point[0] < 0:
            point[0] = 0
        if point[1] < 0:
            point[1] = 0

    def fixPosition(self, img, img_bin):
        # 根据设定的阈值和图片直方图，找出波峰，用于分隔字符
        def find_waves(threshold, histogram):
            up_point = -1  # 上升点
            is_peak = False
            if histogram[0] > threshold:
                up_point = 0
                is_peak = True
            wave_peaks = []
            for i, x in enumerate(histogram):
                if is_peak and x < threshold:
                    if i - up_point > 2:
                        is_peak = False
                        wave_peaks.append((up_point, i))
                elif not is_peak and x >= threshold:
                    is_peak = True
                    up_point = i
            if is_peak and up_point != -1 and i - up_point > 4:
                wave_peaks.append((up_point, i))
            return wave_peaks

        def remove_plate_upanddown_border(card_img, card_gray):
            """
            这个函数将截取到的车牌照片转化为灰度图，然后去除车牌的上下无用的边缘部分，确定上下边框
            输入： card_img是从原始图片中分割出的车牌照片
            输出: 在高度上缩小后的字符二值图片
            """
            # plate_Arr = cv2.imread(card_img)
            plate_gray_Arr = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            ret, plate_binary_img = cv2.threshold(plate_gray_Arr, 0, 255,
                                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # thresh = cv2.bitwise_not(plate_binary_img)
            # kernel = np.ones((2, 2), np.uint8)
            # plate_binary_img = cv2.erode(thresh, kernel, iterations=4)
            # cv2.imshow('2',plate_binary_img)
            # cv2.waitKey(0)

            row_histogram = np.sum(plate_binary_img, axis=1)  # 数组的每一行求和
            row_min = np.min(row_histogram)
            row_average = np.sum(row_histogram) / plate_binary_img.shape[0]
            row_threshold = (row_min + row_average) / 2
            wave_peaks = find_waves(row_threshold, row_histogram)
            # 接下来挑选跨度最大的波峰
            wave_span = 0.0
            # print(wave_peaks)
            for wave_peak in wave_peaks:
                span = wave_peak[1] - wave_peak[0]
                if span > wave_span:
                    wave_span = span
                    selected_wave = wave_peak
            plate_binary_img = card_img[selected_wave[0]:selected_wave[1], :]
            plate_binary_geay_img = card_gray[selected_wave[0]:selected_wave[1], :]
            # cv2.imshow("plate_binary_img", plate_binary_img)

            return plate_binary_img, plate_binary_geay_img

        def calc_slope_point(rotated_bin):
            h, w = rotated_bin.shape[:2]
            l_line_x = []
            l_line_y = []
            r_line_x = []
            r_line_y = []
            for i in range(h):
                for j in range(w):
                    if rotated_bin[i, j] == 255:
                        l_line_x.append(i)
                        l_line_y.append(-j)
                        break
                for k in range(w)[::-1]:
                    if rotated_bin[i, k] == 255:
                        r_line_x.append(i)
                        r_line_y.append(-k)
                        break
            lx = np.array(l_line_x)
            ly = np.array(l_line_y)
            rx = np.array(r_line_x)
            ry = np.array(r_line_y)
            # print(l_line_x)
            # print(l_line_y)
            rslope, intercept, r_value, p_value, std_err = stats.linregress(rx, ry)
            lslope, intercept, r_value, p_value, std_err = stats.linregress(lx, ly)
            print(f"双边斜率:{lslope, rslope}")
            if lslope > 0:
                left_point = (int(lslope * h) / 4, 0)
                down_point = (0, h)
            else:
                left_point = (0, 0)
                down_point = (-int(lslope * h) / 4, h)
            if rslope > 0:
                right_point = (w - int(rslope * h) / 4, h)
                up_point = (w, 0)
            else:
                right_point = (w, h)
                up_point = (w + int(rslope * h) / 4, 0)

            return left_point, up_point, right_point, down_point

        # 检测所有外轮廓，只留矩形的四个顶点
        contours, _ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        oldimg = img
        car_contours = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            area_width, area_height = rect[1]
            if area_width < area_height:
                area_width, area_height = area_height, area_width
            wh_ratio = area_width / area_height
            # print(wh_ratio)
            # 要求矩形区域长宽比在2到5.5之间，2到5.5是车牌的长宽比，其余的矩形排除
            if (2 < wh_ratio < 5.5) and area_height > 50:
                # car_contours.append(rect)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # car_contours.append(box)

                angle = rect[-1]
                # print(rect)

                y = [_[0] for _ in box]
                x = [_[1] for _ in box]
                lp_img = img[min(x):max(x), min(y) - 5:max(y) + 5]
                lp_g_img = img_bin[min(x):max(x), min(y) - 5:max(y) + 5]
                # cv2.imshow('lp_img', lp_img)
                # cv2.waitKey(0)

                h, w = lp_img.shape[:2]
                center = (w // 2, h // 2)
                # if angle % 90 != 0:
                print(f"倾斜角度:{angle}")
                if 1 < angle < 90:
                    M = cv2.getRotationMatrix2D(center, angle, 1.0) if angle < 45 else cv2.getRotationMatrix2D(
                        center, -(90 - angle), 1.0)
                else:
                    angle = 0
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(lp_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                rotated_bin = cv2.warpAffine(
                    lp_g_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                rotated = cv2.resize(rotated, (230, 70))  # [:, 10:220]
                rotated_bin = cv2.resize(rotated_bin, (230, 70))

                remove, remove_gery = remove_plate_upanddown_border(rotated, rotated_bin)

                # cv2.imshow('rotated', rotated)
                # cv2.waitKey(0)

                remove_h, remove_w = remove.shape[:2]

                left_point, up_point, right_point, down_point = calc_slope_point(remove_gery)
                # 变换前的四个点
                srcArr = np.float32([list(left_point), list(up_point), list(right_point), list(down_point)])
                print('原始点位>>>', [list(left_point), list(up_point), list(right_point), list(down_point)])
                # 变换后的四个点
                dstArr = np.float32([[0, 0], [remove_w, 0], [remove_w, remove_h], [0, remove_h]])
                print('校正点位<<<', [[0, 0], [remove_w, 0], [remove_w, remove_h], [0, remove_h]])
                # 获取变换矩阵
                MM = cv2.getPerspectiveTransform(srcArr, dstArr)

                dst = cv2.warpPerspective(remove, MM, (remove_w, remove_h))[0:remove_h, 0:remove_w][:, 10:220]
                remove = remove
                # print(dst.shape)

                # cv2.imshow('dst', dst)
                # cv2.waitKey(0)
                # if angle == 0:
                #     return remove
                # else:
                return dst

    def imgProcess(self, path):
        if isinstance(path, str):
            img = cv2.imread(path)
        else:
            img = path
        # 等比例缩放
        size = 1188
        # 获取原始图像宽高。
        height, width = img.shape[0], img.shape[1]
        # 等比例缩放尺度。
        scale = height / size
        # 获得相应等比例的图像宽度。
        width_size = int(width / scale)
        # resize
        img = cv2.resize(img, (width_size, size))
        # 高斯模糊
        img_Gas = cv2.GaussianBlur(img, (5, 5), 0)
        # RGB通道分离
        img_B = cv2.split(img_Gas)[0]
        img_G = cv2.split(img_Gas)[1]
        img_R = cv2.split(img_Gas)[2]
        # 读取灰度图和HSV空间图
        img_gray = cv2.cvtColor(img_Gas, cv2.COLOR_BGR2GRAY)
        img_HSV = cv2.cvtColor(img_Gas, cv2.COLOR_BGR2HSV)
        return img, img_Gas, img_B, img_G, img_R, img_gray, img_HSV

    def preIdentification(self, img_gray, img_HSV, img_B, img_R):
        h, w = self.img.shape[:2]
        for i in range(h):
            for j in range(w):
                # 普通蓝色车牌，同时排除透明反光物质的干扰
                if ((img_HSV[:, :, 0][i, j] - 115) ** 2 < 15 ** 2) and (img_B[i, j] > 70) and (img_R[i, j] < 40):
                    img_gray[i, j] = 255
                else:
                    img_gray[i, j] = 0
        # cv2.imshow('a', img_gray)
        # cv2.waitKey(0)
        # 定义核
        kernel_small = np.ones((3, 3))
        kernel_big = np.ones((7, 7))

        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)  # 高斯平滑
        img_di = cv2.dilate(img_gray, kernel_small, iterations=5)  # 腐蚀5次
        img_close = cv2.morphologyEx(img_di, cv2.MORPH_CLOSE, kernel_big)  # 闭操作
        img_close = cv2.GaussianBlur(img_close, (5, 5), 0)  # 高斯平滑

        _, img_bin = cv2.threshold(img_close, 100, 255, cv2.THRESH_BINARY)  # 二值化

        return img_bin


# cv2.waitKey(0)


if __name__ == '__main__':
    det = Det(r'../static/exa_img/3.jpg')
    # print(det.lp_img)
    # cv2.imshow('lp_img', det.lp_img)
    [cv2.imshow(str(k), _) for k, _ in enumerate(det.lps)]
    cv2.waitKey(0)
    from recognize.test_code import infer

    print(infer(img=det.lps[0], model_path=r'../recognize/model/LPDR_ZH.pkl'))
    print(infer(img=det.lps[1]))
    print(infer(img=det.lps[2]))
    print(infer(img=det.lps[3]))
    print(infer(img=det.lps[4]))
    print(infer(img=det.lps[5]))
    print(infer(img=det.lps[6]))
