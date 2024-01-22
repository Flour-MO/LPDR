# -*- coding: utf-8 -*-
"""
File Name：     main
Description :
date：          2024/1/18 018
"""
import os
import time

import cv2
import numpy as np
import gradio as gr
from PIL import Image
from detection.dete import Det
from recognize.test_code import infer


def cv2array(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def greet(image):
    # 类型转换
    array = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    start_time = time.time()
    print('>>>检测开始>>>')
    det = Det(array)
    end_time = time.time()
    timec = end_time - start_time
    img = np.ones((200, 800, 3), dtype=np.uint8)
    img *= 255  # white background
    ocr_text = ""
    if det.lps:
        for x, _ in enumerate(det.lps):
            # cv2.imshow(str(x), _)
            if x == 0:
                model_path = r'recognize\model\LPDR_ZH.pkl'
            else:
                model_path = r'recognize\model\LPDR_0_Z.pkl'
            ocr_text += infer(det.lps[x], model_path=model_path)
            height = int(_.shape[0])
            width = int(_.shape[1])
            for i in range(height):
                for j in range(width):
                    img[i + 40, j + 40 + (65 * x)] = _[i, j]
        print(f"车牌号:{ocr_text}")
        print(f"耗时:{timec}")
        print("<<<检测结束<<<")
        # cv2.waitKey(0)
        return cv2array(det.lp_img), cv2array(img), ocr_text
    else:
        return cv2array(det.lp_img), None, None


demo = gr.Interface(
    fn=greet,
    # 自定义输入框
    # 具体设置方法查看官方文档
    inputs=gr.Image(sources=["upload", "clipboard"], label="输入正面清晰的车牌", height="60vh"),
    outputs=[gr.Image(sources=[], label="检测", height="20vh"),
             gr.Image(sources=[], label="分割", height="20vh"),
             gr.Label(label="识别", )
             ],
    examples=[[fr'static\exa_img\{_}'] for _ in os.listdir(r'static\exa_img')]
)
demo.launch()
