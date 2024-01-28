# LPDR

***

### 目录说明：

```
├─detection(检测部分)
│     ├─dete.py(车牌轮廓检测)
│     └─lps.py(车牌字符分割)
├─recognize(识别部分)
│  ├─database(数据集目录)
│  │  ├─0_Z(包含0-Z图片数据若干)
│  │  └─ZH(包含31省份图片数据若干)
│  ├─model(模型存储目录)
│  │      ├─LPDR_0_Z.pkl(英文模型目录)
│  │      └─LPDR_ZH.pkl(中文模型目录)
│  ├─test_code.py(测试模型代码)
│  └─train_code.py(训练模型代码)
├─static(静态文件目录)
│  └─exa_img(样例图片)
├─main.py(GUI界面)
├─REAMD.md(说明文件)
```


#### 在线体验地址
https://replit.com/@WangHa/LPDR
