# DASDU-Net: Dual-Attention Spectral Diffusion Unfolding Network

[![Stars](https://img.shields.io/github/stars/XWangBin/DASDU-Net?style=flat-square)](https://github.com/XWangBin/DASDU-Net/stargazers)
[![PyTorch](https://img.shields.io/badge/PyTorch-v1.8+-ee4c2c?logo=pytorch&style=flat-square)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)

这是论文 **"DASDU-Net for Spectral Reconstruction from Multispectral Images"** 的官方 PyTorch 实现代码。

---

## 📖 简介 (Introduction)
DASDU-Net 提出了一种结合**双注意力机制**与**深度展开网络**的光谱重建框架。它能够有效地从多光谱图像（MSI）或 RGB 图像中恢复出高精度的光谱信息（HSI）。该模型在多个遥感及可见光数据集上表现出了卓越的性能。



---

## 🎨 流程图 (Flowchart)
*目前正在准备发布中 (Waiting for release...)*

---

## 📊 实验结果展示 (Result Presentation)

### 1. 模拟数据集结果
在 **CAVE** 和 **NTIRE 2022** 数据集上的重建性能对比：

![Performance](https://github.com/XWangBin/DASDU-Net/blob/main/IMGs/result1.png?raw=true)
*Fig 1. (a) CAVE 数据集与 (b) NTIRE 2022 数据集上的仿真结果对比。*

在 **Chikusei** 和 **雄安 (Xiong'an)** 遥感数据集上的结果对比：

![Performance](https://github.com/XWangBin/DASDU-Net/blob/main/IMGs/result2.png?raw=true)
*Fig 2. (a) Chikusei 数据集与 (b) 雄安数据集上的仿真结果对比。*

### 2. 真实场景应用
模型在真实遥感图像及可见光图像上的泛化能力：

| 真实遥感图像重建 | 真实可见光图像重建 |
| :---: | :---: |
| ![Real-RS](https://github.com/XWangBin/DASDU-Net/blob/main/IMGs/result4.png?raw=true) | ![Real-VL](https://github.com/XWangBin/DASDU-Net/blob/main/IMGs/result3.png?raw=true) |
| *Fig 3. 真实遥感图像实验结果* | *Fig 4. 真实可见光图像实验结果* |

---

## 📂 数据集下载 (Datasets)

| 数据集 | 描述 | 下载链接 |
| :--- | :--- | :--- |
| **CAVE** | 实验室环境下的可见光光谱数据 | [原始数据](https://www1.cs.columbia.edu/CAVE/databases/multispectral/) \| [预处理版 (百度网盘)](https://aistudio.baidu.com/aistudio/datasetdetail/147509) |
| **NTIRE 2022** | 自然场景光谱数据 | [从 MST++ 下载](https://github.com/caiyuanhao1998/MST-plus-plus) |
| **Chikusei** | 遥感影像数据集 (筑波) | [预处理版 (百度网盘)](https://aistudio.baidu.com/aistudio/datasetdetail/262154) |
| **Xiong'an** | 遥感影像数据集 (雄安) | [预处理版 (百度网盘)](https://aistudio.baidu.com/aistudio/datasetdetail/277497) |
| **Real-world** | 真实场景采集数据 | *即将发布...* |

---

### 不同数据集的推荐参数

针对可见光与高维遥感场景，请根据下表配置对应的光谱波段参数：

| 数据集 (Dataset) | 输入通道 (`in_c`) | 输出波段 (`out_c`) | 特征维度 (`n_feat`) | 场景说明 |
| :--- | :---: | :---: | :---: | :--- |
| **CAVE** | 3 | 31 | **31** | 实验室可见光光谱 |
| **NTIRE 2022** | 3 | 31 | **31** | 自然场景可见光光谱 |
| **雄安 (Xiong'an)** | 3 | 93 | **93** | 遥感高光谱 (93通道) |
| **筑波 (Chikusei)** | 3 | 128 | **128** | 遥感高光谱 (128通道) |

## 🛠️ 环境配置 (Environment)
```bash
# 创建并激活环境
conda create -n dasdu python=3.8
conda activate dasdu

# 安装依赖项
pip install torch torchvision einops tqdm matplotlib opencv-python
