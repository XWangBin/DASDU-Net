# DASDU-Net: Deep Adaptive Spectral Degradation Unfolding Network for Spectral Reconstruction from Multispectral Images

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

## 📂 数据集下载 (Datasets)

| 数据集 | 描述 | 下载链接 |
| :--- | :--- | :--- |
| **CAVE** | 实验室环境下的可见光光谱数据 | [原始数据](https://www1.cs.columbia.edu/CAVE/databases/multispectral/) \| [预处理版 (百度网盘)](https://aistudio.baidu.com/aistudio/datasetdetail/147509) |
| **NTIRE 2022** | 自然场景光谱数据 | [从 MST++ 下载](https://github.com/caiyuanhao1998/MST-plus-plus) |
| **Chikusei** | 遥感影像数据集 | [预处理版 (百度网盘)](https://aistudio.baidu.com/aistudio/datasetdetail/262154) |
| **Xiong'an** | 遥感影像数据集 | [预处理版 (百度网盘)](https://aistudio.baidu.com/aistudio/datasetdetail/277497) |
| **Real-world** | 真实场景采集数据 | *即将发布...* |

---

### 不同数据集的推荐参数

针对可见光与高维遥感场景，请根据下表配置对应的光谱波段参数：

| 数据集 (Dataset) | 输入通道 (`in_c`) | 输出波段 (`out_c`) | 特征维度 (`n_feat`) | 场景说明 |
| :--- | :---: | :---: | :---: | :--- |
| **CAVE** | 3 | 31 | **31** | 实验室可见光光谱 |
| **NTIRE 2022** | 3 | 31 | **31** | 自然场景可见光光谱 |
| **Xiongan** | 4 | 93 | **93** | 遥感高光谱 (93通道) |
| **Chikusei** | 4 | 128 | **128** | 遥感高光谱 (128通道) |

## 🛠️ 环境配置 (Environment)
```bash
# 创建并激活环境
conda create -n dasdu python=3.8
conda activate dasdu

# 安装依赖项
pip install torch torchvision einops
