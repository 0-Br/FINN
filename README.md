# FINN: Flood-Informed Neural Networks

基于物理信息神经网络的城市内涝水深预测 / Physics-Informed Neural Networks for Urban Flood Water Depth Prediction

## 概述 / Overview

城市内涝是一种常见且危害严重的自然灾害。FINN（Flood-Informed Neural Networks）将 U-Net、GAN 架构与水力学物理约束相结合，用于预测城市暴雨内涝场景下的水深空间分布。与传统数值模拟模型（如 MIKE21）相比，FINN 在保持较高预测质量的同时显著降低了计算成本，可作为代理模型部署于洪水防控工作中。

Urban flooding is a frequent and damaging natural hazard. FINN (Flood-Informed Neural Networks) combines U-Net and GAN architectures with hydraulic physics constraints to predict the spatial distribution of water depth under urban pluvial flooding scenarios. Compared to traditional numerical simulation models such as MIKE21, FINN significantly reduces computational cost while maintaining high prediction quality, serving as a surrogate model for flood risk management.

## 研究区域与数据 / Study Area & Data

- **研究区域 / Study area**：丹麦 Odense 市（3740×4273 像素，5m 分辨率）/ Odense, Denmark (3740×4273 pixels, 5m resolution)
- **地形特征 / Topographic features**（6 个 / 6 channels）：地形凹凸度、不透水性、汇水面积、淹没水深、流向角 cos/sin / terrain convexity, imperviousness, catchment area, flood depth, flow direction cos/sin
- **降雨事件 / Rainfall events**：53 场（43 场高强度 CDS + 10 场中等强度 NAT），提取 9 个统计特征作为输入 / 53 events (43 high-intensity CDS + 10 moderate NAT), with 9 statistical features extracted as input
- **标签数据 / Ground truth**：MIKE21 二维水动力模拟结果（水深分布）/ MIKE21 2D hydrodynamic simulation results (water depth distribution)

数据未包含在本仓库中。训练数据应放置在 `configs/test.yaml` 中 `data_dir` 指定的路径下，目录结构如下：

Data is not included in this repository. Training data should be placed at the path specified by `data_dir` in `configs/test.yaml`, with the following directory structure:

```
<data_dir>/
├── background/    # 地形 patch 数据（memmap 格式）/ topographic patches (memmap format)
└── flood/         # 洪涝事件数据 / flood event data
```

## 模型架构 / Model Architecture

| 模型 / Model | 说明 / Description |
|---|---|
| **UNet** | 基线模型。编码器提取地形空间特征，降雨特征经线性层融合后注入瓶颈层，解码器输出水深预测 / Baseline. Encoder extracts spatial topographic features; rainfall features are fused via linear layers into the bottleneck; decoder outputs water depth predictions |
| **PI-UNet** | 在 UNet 基础上引入物理约束损失：水量守恒惩罚项 + 空间连续性惩罚项 / UNet with physics-informed loss: mass conservation penalty + spatial continuity penalty |
| **UNet-GAN** | 将 UNet 作为生成器，配合判别器进行对抗训练，提升水深分布的空间真实性 / UNet as generator with adversarial training via a discriminator to improve spatial realism of predicted depth distributions |

## 项目结构 / Project Structure

```
FINN/
├── train.py          # 训练入口 / training entry (PyTorch Lightning)
├── train.sh          # 训练启动脚本 / training launch script
├── configs/
│   └── test.yaml     # 训练配置 / training config (data paths, hyperparameters)
├── data/             # 数据加载模块 / data loading module
│   ├── dataset.py    # Dataset 类与数据加载 / Dataset class & data loading
│   ├── feature.py    # 降雨特征提取 / rainfall feature extraction
│   └── rainvec.py    # 降雨过程向量生成 / rainfall process vector generation
├── models/
│   ├── UNet.py       # UNet 架构 / UNet architecture
│   ├── PINN.py       # 物理信息约束扩展 / physics-informed extensions
│   └── config/       # 模型超参数配置 / model hyperparameter configs
├── utils/
│   ├── metrics.py    # 评价指标 / evaluation metrics (R², MSE, MAE)
│   └── learn.py      # 学习率调度等 / learning rate scheduling, etc.
├── report/           # 研究报告 / research report (LaTeX)
└── results/          # 训练输出 / training outputs (not version-controlled)
```

## 使用方法 / Usage

修改 `configs/test.yaml` 中的 `data_dir` 指向数据目录，然后运行训练：

Edit `data_dir` in `configs/test.yaml` to point to the data directory, then start training:

```bash
# 通过启动脚本运行 / via launch script
bash train.sh

# 或直接调用，指定模型与版本 / or call directly with model and version
python train.py --config="./configs/test.yaml" --model="PINN" --version="test"
```

## 依赖 / Dependencies

- Python 3
- PyTorch + PyTorch Lightning
- NumPy, scikit-learn
- OmegaConf

## 作者 / Authors

张景皓、钱昱诚、刘怀川、刘滨瑞 (Jinghao Zhang, Yucheng Qian, Huaichuan Liu, Binrui Liu)，清华大学 / Tsinghua University
