# FINN: Flood-Informed Neural Networks

基于物理信息神经网络的城市内涝水深建模。

## 简介

城市内涝是一种常见且危害严重的自然灾害。FINN（Flood-Informed Neural Networks）将 U-Net、GAN 架构与水力学物理约束相结合，用于预测城市暴雨内涝场景下的水深空间分布。与传统数值模拟模型（如 MIKE21）相比，FINN 在保持较高预测质量的同时显著降低了计算成本，可作为代理模型部署于洪水防控工作中。

## 研究区域与数据

- **研究区域**：丹麦 Odense 市（3740×4273 像素，5m 分辨率）
- **地形特征**（6 个）：地形凹凸度、不透水性、汇水面积、淹没水深、流向角 cos/sin
- **降雨事件**：53 场（43 场高强度 CDS + 10 场中等强度 NAT），提取 9 个统计特征作为输入
- **标签数据**：MIKE21 二维水动力模拟结果（水深分布）

数据未包含在本仓库中。训练数据应放置在 `configs/test.yaml` 中 `data_dir` 指定的路径下，目录结构如下：

```
<data_dir>/
├── background/    # 地形 patch 数据（memmap 格式）
└── flood/         # 洪涝事件数据
```

## 模型架构

| 模型 | 说明 |
|------|------|
| **UNet** | 基线模型。编码器提取地形空间特征，降雨特征经线性层融合后注入瓶颈层，解码器输出水深预测 |
| **PI-UNet** | 在 UNet 基础上引入物理约束损失：水量守恒惩罚项 + 空间连续性惩罚项 |
| **UNet-GAN** | 将 UNet 作为生成器，配合判别器进行对抗训练，提升水深分布的空间真实性 |

## 项目结构

```
FINN/
├── train.py          # 训练入口（PyTorch Lightning）
├── train.sh          # 训练启动脚本
├── configs/
│   └── test.yaml     # 训练配置（数据路径、超参数等）
├── data/             # 数据加载模块
│   ├── dataset.py    # Dataset 类与数据加载逻辑
│   ├── feature.py    # 降雨特征提取
│   └── rainvec.py    # 降雨过程向量生成
├── models/
│   ├── UNet.py       # UNet 架构
│   ├── PINN.py       # 物理信息约束扩展
│   └── config/       # 模型超参数配置
├── utils/
│   ├── metrics.py    # 评价指标（R², MSE, MAE）
│   └── learn.py      # 学习率调度等工具
├── report/           # 研究报告
└── results/          # 训练输出（未纳入版本控制）
```

## 使用方法

```bash
# 修改 configs/test.yaml 中的 data_dir 指向数据目录
# 选择模型（UNet 或 PINN）运行训练
bash train.sh
```

也可以直接调用：

```bash
python train.py --config="./configs/test.yaml" --model="PINN" --version="test"
```

## 主要依赖

- Python 3
- PyTorch + PyTorch Lightning
- NumPy, scikit-learn
- OmegaConf

## 作者

张景皓、钱昱诚、刘怀川、刘滨瑞
