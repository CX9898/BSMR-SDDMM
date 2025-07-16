#!/bin/bash

# 删除旧环境
conda remove -n TCGNN --all -y

# 创建新环境
conda create -n TCGNN python=3.9 -y

# 加载 conda 环境脚本
source ~/miniconda3/etc/profile.d/conda.sh

# 激活环境
conda activate TCGNN

# 安装 PyTorch + CUDA
conda install pytorch=2.5.1 torchvision=0.20.1 torchaudio=2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y

# 可选：安装 ninja 加快编译速度
conda install ninja -y

# 安装 TCGNN 模块（推荐方式）
cd ../baselines/TCGNN_kernel
rm -rf build/
pip install .
