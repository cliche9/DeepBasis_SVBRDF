#!/bin/bash

# 设置Python解释器路径
PYTHON="/home/sunqi/data/miniconda3/envs/deepbasis/bin/python"

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=1

# 运行Python脚本，并传入参数
$PYTHON "/home/sunqi/data/code/DeepBasis_SVBRDF/MatSynth_1k_train.py" \
    --name "DeepBasisMatSynth_1k_Train" \
    --save_root "/home/sunqi/data/code/DeepBasis_SVBRDF/output/MatSynth_1k/train" \
    --dataset_root "/home/zhupengfei/data/datasets/MatSynth_1k/train/" \
    --test_data_root "/home/sunqi/data/code/DeepBasis_SVBRDF/source/MatSynth_1k/train" \
    --loadpath_network_g "/home/sunqi/data/code/DeepBasis_SVBRDF/pretrain/net_g_2.414.pth" \
    --loadpath_network_l "/home/sunqi/data/code/DeepBasis_SVBRDF/pretrain/net_l_2.414.pth" \
    --fovZ "2.414"
