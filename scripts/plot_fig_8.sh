#!/bin/bash

# 设置变量
results_path="./results_suiteSparse_dataset/"
deep_failure_dir="${results_path}deep_failure_analysis/"

# Step 1: 生成 per_case_diagnostics.csv
#   读取 ${results_path}k128/results_128.csv 和 ${results_path}BSMR_results/BSMR_k_128_*.log
#   输出到 ${deep_failure_dir}
python failure_case_deep_analysis.py \
  --k 128 \
  --out-dir ${deep_failure_dir}

# Step 2: 基于 per_case_diagnostics.csv 绘制 K=128 失败分析散点图
#   输出: ${deep_failure_dir}k128_failure_metric_scatter.{png,pdf}
python plot_failure_speed_scatter.py \
  --case-csv ${deep_failure_dir}per_case_diagnostics.csv \
  --out-dir ${deep_failure_dir}
