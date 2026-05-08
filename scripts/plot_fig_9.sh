#!/bin/bash

# 设置变量
results_path="./results_suiteSparse_dataset/"
k128_results_path="${results_path}k128/"

python3 numerical_error_analysis.py \
  --results-csv ${k128_results_path}results_128.csv
