#!/bin/bash

# 设置变量
results_path="results_dataset_of_suiteSparse/2025-6-22/"
k32_results_path="${results_path}k32/"
k128_results_path="${results_path}k128/"

# 创建结果目录
mkdir -p ${k32_results_path}
mkdir -p ${k128_results_path}

# 创建结果目录
mkdir -p ${results_path}

g++ analyze_results.cpp -o analyze_results

log_files_k32=$(find "${k32_results_path}" -type f -name "*.log")
log_files_k128=$(find "${k128_results_path}" -type f -name "*.log")

./analyze_results ${log_files_k32} > ${results_path}analysis_results_k32.log
#./analyze_results ${log_files_k128} > ${results_path}analysis_results_k128.log
