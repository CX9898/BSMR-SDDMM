#!/bin/bash

# 设置变量
results_path="./results_suiteSparse_dataset/"

k32_results_path="${results_path}k32/"

BSMR_results_path="${results_path}BSMR_results/"

cp ${BSMR_results_path}BSMR_k_32* ${k32_results_path}

g++ analyze_results.cpp -o analyze_results

log_files_k32=$(find "${k32_results_path}" -type f -name "*.log")

./analyze_results ${log_files_k32} > ${results_path}analysis_results_k32.log
echo "Analysis for k=32 results saved to ${results_path}analysis_results_k32.log"

# plot
python plot_reordering_overhead.py --file ${results_path}analysis_results_k32.log --outdir ${results_path}

# 清理临时文件
rm ${k32_results_path}BSMR_k_32*
