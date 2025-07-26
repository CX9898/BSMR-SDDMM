#!/bin/bash

# 设置变量
results_path="./results_suiteSparse_dataset/"

BSMR_results_path="${results_path}BSMR_results/"
BSA_results_path="${results_path}BSA_results/"

cp ${BSMR_results_path}BSMR_k_32* ${BSA_results_path}

log_files=$(find "${BSA_results_path}" -type f -name "*.log")

g++ analyze_results.cpp -o analyze_results

./analyze_results ${log_files} > ${results_path}analysis_results_reordering.log
echo "Analysis reordering results saved to ${results_path}analysis_results_reordering.log"

python plot_reordering.py --file ${results_path}analysis_results_reordering.log --outdir ${results_path}

rm ${BSA_results_path}BSMR_k_32*
