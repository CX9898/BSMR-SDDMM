#!/bin/bash

# 设置变量
results_path="./results_suiteSparse_dataset/"

k32_results_path="${results_path}k32/"
k64_results_path="${results_path}k64/"
k128_results_path="${results_path}k128/"
k256_results_path="${results_path}k256/"

BSMR_results_path="${results_path}BSMR_results/"

cp ${BSMR_results_path}BSMR_k_32* ${k32_results_path}
cp ${BSMR_results_path}BSMR_k_64* ${k64_results_path}
cp ${BSMR_results_path}BSMR_k_128* ${k128_results_path}
cp ${BSMR_results_path}BSMR_k_256* ${k256_results_path}

g++ analyze_results.cpp -o analyze_results

log_files_k32=$(find "${k32_results_path}" -type f -name "*.log")
log_files_k64=$(find "${k64_results_path}" -type f -name "*.log")
log_files_k128=$(find "${k128_results_path}" -type f -name "*.log")
log_files_k256=$(find "${k256_results_path}" -type f -name "*.log")

./analyze_results ${log_files_k32} > ${results_path}analysis_results_k32.log
echo "Analysis for k=32 results saved to ${results_path}analysis_results_k32.log"

./analyze_results ${log_files_k64} > ${results_path}analysis_results_k64.log
echo "Analysis for k=64 results saved to ${results_path}analysis_results_k64.log"

./analyze_results ${log_files_k128} > ${results_path}analysis_results_k128.log
echo "Analysis for k=128 results saved to ${results_path}analysis_results_k128.log"

./analyze_results ${log_files_k256} > ${results_path}analysis_results_k256.log
echo "Analysis for k=256 results saved to ${results_path}analysis_results_k256.log"

# plot
python plot_sddmm.py --k32 ${k32_results_path}results_32.csv --outdir ${results_path} --k64 ${k64_results_path}results_64.csv --k128 ${k128_results_path}results_128.csv --k128 ${k128_results_path}results_128.csv --k256 ${k256_results_path}results_256.csv

python plot_hybrid.py --k32 ${k32_results_path}results_hybrid_32.csv --k128 ${k128_results_path}results_hybrid_128.csv --outdir ${results_path}

# 清理临时文件
rm ${k32_results_path}BSMR_k_32*
rm ${k64_results_path}BSMR_k_64*
rm ${k128_results_path}BSMR_k_128*
rm ${k256_results_path}BSMR_k_256*
