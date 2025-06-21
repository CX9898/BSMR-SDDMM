#!/bin/bash

# 设置变量
results_path="results_dataset_of_suiteSparse/"
k32_results_path="${results_path}k32/"
k128_results_path="${results_path}k128/"
dataset_path="./dataset_of_suiteSparse/"

# 创建结果目录
mkdir -p ${results_path}
mkdir -p ${k32_results_path}
mkdir -p ${k128_results_path}

# 生成矩阵文件列表
bash make_matrices_list.sh ${dataset_path}
matrix_list_file="${dataset_path}matrix_file_list.txt"

# 编译程序
bash build_program.sh
program_zcx="./build_zcx/sddmm-gpu"
program_cuSDDMM="./build_cuSDDMM/cuSDDMM-sddmm"
program_ASpT_32="./build_ASpT/ASpT-sddmm-32"
program_ASpT_128="./build_ASpT/ASpT-sddmm-32"
program_BSA="./build_BSA/BSA-spmm"
program_RoDe="./build_RoDe/RoDe-sddmm"

ALPHA=( 0.3 )
BETA=( 4)

## 运行测试程序
#for A in "${ALPHA[@]}"; do
#  for B in "${BETA[@]}"; do
#    bash test_script.sh -f ${matrix_list_file} -p ${program_zcx} -n "${results_path}zcx_32_a_${A}_b_${B}" -k 32 -a ${A} -b ${B}
#  done
#done
#
#for A in "${ALPHA[@]}"; do
#  for B in "${BETA[@]}"; do
#    bash test_script.sh -f ${matrix_list_file} -p ${program_zcx} -n "${results_path}zcx_128_a_${A}_b_${B}" -k 128 -a ${A} -b ${B}
#  done
#done

#bash test_script.sh -f ${matrix_list_file} -p ${program_cuSDDMM} -n "${results_path}cuSDDMM_32" -k 32
#bash test_script.sh -f ${matrix_list_file} -p ${program_cuSDDMM} -n "${results_path}cuSDDMM_128" -k 128
#bash test_script.sh -f ${matrix_list_file} -p ${program_ASpT_32} -n "${results_path}ASpT_32" -k 32
#bash test_script.sh -f ${matrix_list_file} -p ${program_ASpT_128} -n "${results_path}ASpT_128" -k 128
#bash test_script.sh -f ${matrix_list_file} -p ${program_RoDe} -n "${results_path}RoDe_32" -k 32
#bash test_script.sh -f ${matrix_list_file} -p ${program_RoDe} -n "${results_path}RoDe_128" -k 128
#bash test_script.sh -f ${matrix_list_file} -p ${program_BSA} -n "${results_path}BSA_32" -k 128

g++ analyze_results.cpp -o analyze_results

log_files_k32=$(find "${k32_results_path}" -type f -name "*.log")
log_files_k128=$(find "${k128_results_path}" -type f -name "*.log")

./analyze_results ${log_files_k32} > ${results_path}analysis_results_k32.log
./analyze_results ${log_files_k128} > ${results_path}analysis_results_k128.log


# ./analyze_results "${results_path}zcx_32.log" \
#                       "${results_path}cuSDDMM_32.log" \
#                       "${results_path}ASpT_32.log" \
#                       "${results_path}BSA_32.log" \
#                       > ${results_path}analysis_results_32.log
# echo "Results analysis completed: ${results_path}analysis_results_32.log"

# ./analyze_results "${results_path}zcx_128.log" \
#                       "${results_path}cuSDDMM_128.log" \
#                       "${results_path}ASpT_128.log" \
#                       "${results_path}BSA_128.log" \
#                       > ${results_path}analysis_results_128.log
# echo "Results analysis completed: ${results_path}analysis_results_128.log"

# # 结果可视化
# python3 plot_sddmm_line_chart.py -file ${results_path}analysis_results_32.log -outdir ${results_path}
# python3 plot_sddmm_line_chart.py -file ${results_path}analysis_results_128.log -outdir ${results_path}
# python3 plot_reordering_line_chart.py -file ${results_path}analysis_results_32.log -outdir ${results_path}