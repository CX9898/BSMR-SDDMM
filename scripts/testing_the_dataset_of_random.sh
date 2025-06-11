#!/bin/bash

# 设置变量
results_path="results_dataset_of_random/"
dataset_path="../dataset/test/"

# 创建结果目录
mkdir -p ${results_path}

# 生成矩阵文件列表
bash make_matrices_list.sh ${dataset_path}
matrix_list_file="${dataset_path}matrix_file_list.txt"

# 编译程序
bash build_program.sh
program_zcx="./build_zcx/sddmm-gpu"
program_cuSDDMM="./build_cuSDDMM/cuSDDMM-sddmm"
program_RoDe="./build_RoDe/RoDe-sddmm"
program_ASpT_32="./build_ASpT/ASpT-sddmm-32"
program_ASpT_128="./build_ASpT/ASpT-sddmm-32"

# 运行测试程序
bash test_script.sh -f ${matrix_list_file} -p ${program_zcx} -n "${results_path}zcx_results_32" -k 32
bash test_script.sh -f ${matrix_list_file} -p ${program_zcx} -n "${results_path}zcx_results_128" -k 128
bash test_script.sh -f ${matrix_list_file} -p ${program_cuSDDMM} -n "${results_path}cuSDDMM_results_32" -k 32
bash test_script.sh -f ${matrix_list_file} -p ${program_cuSDDMM} -n "${results_path}cuSDDMM_results_128" -k 128
bash test_script.sh -f ${matrix_list_file} -p ${program_RoDe} -n "${results_path}RoDe_results_32" -k 32
bash test_script.sh -f ${matrix_list_file} -p ${program_RoDe} -n "${results_path}RoDe_results_128" -k 128
bash test_script.sh -f ${matrix_list_file} -p ${program_ASpT_32} -n "${results_path}ASpT_results_32" -k 32
bash test_script.sh -f ${matrix_list_file} -p ${program_ASpT_128} -n "${results_path}ASpT_results_128" -k 128

# 分析结果
g++ analyze_results.cpp -o analyze_results
./analyze_results "${results_path}zcx_results_32.log" "${results_path}zcx_results_128.log" \
                      "${results_path}cuSDDMM_results_32.log" "${results_path}cuSDDMM_results_128.log" \
                      "${results_path}RoDe_results_32.log" "${results_path}RoDe_results_128.log" \
                      "${results_path}ASpT_results_32.log" "${results_path}ASpT_results_128.log" \
                      > ${results_path}analysisResults.log
echo "Results analysis completed: ${results_path}analysisResults.log"

# 结果可视化
python3 plot_sddmm_line_chart.py -file ${results_path}analysisResults.log -outdir ${results_path}