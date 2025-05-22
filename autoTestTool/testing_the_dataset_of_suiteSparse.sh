#!/bin/bash

# 设置变量
results_path="results_dataset_of_suiteSparse/"
dataset_path="${results_path}dataset/"

# 下载数据集并解压
python download_matrix_from_suiteSparse.py -num 100 -outdir ${dataset_path} -row_min 512 -row_max 33288 -col_min 512 -col_max 33288

# 生成矩阵文件列表
bash make_matrices_list.sh ${dataset_path}
matrix_list_file="${dataset_path}matrix_file_list.txt"

# 编译程序
bash build_program.sh
program_zcx="./build_zcx/sddmm-gpu"
program_HiPC18="./build_HiPC18/HiPC18-sddmm"

# 运行测试程序
bash testTool.sh -f ${matrix_list_file} -p ${program_zcx} -n "${results_path}zcx_results"
bash testTool.sh -f ${matrix_list_file} -p ${program_HiPC18} -n "${results_path}HiPC18_results"

## 分析结果
g++ autoAnalysisResults.cpp -o autoAnalysisResults
./autoAnalysisResults "${results_path}zcx_results.log" "${results_path}HiPC18_results.log" > ${results_path}analysisResults.log
echo "Results analysis completed: ${results_path}analysisResults.log"

# 结果可视化
python3 resultsVisualization.py -file ${results_path}analysisResults.log -outdir ${results_path}
echo "The result visualization is successful! The file is stored in: ${results_path}"
