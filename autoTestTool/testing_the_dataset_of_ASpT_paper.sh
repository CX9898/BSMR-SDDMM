#!/bin/bash

# 设置变量
results_path="results_dataset_of_ASpT_paper/"
dataset_path="../../ASpT-mirror/data/"

mkdir -p ${results_path}

# 创建矩阵列表文件
bash make_matrices_list.sh ${dataset_path}
matrix_list_file="${dataset_path}matrix_file_list.txt"

# 编译程序
bash build_program.sh
program_zcx="./build_zcx/sddmm-gpu"
program_isratnisa="./build_isratnisa/isratnisa-sddmm"

# 运行测试程序
bash testTool.sh -f ${matrix_list_file} -p ${program_zcx} -n "${results_path}zcx_results"
bash testTool.sh -f ${matrix_list_file} -p ${program_isratnisa} -n "${results_path}isratnisa_results"

# 分析结果
g++ autoAnalysisResults.cpp -o autoAnalysisResults
./autoAnalysisResults "${results_path}zcx_results.log" "${results_path}isratnisa_results.log" > ${results_path}analysisResults.log
echo "Results analysis completed: ${results_path}analysisResults.log"

# 结果可视化
python3 resultsVisualization.py -file ${results_path}analysisResults.log -outdir ${results_path}