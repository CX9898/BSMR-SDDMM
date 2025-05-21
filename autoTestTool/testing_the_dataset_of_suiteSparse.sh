#!/bin/bash

# 设置变量
results_path="results_dataset_of_suiteSparse/"
dataset_path="${results_path}dataset/"

# 下载数据集并解压
python download_matrix_from_suiteSparse.py -num 10 -outdir ${dataset_path}

# 生成矩阵文件列表
matrix_list_file="${results_path}matrix_file_list.txt"
> "${matrix_list_file}"
find "${dataset_path}" -type f | sed "s|^${results_path}||" > "${matrix_list_file}"

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