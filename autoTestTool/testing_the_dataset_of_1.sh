#!/bin/bash

# 设置变量
results_path="results_dataset_of_1/"
dataset_path="${results_path}dataset/"

# 创建结果目录
mkdir -p ${results_path}
mkdir -p ${dataset_path}

# 函数定义
# 下载数据集并解压, 并移动到指定目录
download_decompressing_move(){
  local url=$1
  local compressed_file=$(basename "$url")
  wget -O ${compressed_file} ${url}
  tar -xvzf $compressed_file -C ${dataset_path}
  rm ${compressed_file}
}

# 下载数据集并解压
download_decompressing_move "https://suitesparse-collection-website.herokuapp.com/MM/ND/nd3k.tar.gz"
download_decompressing_move "https://suitesparse-collection-website.herokuapp.com/MM/Gupta/gupta3.tar.gz"
download_decompressing_move "https://suitesparse-collection-website.herokuapp.com/MM/Chen/pkustk08.tar.gz"
download_decompressing_move "https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/ramage02.tar.gz"
download_decompressing_move "https://suitesparse-collection-website.herokuapp.com/MM/TKK/smt.tar.gz"

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
bash testTool.sh -f ${matrix_list_file} -p ${program_zcx} -n "${results_path}zcx_results_32" -k 32
bash testTool.sh -f ${matrix_list_file} -p ${program_zcx} -n "${results_path}zcx_results_128" -k 128
bash testTool.sh -f ${matrix_list_file} -p ${program_cuSDDMM} -n "${results_path}cuSDDMM_results_32" -k 32
bash testTool.sh -f ${matrix_list_file} -p ${program_cuSDDMM} -n "${results_path}cuSDDMM_results_128" -k 128
bash testTool.sh -f ${matrix_list_file} -p ${program_RoDe} -n "${results_path}RoDe_results_32" -k 32
bash testTool.sh -f ${matrix_list_file} -p ${program_RoDe} -n "${results_path}RoDe_results_128" -k 128
bash testTool.sh -f ${matrix_list_file} -p ${program_ASpT_32} -n "${results_path}ASpT_results_32" -k 32
bash testTool.sh -f ${matrix_list_file} -p ${program_ASpT_128} -n "${results_path}ASpT_results_128" -k 128

# 分析结果
g++ autoAnalysisResults.cpp -o autoAnalysisResults
./autoAnalysisResults "${results_path}zcx_results_32.log" "${results_path}zcx_results_128.log" \
                      "${results_path}cuSDDMM_results_32.log" "${results_path}cuSDDMM_results_128.log" \
                      "${results_path}RoDe_results_32.log" "${results_path}RoDe_results_128.log" \
                      "${results_path}ASpT_results_32.log" \
                      > ${results_path}analysisResults.log
echo "Results analysis completed: ${results_path}analysisResults.log"

# 结果可视化
python3 resultsVisualizationBarChart.py -file ${results_path}analysisResults.log -outdir ${results_path}
echo "The bar chart was generated successfully! The file is stored in: ${results_path}"

python3 resultsVisualizationLineChart.py -file ${results_path}analysisResults.log -outdir ${results_path}
echo "The line chart was generated successfully! The file is stored in: ${results_path}"