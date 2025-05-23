#!/bin/bash

wget -nc https://storage.googleapis.com/sgk-sc2020/dlmc.tar.gz

tar -xvf dlmc.tar.gz

results_path="dataset_of_Sogang_University_results/"
dataset_path="${results_path}dataset/"
rn50_results_path="${dataset_path}rn50_results/"
transformer_results_path="${dataset_path}transformer_results/"

mkdir -p ${results_path}
mkdir -p ${rn50_results_path}
mkdir -p ${transformer_results_path}

mv -n dlmc ${results_path}
mv -n ${results_path}2048_512_dlmc_data.txt dataset/dlmc/

# 创建矩阵列表文件
bash make_matrices_list.sh ${rn50_results_path}
matrix_list_file="${dataset_path}matrix_file_list.txt"

# 编译程序
bash build_program.sh
program_zcx="./build_zcx/sddmm-gpu"
program_HiPC18="./build_HiPC18/HiPC18-sddmm"
program_RoDe="./build_RoDe/RoDe-sddmm"
program_ASpT_32="./build_ASpT/ASpT-sddmm-32"
program_ASpT_128="./build_ASpT/ASpT-sddmm-32"

# 测试 rn50 dataset
bash testTool.sh -f ${results_path}dlmc/rn50_matrices.txt -p ./build_zcx/sddmm-gpu -n ${rn50_results_path}zcx_results_rn50
bash testTool.sh -f ${results_path}dlmc/transformer_matrices.txt -p ./build_zcx/sddmm-gpu -n ${transformer_results_path}zcx_results_transformer

# 分析结果
g++ autoAnalysisResults.cpp -o autoAnalysisResults
./autoAnalysisResults "${results_path}zcx_results_32.log" "${results_path}zcx_results_128.log" \
                      "${results_path}HiPC18_results_32.log" "${results_path}HiPC18_results_128.log" \
                      "${results_path}RoDe_results_32.log" "${results_path}RoDe_results_128.log" \
                      "${results_path}ASpT_results_32.log" "${results_path}ASpT_results_128.log" \
                      > ${results_path}analysisResults.log
echo "Results analysis completed: ${results_path}analysisResults.log"

# 结果可视化
python3 resultsVisualizationBarChart.py -file ${results_path}analysisResults.log -outdir ${results_path}
echo "The bar chart was generated successfully! The file is stored in: ${results_path}"

python3 resultsVisualizationLineChart.py -file ${results_path}analysisResults.log -outdir ${results_path}
echo "The line chart was generated successfully! The file is stored in: ${results_path}"
