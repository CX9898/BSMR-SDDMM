#!/bin/bash

# 设置变量
results_path="results_dataset_of_cuSDDMM_paper/"
dataset_path="${results_path}dataset/"

# 函数定义
# 下载数据集并解压, 并移动到指定目录
download_decompressing_move(){
  local url=$1
  local compressed_file=$(basename "$url")
  wget -O ${compressed_file} ${url}
  gunzip -f $compressed_file
  local file=$(basename "${compressed_file}" .gz)
  mkdir -p ${dataset_path}
  mv ${file} ${dataset_path}
}

#############################################################
# main

# 创建结果目录
mkdir -p ${results_path}

# 数据集准备
mkdir -p ${results_path}
download_decompressing_move "https://snap.stanford.edu/data/cit-HepPh.txt.gz"
download_decompressing_move "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz"
download_decompressing_move "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz"
download_decompressing_move "https://snap.stanford.edu/data/email-Enron.txt.gz"
download_decompressing_move "https://snap.stanford.edu/data/web-BerkStan.txt.gz"
download_decompressing_move "https://snap.stanford.edu/data/web-Google.txt.gz"
download_decompressing_move "https://snap.stanford.edu/data/web-NotreDame.txt.gz"
download_decompressing_move "https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz"
wget "https://graphchallenge.s3.amazonaws.com/snap/facebook_combined/facebook_combined_adj.mmio"
mv facebook_combined_adj.mmio ${dataset_path}

# 在没有头部信息的文件中添加头部信息
sed -i '1i # Nodes: 1005 Edges: 25571' ${dataset_path}email-Eu-core.txt
sed -i '1i # Nodes: 196591 Edges: 1900654' ${dataset_path}loc-gowalla_edges.txt

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
g++ autoAnalysisResults.cpp -o autoAnalysisResults
./autoAnalysisResults "${results_path}zcx_results_32.log" "${results_path}zcx_results_128.log" \
                      "${results_path}cuSDDMM_results_32.log" "${results_path}cuSDDMM_results_128.log" \
                      "${results_path}RoDe_results_32.log" "${results_path}RoDe_results_128.log" \
                      "${results_path}ASpT_results_32.log" "${results_path}ASpT_results_128.log" \
                      > ${results_path}analysisResults.log
echo "Results analysis completed: ${results_path}analysisResults.log"

# 结果可视化
python3 plot_sddmm_line_chart.py -file ${results_path}analysisResults.log -outdir ${results_path}