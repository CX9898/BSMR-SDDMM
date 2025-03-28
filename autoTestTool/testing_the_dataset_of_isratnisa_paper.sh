#!/bin/bash

results_path="dataset_of_isratnisa_paper_results/"
dataset_path="${results_path}dataset/"

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

#mkdir -p ${results_path}
#download_decompressing_move "https://snap.stanford.edu/data/cit-HepPh.txt.gz"
#download_decompressing_move "https://snap.stanford.edu/data/bigdata/communities/com-amazon.ungraph.txt.gz"
#download_decompressing_move "https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz"
#download_decompressing_move "https://snap.stanford.edu/data/email-Enron.txt.gz"
#download_decompressing_move "https://snap.stanford.edu/data/web-BerkStan.txt.gz"
#download_decompressing_move "https://snap.stanford.edu/data/web-Google.txt.gz"
#download_decompressing_move "https://snap.stanford.edu/data/web-NotreDame.txt.gz"
#download_decompressing_move "https://snap.stanford.edu/data/loc-gowalla_edges.txt.gz"

#wget "https://graphchallenge.s3.amazonaws.com/snap/facebook_combined/facebook_combined_adj.mmio"
#mv facebook_combined_adj.mmio ${dataset_path}

# 在没有头部信息的文件中添加头部信息
#sed -i '1i # Nodes: 1005 Edges: 25571' ${dataset_path}email-Eu-core.txt
#sed -i '1i # Nodes: 196591 Edges: 1900654' ${dataset_path}loc-gowalla_edges.txt

# 生成矩阵文件列表
matrix_list_file="${results_path}matrix_file_list.txt"
> "${matrix_list_file}"
find ${dataset_path} -type f > ${matrix_list_file}

bash build_program.sh
program_zcx="./build_zcx/sddmm-gpu"
program_isratnisa="./build_isratnisa/isratnisa-sddmm"

bash testTool.sh -f ${matrix_list_file} -p ${program_zcx} -n "${results_path}zcx_results"
bash testTool.sh -f ${matrix_list_file} -p ${program_isratnisa} -n "${results_path}isratnisa_results"

# 分析结果
g++ autoAnalysisResults.cpp -o autoAnalysisResults
./autoAnalysisResults "${results_path}zcx_results.log" "${results_path}isratnisa_results.log" > ${results_path}analysisResults.log
echo "Results analysis completed: ${results_path}analysisResults.log"