#!/bin/bash

#python download_matrix_from_suiteSparse.py -n 1000

bash build_program.sh

matrix_list_file="../dataset/ssgetpy/matrix_file_list.txt"

bash testTool.sh -f ${matrix_list_file} -p ./build_zcx/sddmm-gpu -n zcx_results

bash testTool.sh -f ${matrix_list_file} -p ./build_isratnisa/isratnisa-sddmm -n isratnisa_results

# 分析结果
g++ autoAnalysisResults.cpp -o autoAnalysisResults
./autoAnalysisResults zcx_results.log  isratnisa_results.log >> analysisResults.log
