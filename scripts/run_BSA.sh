#!/bin/bash

# 设置变量
results_path="./results_suiteSparse_dataset/"
dataset_path="./suiteSparse_dataset/"

BSA_results_path="${results_path}BSA_results/"

# 创建结果目录
mkdir -p ${BSA_results_path}

bash make_matrices_list.sh ${dataset_path}
matrix_list_file="${dataset_path}matrix_file_list_mtx.txt"

# run BSA
program_BSA="./build_BSA/BSA-spmm"
bash test_script.sh -f ${matrix_list_file} -p ${program_BSA} -n "${BSA_results_path}results_log" -l ${BSA_results_path}

