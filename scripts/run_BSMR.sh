#!/bin/bash

# 设置变量
results_path="./results_suiteSparse_dataset/"
dataset_path="./suiteSparse_dataset/"

BSMR_results_path="${results_path}BSMR_results/"

# 创建结果目录
mkdir -p ${BSMR_results_path}

bash make_matrices_list.sh ${dataset_path}
matrix_list_file="${dataset_path}matrix_file_list_mtx.txt"

# run BSMR
program_BSMR="./build_BSMR/BSMR-sddmm"
bash test_script.sh -f ${matrix_list_file} -p ${program_BSMR} -n "${BSMR_results_path}results_log" -l ${BSMR_results_path}
