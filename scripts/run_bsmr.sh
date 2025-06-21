#!/bin/bash

# 设置变量
results_path="results_dataset_of_suiteSparse/"
k32_results_path="${results_path}k32/"
k128_results_path="${results_path}k128/"

# 创建结果目录
mkdir -p ${k32_results_path}
mkdir -p ${k128_results_path}

dataset_path="./dataset_of_suiteSparse/dataset/"
matrix_list_file="${dataset_path}matrix_file_list.txt"

program_zcx="./build_zcx/sddmm-gpu"

ALPHA=( 0.1 0.3 0.5 0.7 0.9 )
DELTA=( 0.1 0.3 0.5 0.7 0.9 )

# 运行测试程序
for A in "${ALPHA[@]}"; do
  for D in "${DELTA[@]}"; do
    bash test_script.sh -f ${matrix_list_file} -p ${program_zcx} -n "${k32_results_path}zcx_32_a_${A}_d_${D}" -k 32 -a ${A} -d ${D}
  done
done

for A in "${ALPHA[@]}"; do
  for D in "${DELTA[@]}"; do
    bash test_script.sh -f ${matrix_list_file} -p ${program_zcx} -n "${k128_results_path}zcx_128_a_${A}_d_${D}" -k 128 -a ${A} -d ${D}
  done
done
