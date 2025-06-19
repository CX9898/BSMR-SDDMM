#!/bin/bash

# 设置变量
results_path="results_dataset_of_1/"
dataset_path="./dataset_of_suiteSparse/"
matrix_list_file="${dataset_path}matrix_file_list.txt"

program_zcx="./build_zcx/sddmm-gpu"

ALPHA=( 0.1 0.3 0.5 0.7 0.9 )
BETA=( 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 )

# 运行测试程序
for A in "${ALPHA[@]}"; do
  for B in "${BETA[@]}"; do
    bash test_script.sh -f ${matrix_list_file} -p ${program_zcx} -n "${results_path}zcx_32_a_${A}_b_${B}" -k 32 -a ${A} -b ${B}
  done
done

for A in "${ALPHA[@]}"; do
  for B in "${BETA[@]}"; do
    bash test_script.sh -f ${matrix_list_file} -p ${program_zcx} -n "${results_path}zcx_128_a_${A}_b_${B}" -k 128 -a ${A} -b ${B}
  done
done
