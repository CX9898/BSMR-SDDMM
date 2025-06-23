#!/bin/bash

# 设置变量
results_path="../../dlmc/results_rn50/"
dataset_path="../../dlmc/rn50/"

k32_results_path="${results_path}k32/"
k128_results_path="${results_path}k128/"

# 创建结果目录
mkdir -p ${k32_results_path}
mkdir -p ${k128_results_path}

bash make_matrices_list.sh ${dataset_path}
matrix_list_file="${dataset_path}matrix_file_list.txt"

program_bsma="./build_bsma/bsma-sddmm"
program_cuSDDMM="./build_cuSDDMM/cuSDDMM-sddmm"
program_ASpT_32="./build_ASpT/ASpT-sddmm-32"
program_ASpT_128="./build_ASpT/ASpT-sddmm-32"
program_BSA="./build_BSA/BSA-spmm"
program_RoDe="./build_RoDe/RoDe-sddmm"

bash test_script.sh -f ${matrix_list_file} -p ${program_cuSDDMM} -n "${k32_results_path}cuSDDMM_32" -k 32
bash test_script.sh -f ${matrix_list_file} -p ${program_ASpT_32} -n "${k32_results_path}ASpT_32" -k 32
bash test_script.sh -f ${matrix_list_file} -p ${program_RoDe} -n "${k32_results_path}RoDe_32" -k 32

bash test_script.sh -f ${matrix_list_file} -p ${program_cuSDDMM} -n "${k128_results_path}cuSDDMM_128" -k 128
bash test_script.sh -f ${matrix_list_file} -p ${program_ASpT_128} -n "${k128_results_path}ASpT_128" -k 128
bash test_script.sh -f ${matrix_list_file} -p ${program_RoDe} -n "${k128_results_path}RoDe_128" -k 128

ALPHA=( 0.1 0.3 0.5 0.7 0.9 )
DELTA=( 0.1 0.3 0.5 0.7 0.9 )

for A in "${ALPHA[@]}"; do
  for D in "${DELTA[@]}"; do
    bash test_script.sh -f ${matrix_list_file} -p ${program_BSA} -n "${k32_results_path}BSA_32_a_${A}_d_${D}" -k 32 -a ${A} -d ${D}
  done
done

for A in "${ALPHA[@]}"; do
  for D in "${DELTA[@]}"; do
    bash test_script.sh -f ${matrix_list_file} -p ${program_bsma} -n "${k32_results_path}bsma_32_a_${A}_d_${D}" -k 32 -a ${A} -d ${D}
  done
done

for A in "${ALPHA[@]}"; do
  for D in "${DELTA[@]}"; do
    bash test_script.sh -f ${matrix_list_file} -p ${program_bsma} -n "${k128_results_path}bsma_128_a_${A}_d_${D}" -k 128 -a ${A} -d ${D}
  done
done

