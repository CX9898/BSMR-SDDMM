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
matrix_list_file="${dataset_path}matrix_file_list_mtx.txt"

program_BSMR="./build_BSMR/BSMR-sddmm"
program_cuSDDMM="./build_cuSDDMM/cuSDDMM-sddmm"
program_ASpT_32="./build_ASpT/ASpT-sddmm-32"
program_ASpT_128="./build_ASpT/ASpT-sddmm-32"
program_BSA="./build_BSA/BSA-spmm"
program_RoDe="./build_RoDe/RoDe-sddmm"

bash test_script.sh -f ${matrix_list_file} -p ${program_cuSDDMM} -n "${k32_results_path}cuSDDMM_32" -k 32
bash test_script.sh -f ${matrix_list_file} -p ${program_ASpT_32} -n "${k32_results_path}ASpT_32" -k 32
bash test_script.sh -f ${matrix_list_file} -p ${program_RoDe} -n "${k32_results_path}RoDe_32" -k 32

ALPHA=( 0.1 0.3 0.5 0.7 0.9 )
DELTA=( 0.1 0.3 0.5 0.7 0.9 )

for A in "${ALPHA[@]}"; do
  for D in "${DELTA[@]}"; do
    bash test_script.sh -f ${matrix_list_file} -p ${program_BSA} -n "${k32_results_path}BSA_32_a_${A}_d_${D}" -k 32 -a ${A} -d ${D}
  done
done

for A in "${ALPHA[@]}"; do
  for D in "${DELTA[@]}"; do
    bash test_script.sh -f ${matrix_list_file} -p ${program_BSMR} -n "${k32_results_path}BSMR_32_a_${A}_d_${D}" -k 32 -a ${A} -d ${D}
  done
done

g++ analyze_results.cpp -o analyze_results

log_files_k32=$(find "${k32_results_path}" -type f -name "*.log")
log_files_k128=$(find "${k128_results_path}" -type f -name "*.log")

./analyze_results ${log_files_k32} > ${results_path}analysis_results_k32.log
echo "Analysis for k=32 results saved to ${results_path}analysis_results_k32.log"

bash test_script.sh -f ${matrix_list_file} -p ${program_cuSDDMM} -n "${k128_results_path}cuSDDMM_128" -k 128
bash test_script.sh -f ${matrix_list_file} -p ${program_ASpT_128} -n "${k128_results_path}ASpT_128" -k 128
bash test_script.sh -f ${matrix_list_file} -p ${program_RoDe} -n "${k128_results_path}RoDe_128" -k 128

for A in "${ALPHA[@]}"; do
  for D in "${DELTA[@]}"; do
    bash test_script.sh -f ${matrix_list_file} -p ${program_BSMR} -n "${k128_results_path}BSMR_128_a_${A}_d_${D}" -k 128 -a ${A} -d ${D}
  done
done

./analyze_results ${log_files_k128} > ${results_path}analysis_results_k128.log
echo "Analysis for k=128 results saved to ${results_path}analysis_results_k128.log"
