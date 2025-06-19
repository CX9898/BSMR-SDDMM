#!/bin/bash

# 设置变量
results_path="results_dataset_of_1/"
dataset_path="./dataset_of_suiteSparse/"
matrix_list_file="${dataset_path}matrix_file_list.txt"

program_cuSDDMM="./build_cuSDDMM/cuSDDMM-sddmm"
program_ASpT_32="./build_ASpT/ASpT-sddmm-32"
program_ASpT_128="./build_ASpT/ASpT-sddmm-32"
program_BSA="./build_BSA/BSA-spmm"
program_RoDe="./build_RoDe/RoDe-sddmm"

bash test_script.sh -f ${matrix_list_file} -p ${program_cuSDDMM} -n "${results_path}cuSDDMM_32" -k 32
bash test_script.sh -f ${matrix_list_file} -p ${program_cuSDDMM} -n "${results_path}cuSDDMM_128" -k 128
bash test_script.sh -f ${matrix_list_file} -p ${program_ASpT_32} -n "${results_path}ASpT_32" -k 32
bash test_script.sh -f ${matrix_list_file} -p ${program_ASpT_128} -n "${results_path}ASpT_128" -k 128
bash test_script.sh -f ${matrix_list_file} -p ${program_RoDe} -n "${results_path}RoDe_32" -k 32
bash test_script.sh -f ${matrix_list_file} -p ${program_RoDe} -n "${results_path}RoDe_128" -k 128
bash test_script.sh -f ${matrix_list_file} -p ${program_BSA} -n "${results_path}BSA_32" -k 128
