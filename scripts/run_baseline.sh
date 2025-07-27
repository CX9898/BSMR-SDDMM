#!/bin/bash

# 设置变量
results_path="./results_suiteSparse_dataset/"
dataset_path="./suiteSparse_dataset/"

k32_results_path="${results_path}k32/"
k64_results_path="${results_path}k64/"
k128_results_path="${results_path}k128/"
k256_results_path="${results_path}k256/"

# 创建结果目录
mkdir -p ${k32_results_path}
mkdir -p ${k64_results_path}
mkdir -p ${k128_results_path}
mkdir -p ${k256_results_path}

bash make_matrices_list.sh ${dataset_path}
matrix_list_file="${dataset_path}matrix_file_list_mtx.txt"

# run cuSPARSE
program_cuSPARSE="./build_cuSPARSE/cuSPARSE-sddmm"
bash test_script.sh -f ${matrix_list_file} -p ${program_cuSPARSE} -n "${k32_results_path}cuSPARSE_32" -k 32
bash test_script.sh -f ${matrix_list_file} -p ${program_cuSPARSE} -n "${k64_results_path}cuSPARSE_64" -k 64
bash test_script.sh -f ${matrix_list_file} -p ${program_cuSPARSE} -n "${k128_results_path}cuSPARSE_128" -k 128
bash test_script.sh -f ${matrix_list_file} -p ${program_cuSPARSE} -n "${k256_results_path}cuSPARSE_256" -k 256

# run ASpT
program_ASpT_32="./build_ASpT/ASpT-sddmm-32"
program_ASpT_128="./build_ASpT/ASpT-sddmm-128"
bash test_script.sh -f ${matrix_list_file} -p ${program_ASpT_32} -n "${k32_results_path}ASpT_32" -k 32
bash test_script.sh -f ${matrix_list_file} -p ${program_ASpT_128} -n "${k128_results_path}ASpT_128" -k 128

# run RoDe
program_RoDe="./build_RoDe/RoDe-sddmm"
bash test_script.sh -f ${matrix_list_file} -p ${program_RoDe} -n "${k32_results_path}RoDe_32" -k 32
bash test_script.sh -f ${matrix_list_file} -p ${program_RoDe} -n "${k128_results_path}RoDe_128" -k 128

# run Sputnik
program_Sputnik="./build_Sputnik/Sputnik-sddmm"
bash test_script.sh -f ${matrix_list_file} -p ${program_Sputnik} -n "${k32_results_path}Sputnik_32" -k 32
bash test_script.sh -f ${matrix_list_file} -p ${program_Sputnik} -n "${k64_results_path}Sputnik_64" -k 64
bash test_script.sh -f ${matrix_list_file} -p ${program_Sputnik} -n "${k128_results_path}Sputnik_128" -k 128
bash test_script.sh -f ${matrix_list_file} -p ${program_Sputnik} -n "${k256_results_path}Sputnik_256" -k 256

# run cuSDDMM
program_cuSDDMM="./build_cuSDDMM/cuSDDMM-sddmm"
bash test_script.sh -f ${matrix_list_file} -p ${program_cuSDDMM} -n "${k32_results_path}cuSDDMM_32" -k 32
bash test_script.sh -f ${matrix_list_file} -p ${program_cuSDDMM} -n "${k64_results_path}cuSDDMM_64" -k 64
bash test_script.sh -f ${matrix_list_file} -p ${program_cuSDDMM} -n "${k128_results_path}cuSDDMM_128" -k 128
bash test_script.sh -f ${matrix_list_file} -p ${program_cuSDDMM} -n "${k256_results_path}cuSDDMM_256" -k 256

# run TCGNN
source ~/miniconda3/etc/profile.d/conda.sh
conda activate TCGNN
python test_TCGNN.py --matrix_list ${matrix_list_file} -K 32 --log_file ${k32_results_path}TCGNN_32
python test_TCGNN.py --matrix_list ${matrix_list_file} -K 64 --log_file ${k64_results_path}TCGNN_64
python test_TCGNN.py --matrix_list ${matrix_list_file} -K 128 --log_file ${k128_results_path}TCGNN_128
python test_TCGNN.py --matrix_list ${matrix_list_file} -K 256 --log_file ${k256_results_path}TCGNN_256

# run FlashSparse
source ~/miniconda3/etc/profile.d/conda.sh
conda activate FlashSparse
python test_FlashSparse.py --matrix_list ${matrix_list_file} -K 32 --log_file ${k32_results_path}FlashSparse_32
python test_FlashSparse.py --matrix_list ${matrix_list_file} -K 64 --log_file ${k64_results_path}FlashSparse_64
python test_FlashSparse.py --matrix_list ${matrix_list_file} -K 128 --log_file ${k128_results_path}FlashSparse_128
python test_FlashSparse.py --matrix_list ${matrix_list_file} -K 256 --log_file ${k256_results_path}FlashSparse_256
