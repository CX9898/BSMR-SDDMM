#!/bin/bash

##############################################################################################
# Script setting

BSMR_build_folder_path="${script_file_path}build_BSMR/"
BSMR_cmake_file_path="${script_file_path}../"
BSMR_program_path="${BSMR_build_folder_path}"
BSMR_program_name="BSMR-sddmm"

cuSDDMM_build_folder_path="${script_file_path}build_cuSDDMM/"
cuSDDMM_cmake_file_path="${script_file_path}../baselines/cuSDDMM_SDDMM/"
cuSDDMM_program_path="${cuSDDMM_build_folder_path}"
cuSDDMM_program_name="cuSDDMM-sddmm"

ASpT_build_folder_path="${script_file_path}build_ASpT/"
ASpT_file_path="${script_file_path}../baselines/ASpT_SDDMM_GPU/"
ASpT_32_program_name="ASpT-sddmm-32"
ASpT_128_program_name="ASpT-sddmm-128"

RoDe_build_folder_path="${script_file_path}build_RoDe/"
RoDe_cmake_file_path="${script_file_path}../baselines/RoDe_SDDMM/"
RoDe_program_path="${RoDe_build_folder_path}"
RoDe_program_name="RoDe-sddmm"

Sputnik_build_folder_path="${script_file_path}build_Sputnik/"
Sputnik_cmake_file_path="${script_file_path}../baselines/Sputnik_SDDMM/"
Sputnik_program_path="${Sputnik_build_folder_path}"
Sputnik_program_name="Sputnik-sddmm"

BSA_build_folder_path="${script_file_path}build_BSA/"
BSA_cmake_file_path="${script_file_path}../baselines/BSA_SpMM/"
BSA_program_path="${BSA_build_folder_path}"
BSA_program_name="BSA-spmm"

##############################################################################################

# 参数1: 构建地址
# 参数2: "CMakeLists.txt"文件路径
build_program(){
  local build_path=${1}
  local cmake_file_path=${2}

  echo "${print_tag}Building the program..."
  if [ ! -e ${build_path} ]; then
      mkdir ${build_path}
  fi
  cmake -S ${cmake_file_path} -B ${build_path} -DCMAKE_BUILD_TYPE=Release > /dev/null
  cmake --build ${build_path} > /dev/null
  echo "${print_tag}Build complete : ${build_path}"
}

# 编译程序
build_program ${BSMR_build_folder_path} ${BSMR_cmake_file_path}
build_program ${cuSDDMM_build_folder_path} ${cuSDDMM_cmake_file_path}
build_program ${RoDe_build_folder_path} ${RoDe_cmake_file_path}
build_program ${Sputnik_build_folder_path} ${Sputnik_cmake_file_path}
build_program ${BSA_build_folder_path} ${BSA_cmake_file_path}

mkdir ${ASpT_build_folder_path}
nvcc -o ${ASpT_build_folder_path}${ASpT_32_program_name} ${ASpT_file_path}sddmm_32.cu -O3 -Xcompiler -fopenmp -arch=sm_80 > /dev/null  2> /dev/null
nvcc -o ${ASpT_build_folder_path}${ASpT_128_program_name} ${ASpT_file_path}sddmm_128.cu -O3 -Xcompiler -fopenmp -arch=sm_80 > /dev/null  2> /dev/null
echo "${print_tag}Build complete : ${ASpT_build_folder_path}"
