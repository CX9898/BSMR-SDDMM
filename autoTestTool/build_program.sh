#!/bin/bash

##############################################################################################
# Script setting

zcx_build_folder_path="${script_file_path}build_zcx/"
zcx_cmake_file_path="${script_file_path}../"
zcx_program_path="${zcx_build_folder_path}"
zcx_program_name="sddmm-gpu"

HiPC18_build_folder_path="${script_file_path}build_HiPC18/"
HiPC18_cmake_file_path="${script_file_path}../HiPC18_SDDMM/"
HiPC18_program_path="${HiPC18_build_folder_path}"
HiPC18_program_name="HiPC18-sddmm"

ASpT_build_folder_path="${script_file_path}build_ASpT/"
ASpT_file_path="${script_file_path}../ASpT_SDDMM_GPU/"
ASpT_32_program_name="ASpT-sddmm-32"
ASpT_128_program_name="ASpT-sddmm-128"

RoDe_build_folder_path="${script_file_path}build_RoDe/"
RoDe_file_path="${script_file_path}../RoDe_SDDMM/"
RoDe_32_program_name="RoDe-sddmm-32"
RoDe_128_program_name="RoDe-sddmm-128"

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
build_program ${zcx_build_folder_path} ${zcx_cmake_file_path}
build_program ${HiPC18_build_folder_path} ${HiPC18_cmake_file_path}

mkdir ${ASpT_build_folder_path}
nvcc -o ${ASpT_build_folder_path}${ASpT_32_program_name} ${ASpT_file_path}sddmm_32.cu -O3 -Xcompiler -fopenmp -arch=sm_80
nvcc -o ${ASpT_build_folder_path}${ASpT_128_program_name} ${ASpT_file_path}sddmm_128.cu -O3 -Xcompiler -fopenmp -arch=sm_80

