#!/bin/bash

##############################################################################################
# Script setting

zcx_build_folder_path="${script_file_path}build_zcx/"
zcx_cmake_file_path="${script_file_path}../"
zcx_program_path="${zcx_build_folder_path}"
zcx_program_name="sddmm-gpu"

isratnisa_build_folder_path="${script_file_path}build_isratnisa/"
isratnisa_cmake_file_path="${script_file_path}../isratnisa/HiPC18/"
isratnisa_program_path="${isratnisa_build_folder_path}"
isratnisa_program_name="isratnisa-sddmm"

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
build_program ${isratnisa_build_folder_path} ${isratnisa_cmake_file_path}

