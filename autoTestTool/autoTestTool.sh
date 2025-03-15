#!/bin/bash

#
# Script Name: autoTestTool.sh
# Description: 自动测试工具. 用于
# Usage: ./autoTestTool.sh [参数]
#
# Notes:
# - 额外的注意事项或说明
# - 例如依赖项、支持的平台等
#
# Example:
# ./autoTestTool.sh -
#

##############################################################################################
# Script setting

data_split_symbol="\n---New data---\n"
test_done_symbol="\n---Test done---\n"

script_file_path="$(dirname "$0")/"

# 设置多个测试文件路径
test_file_folder_path_list=("${script_file_path}../dataset/test/matrix_5000_5000_/"
                            "${script_file_path}../dataset/test/matrix_10000_10000_/"
                            "${script_file_path}../dataset/test/matrix_15000_15000_/"
                            "${script_file_path}../dataset/test/matrix_20000_20000_/")

# 设置多个K
k_list=(256 512 1024 2048 3072 4096)

zcx_build_folder_path="${script_file_path}build_zcx/"
zcx_cmake_file_path="${script_file_path}../"
zcx_program_path="${zcx_build_folder_path}"
zcx_program_name="sddmm-gpu"

isratnisa_build_folder_path="${script_file_path}build_isratnisa/"
isratnisa_cmake_file_path="${script_file_path}../isratnisa/HiPC18/"
isratnisa_program_path="${isratnisa_build_folder_path}"
isratnisa_program_name="isratnisa-sddmm"

# 日志文件名
zcx_test_log_filename="zcxTestLog"
isratnisa_test_log_filename="isratnisaTestLog"

# 分析结果日志文件名
analysis_results_log_filename="analysisResultsLog"

log_file_suffix=".log"

auto_analysis_results_source_filename="${script_file_path}autoAnalysisResults.cpp"
auto_analysis_results_program="${script_file_path}autoAnalysisResults"

print_tag="* "

##############################################################################################
# function

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

# 创建不重名的日志文件, 并且将日志文件名更新在全局变量`log_file`中
# 参数1 : 原始日志文件名
create_log_file(){
  local log_filename=${1}

  # 避免有同名文件
  if [ -e "${script_file_path}${log_filename}${log_file_suffix}" ]; then
    local file_id=1
    while [ -e "${script_file_path}${log_filename}${file_id}${log_file_suffix}" ]
    do
      ((file_id++))
    done
    log_filename="${log_filename}${file_id}"
  fi
  log_file=${script_file_path}${log_filename}${log_file_suffix}
  echo "${print_tag}Create file: ${log_file}"
  touch ${log_file}
}

# 参数1 : 进行测试的程序
# 参数2 : 测试日志文件
autoTest(){
  local autoTest_program="${1}"
  local autoTest_autoTestlog_file="${2}"

  for ((i=0; i<50; i++))
  do
    echo -n "${print_tag}"
  done
  echo ""

  echo "${print_tag}Start test..."

  echo "${print_tag}Test program: ${autoTest_program}"
  echo "${print_tag}Test log file: ${autoTest_autoTestlog_file}"

  local sum_time=0

  local test_file_folder_id=1
  for test_file_folder_path in "${test_file_folder_path_list[@]}"; do

    # 使用 find 命令读取目录中的所有文件名，并存储到数组中
    local files_list=($(find "${test_file_folder_path}" -maxdepth 1 -type f -printf '%f\n'))

    echo "${print_tag}Test file folder : ${test_file_folder_path} [Remaining: $((${num_test_file_folder} - ${test_file_folder_id}))]"

    local numTestFiles=${#files_list[@]}
    echo "${print_tag}Number of test files in the test folder: ${numTestFiles}"

    echo -e "[Test file folder : ${test_file_folder_path}]\n" >> ${autoTest_autoTestlog_file}
    echo -e "[numTestFiles : ${numTestFiles}]\n" >> ${autoTest_autoTestlog_file}
    echo -e "[num_k : ${num_k}]\n" >> ${autoTest_autoTestlog_file}

    local file_id=1
    for file in "${files_list[@]}"; do
      echo -e "${print_tag}\t${test_file_folder_path}$file start testing... [Remaining: $((${numTestFiles} - ${file_id}))]"

      local execution_time=0

      local k_id=1
      for k in "${k_list[@]}"; do
        echo -e "${print_tag}\t\tK = ${k} start testing... [Remaining: $((${num_k} - ${k_id}))]"
        echo -e ${data_split_symbol} >> ${autoTest_autoTestlog_file}
        local start_time=$(date +%s.%N)
        ${autoTest_program} -f ${test_file_folder_path}${file} -k ${k} -Y 192 -X 50000 >> ${autoTest_autoTestlog_file}
        local end_time=$(date +%s.%N)
        execution_time=$(echo "$end_time - $start_time" | bc)
        sum_time+=${execution_time}
        echo -e "${print_tag}\t\tExecution time: ${execution_time} seconds"
        ((k_id++))
      done

      ((file_id++))
    done

    ((test_file_folder_id++))
  done

  echo -e ${test_done_symbol} >> ${autoTest_autoTestlog_file}

  echo "${print_tag}Test done"
  echo "Total time spent: ${sum_time} seconds"
  echo "${print_tag}Test information file: ${autoTest_autoTestlog_file}"

  for ((i=0; i<50; i++))
  do
    echo -n "${print_tag}"
  done
  echo ""
}
##############################################################################################
# main

num_test_file_folder=${#test_file_folder_path_list[@]}
echo -n "${print_tag}The number of test file folder is ${num_test_file_folder}, which are :"
for element in "${test_file_folder_path_list[@]}"; do
    echo -n " $element"
done
echo

num_k=${#k_list[@]}
echo -n "${print_tag}The number of test k is ${num_k}, which are :"
for element in "${k_list[@]}"; do
    echo -n " $element"
done
echo

# 编译程序
build_program ${zcx_build_folder_path} ${zcx_cmake_file_path}
build_program ${isratnisa_build_folder_path} ${isratnisa_cmake_file_path}

# 编译"分析结果"程序
g++ ${auto_analysis_results_source_filename} -o ${auto_analysis_results_program}
echo "${print_tag}Auto analysis results program : ${auto_analysis_results_source_filename}"

# 创建日志文件
create_log_file ${zcx_test_log_filename}
zcx_test_log_file=${log_file}
create_log_file ${isratnisa_test_log_filename}
isratnisa_test_log_file=${log_file}
create_log_file ${analysis_results_log_filename}
analysis_results_log_file=${log_file}

# 开始测试
autoTest ${isratnisa_program_path}${isratnisa_program_name} ${isratnisa_test_log_file}
autoTest ${zcx_program_path}${zcx_program_name} ${zcx_test_log_file}

# 分析结果
echo "${print_tag}Start analyzing results..."
${auto_analysis_results_program} ${zcx_test_log_file} ${isratnisa_test_log_file} >> ${analysis_results_log_file}
echo "${print_tag}Results analysis completed: ${analysis_results_log_file}"