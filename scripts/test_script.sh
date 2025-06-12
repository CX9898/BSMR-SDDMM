#!/bin/bash

#
# Script Name: test_script.sh
# Description: 用于自动化测试程序的脚本. 该脚本使用给定的测试文件列表和测试程序进行测试.
# Usage: ./test_script.sh [参数] [参数值] ...
#
# Options:
# - -f : 测试文件列表
# - -p : 目标测试程序
# - -n : 日志文件名(可选)
# - -k : k值(可选)
# - -a : similarityThresholdAlpha值(可选)
# - -b : columnNonZeroThresholdBeta值(可选)
# Notes:
# -
#
# Example:
# bash test_script.sh -f test_list.txt -p ./build/sddmm-gpu -n results -k 32
#

##############################################################################################
# Script setting

print_tag="* "

data_split_symbol="\n---New data---\n"
test_done_symbol="\n---Test done---\n"

script_file_path="$(dirname "$0")/"

log_file_suffix=".log"

##############################################################################################
# Function

## 创建不重名的日志文件, 并且将日志文件名更新在全局变量`updated_log_file`中
## 参数1 : 原始日志文件名
#create_log_file(){
#  local log_filename=${1}
#
#  # 避免有同名文件
#  if [ -e "${script_file_path}${log_filename}${log_file_suffix}" ]; then
#    local file_id=1
#    while [ -e "${script_file_path}${log_filename}_${file_id}${log_file_suffix}" ]
#    do
#      ((file_id++))
#    done
#    log_filename="${log_filename}_${file_id}"
#  fi
#  updated_log_file=${script_file_path}${log_filename}${log_file_suffix}
#  echo "${print_tag}Create file: ${updated_log_file}"
#  touch ${updated_log_file}
#}

# 参数1 : 进行测试的程序
# 参数2 : 测试日志文件
testTool(){
  local autoTest_program="${1}"
  local autoTest_autoTestlog_file="${2}"

  for ((i=0; i<50; i++))
  do
    echo -n "${print_tag}"
  done
  echo ""

  echo "${print_tag}Start test..."

  echo "${print_tag}K = ${k}"

  echo "${print_tag}Test program: ${autoTest_program}"
  echo "${print_tag}Test log file: ${autoTest_autoTestlog_file}"

  local numTestFiles=${#test_file_list[@]}
  echo "${print_tag}Number of test files: ${numTestFiles}"

  echo -e "[numTestFiles : ${numTestFiles}]\n" >> ${autoTest_autoTestlog_file}

  local sum_time=0

  local file_id=1
  for file in "${test_file_list[@]}"; do

    echo -e ${data_split_symbol} >> ${autoTest_autoTestlog_file}

    echo -e "${print_tag}\t\"${autoTest_program} -f ${file} -k ${k} -a ${similarityThresholdAlpha} -b ${columnNonZeroThresholdBeta}\" start testing... [Remaining: $((${numTestFiles} - ${file_id}))]"

    local execution_time=0

    local start_time=$(date +%s.%N)
    ${autoTest_program} -f ${file} -k ${k} -a ${similarityThresholdAlpha} -b ${columnNonZeroThresholdBeta} >> ${autoTest_autoTestlog_file}
    local end_time=$(date +%s.%N)

    execution_time=$(echo "$end_time - $start_time" | bc)
    echo -e "${print_tag}\t\tExecution time: ${execution_time} seconds"
    sum_time=$(echo "$sum_time + $execution_time" | bc)

    ((file_id++))
  done

  echo -e ${test_done_symbol} >> ${autoTest_autoTestlog_file}

  echo "${print_tag}Test done"
  echo "${print_tag}Total time spent: ${sum_time} seconds"
  echo "${print_tag}Test information file: ${autoTest_autoTestlog_file}"

  for ((i=0; i<50; i++))
  do
    echo -n "${print_tag}"
  done
  echo ""
}

##############################################################################################
# main

# 解析参数
test_file_list_file=""
target_program=""
target_log_filename="results"
k=32
similarityThresholdAlpha=0.3
columnNonZeroThresholdBeta=4
while getopts "f:p:k:n:a:b:" opt; do
    case ${opt} in
        f) test_file_list_file="$OPTARG" ;;   # 处理 -f 选项(列表文件)
        p) target_program="$OPTARG" ;;  # 处理 -p 选项(程序路径)
        n) target_log_filename="$OPTARG" ;;  # 处理 -n 选项(日志文件名)
        k) k="$OPTARG" ;;  # 处理 -k 选项(k值)
        a) similarityThresholdAlpha="$OPTARG" ;;  # 处理 -a 选项(similarityThresholdAlpha值)
        b) columnNonZeroThresholdBeta="$OPTARG" ;;  # 处理 -b 选项(columnNonZeroThresholdBeta值)
        ?) echo "用法: $0 -f <列表文件> -p <程序> -n <日志文件名> -k <k值>"
           exit 1 ;;
    esac
done

echo "${print_tag}test file list file: $test_file_list_file"
echo "${print_tag}target program: $target_program"

matrixPath="$(dirname "$test_file_list_file")/"

# 检查列表文件是否存在, 并读取文件内容
test_file_list=""
if [ -f "$test_file_list_file" ]; then
    # 读取文件内容，每行作为一个路径
    mapfile -t test_file_list < "$test_file_list_file"

    # 遍历数组, 将绝对路径添加到文件名
    for i in "${!test_file_list[@]}"; do
        test_file_list[$i]="${script_file_path}${matrixPath}${test_file_list[$i]}"
    done
else
    echo "Error: 文件 $test_file_list_file 不存在!请提供有效的列表文件."
    exit 1
fi

# 检查目标测试程序是否存在
if [ ! -f "$target_program" ]; then
    echo "Error: 文件 $target_program 不存在! 请提供有效的程序."
    exit 1
fi

# 创建日志文件
target_log_file="${target_log_filename}${log_file_suffix}"
> "$target_log_file"

# 开始测试
testTool ${target_program} ${target_log_file}