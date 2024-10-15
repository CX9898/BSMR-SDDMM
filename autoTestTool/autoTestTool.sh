#!/bin/bash

# 测试文件路径
test_file_folder="$(pwd)/../dataset/matrix_5000_5000_/"

# 设置多个K
K=(256 500)
echo -n "* The number of test K is ${#K[@]}, which are :"
for element in "${K[@]}"; do
    echo -n " $element"
done
echo

zcx_build_folder_name="build_zcx"
zcx_cmake_file_path="$(pwd)/../"

isratnisa_build_folder_name="build_isratnisa"
isratnisa_cmake_file_path="$(pwd)/../isratnisa/HiPC18"

zcx_program_path="$(pwd)/${zcx_build_folder_name}/"
zcx_program_name="sddmm-gpu"

isratnisa_program_path="$(pwd)/${isratnisa_build_folder_name}/"
isratnisa_program_name="isratnisa-sddmm"

# 日志文件名
zcx_test_log_filename="zcxTestLog"
isratnisa_test_log_filename="isratnisaTestLog"

# 储存测试生成的文件
testFolder="testFolder"
if [ ! -e ${testFolder} ]; then
    mkdir ${testFolder}
fi

# 参数1: 构建地址
# 参数2: "CMakeLists.txt"文件路径
build_program(){
  echo "* Building the program..."
  if [ ! -e ${1} ]; then
      mkdir ${1}
  fi
  cmake -S ${2} -B ${1} > /dev/null
  cmake --build ${1} > /dev/null
  echo "* Build complete : $(pwd)/${1}"
}

build_program ${zcx_build_folder_name} ${zcx_cmake_file_path}
build_program ${isratnisa_build_folder_name} ${isratnisa_cmake_file_path}

# 创建不重名的日志文件, 并且将日志文件名更新在全局变量`log_filename`中
# 参数1 : 原始日志文件名
create_log_file(){
  local log_filename=$1
  local log_file_suffix=".log"

  # 避免有同名文件
  if [ -e "${log_filename}${log_file_suffix}" ]; then
    local file_id=1
    while [ -e "${log_filename}${file_id}${log_file_suffix}" ]
    do
      ((file_id++))
    done
    log_filename="${log_filename}${file_id}"
  fi
  log_file=${log_filename}${log_file_suffix}
  echo "* Create file: ${log_file}"
  touch ${log_file}
}

create_log_file ${zcx_test_log_filename}
zcx_test_log_file=${log_file}
create_log_file ${isratnisa_test_log_filename}
isratnisa_test_log_file=${log_file}

# 参数1 : 程序的路径
# 参数2 : 测试日志文件
autoTest(){
  # 使用 find 命令读取目录中的所有文件名，并存储到数组中
  local filesList=($(find "${test_file_folder}" -maxdepth 1 -type f -printf '%f\n'))
  numTestFiles=${#filesList[@]}
  echo "* Number of test files: = ${numTestFiles}"

  local num_K=${#K[@]}
  echo "* Start test..."
  echo -e "@numTestFiles : ${numTestFiles} @\n" >> ${2}
  echo -e "@num_K : ${num_K} @\n" >> ${2}

  local file_id=1
  for file in "${filesList[@]}"; do
    echo -e "\t * file_id : ${file_id} ${test_file_folder}$file start testing..."
    local k_id=1

    for k in "${K[@]}"; do
      echo -e "\t\t * k_id : ${k_id} K = ${k} start testing..."
      $1 ${test_file_folder}${file} ${k} 192 50000 >> ${2}
#      if [ "${k}" -ne "${K[$((num_K-1))]}" ]; then
        echo -e "\n---next---\n" >> ${2}
#      fi
      echo -e "\t\t * k_id : ${k_id} K = ${k} end test"
      ((k_id++))
    done

    echo -e "\t * file_id : ${file_id} ${test_file_folder}${file} end test"
    ((file_id++))
  done

  echo "* End test"
  echo "* Test information file: $(pwd)/${log_file}"
}

autoTest ${zcx_program_path}${zcx_program_name} ${zcx_test_log_file}
autoTest ${isratnisa_program_path}${isratnisa_program_name} ${isratnisa_test_log_file}

# 编译分析结果程序
g++ autoAnalysisResults.cpp -o autoAnalysisResults

numResultData=$((${numTestFiles} * ${#K[@]}))
echo "* Number of test data : ${numResultData}"
"$(pwd)/autoAnalysisResults" ${numResultData} ${zcx_test_log_file} ${isratnisa_test_log_file}