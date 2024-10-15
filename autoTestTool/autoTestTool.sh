#!/bin/bash

# 测试文件路径
test_file_folder="$(pwd)/../dataset/matrix_5000_5000_/"

# 设置多个K
K=(256 500 1000 2000 3000 4000 5000)
echo -n "* The number of test K is 6, which are :"
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

log_file_suffix=".log"

# 储存测试生成的文件
testFolder="testFolder"
if [ ! -e ${testFolder} ]; then
    mkdir ${testFolder}
fi

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

create_log_file(){
  local filename=$1

  # 避免有同名文件
  if [ -e "${filename}${log_file_suffix}" ]; then
    local file_id=1
    while [ -e "${filename}${file_id}${log_file_suffix}" ]
    do
      ((file_id++))
    done
    filename="${filename}${file_id}"
  fi

  echo "* Create file: ${filename}${log_file_suffix}"
  touch ${filename}${log_file_suffix}
}

create_log_file ${zcx_test_log_filename}
create_log_file ${isratnisa_test_log_filename}

# 开始测试
autoTest(){
  # 使用 find 命令读取目录中的所有文件名，并存储到数组中
  filesList=($(find "${test_file_folder}" -maxdepth 1 -type f -printf '%f\n'))
  numTestFiles=${#filesList[@]}
  echo "* Number of test files: = ${numTestFiles}"

  echo "* Start test..."
  echo -e "@numTestFiles : ${numTestFiles} @\n" >> $2${log_file_suffix}
  for file in "${filesList[@]}"; do
    echo -e "\t * ${test_file_folder}$file start testing..."
    for k in "${K[@]}"; do
      echo -e "\t\t * K = ${k} start testing..."
      $1 ${test_file_folder}${file} ${k} 192 50000 >> $2${log_file_suffix}
      echo -e "\n---next---\n" >> $2${log_file_suffix}
      echo -e "\t\t * K = ${k} end test"
    done
    echo -e "\n---next---\n" >> $2${log_file_suffix}
    echo -e "\t * ${test_file_folder}$file end test"
  done

  echo "* End test"
  echo "* Test information file: $(pwd)/${2}${log_file_suffix}"
}

autoTest ${zcx_program_path}${zcx_program_name} ${zcx_test_log_filename}
autoTest ${isratnisa_program_path}${isratnisa_program_name} ${isratnisa_test_log_filename}

# 编译分析结果程序
g++ autoAnalysisResults.cpp -o autoAnalysisResults

numResultData=$((${numTestFiles}*${#K[@]}))
echo "* Number of test data : ${numResultData}"
"$(pwd)/autoAnalysisResults" ${numResultData} ${zcx_test_log_filename}${log_file_suffix} ${isratnisa_test_log_filename}${log_file_suffix}