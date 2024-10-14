#!/bin/bash

testFolder="testFolder"
if [ ! -e ${testFolder} ]; then
    mkdir ${testFolder}
fi

# 编译程序 zcx
echo "* Building the program..."
build_zcx="build_zcx"
if [ ! -e ${build_zcx} ]; then
    mkdir ${build_zcx}
fi
zcx_cmakeFilePath="../"
cmake -S ${zcx_cmakeFilePath} -B ${build_zcx}
cmake --build ${build_zcx}

zcx_program_path="$(pwd)/${build_zcx}/"
zcx_program_name="sddmm-gpu"

# 编译程序 isratnisa
echo "* Building the program..."
build_isratnisa="build_isratnisa"
if [ ! -e ${build_isratnisa} ]; then
    mkdir ${build_isratnisa}
fi
isratnisa_cmakeFilePath="../isratnisa/HiPC18"
cmake -S ${isratnisa_cmakeFilePath} -B ${build_isratnisa}
cmake --build ${build_isratnisa}

isratnisa_program_path="$(pwd)/${build_isratnisa}/"
isratnisa_program_name="isratnisa-sddmm"

# 日志文件名
zcx_test_log_file_name="zcxTestLog"
isratnisa_test_log_file_name="isratnisaTestLog"

# 避免有同名文件
if [ -e "${zcx_test_log_file_name}" ]; then
  file_id=1
  while [ -e "${zcx_test_log_file_name}${file_id}" ]
  do
    ((file_id++))
  done
  zcx_test_log_file_name="${zcx_test_log_file_name}${file_id}"
fi
echo "* Create file: ${zcx_test_log_file_name}"
touch ${zcx_test_log_file_name}

# 避免有同名文件
if [ -e "${isratnisa_test_log_file_name}" ]; then
  file_id=1
  while [ -e "${isratnisa_test_log_file_name}${file_id}" ]
  do
    ((file_id++))
  done
  isratnisa_test_log_file_name="${isratnisa_test_log_file_name}${file_id}"
fi
echo "* Create file: ${isratnisa_test_log_file_name}"
touch ${isratnisa_test_log_file_name}

# 测试文件路径
test_file_folder="$(pwd)/../dataset/test/"

# 使用 find 命令读取目录中的所有文件名，并存储到数组中
filesList=($(find "${test_file_folder}" -maxdepth 1 -type f -printf '%f\n'))
numTestFiles=${#filesList[@]}

echo "numTestFiles = ${numTestFiles}"

K=256

# 开始测试
autoTest(){
  echo "* Start test..."
  echo -e "numTestFiles : ${numTestFiles}\n" >> $2
  for file in "${filesList[@]}"; do
    echo "** ${test_file_folder}$file start testing..."
    $1 ${test_file_folder}${file} >> $2
    echo -e "\n---next---\n" >> $2
    echo "** ${test_file_folder}$file end test"
  done

  echo "* End test"
  echo "* Test information file: $(pwd)/$2"
}

autoTest ${zcx_program_path}${zcx_program_name} ${zcx_test_log_file_name}

# 开始测试
autoTest2(){
  echo "* Start test..."
  echo -e "\t * numTestFiles : ${numTestFiles}\n" >> $2
  for file in "${filesList[@]}"; do
    echo -e "\t ** ${test_file_folder}$file start testing..."
    $1 ${test_file_folder}${file} ${K} 192 50000 >> $2
    echo -e "\n---next---\n" >> $2
    echo -e "\t ** ${test_file_folder}$file end test"
  done

  echo "* End test"
  echo "* Test information file: $(pwd)/${2}"
}
autoTest2 ${isratnisa_program_path}${isratnisa_program_name} ${isratnisa_test_log_file_name}

# 编译分析结果程序
g++ autoAnalysisResults.cpp -o autoAnalysisResults

"$(pwd)/autoAnalysisResults" ${numTestFiles} ${zcx_test_log_file_name} ${isratnisa_test_log_file_name}