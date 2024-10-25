#!/bin/bash

data_split_symbol="\n---New data---\n"
test_done_symbol="\n---Test done---\n"

script_file_path="$(dirname "$0")/"

# 测试文件路径
test_file_folder_path="${script_file_path}../dataset/test/matrix_10000_10000_/"

# 设置多个K
K=(256 500 1000 2000 3000 4000 5000)
num_K=${#K[@]}
echo -n "* The number of test K is ${num_K}, which are :"
for element in "${K[@]}"; do
    echo -n " $element"
done
echo

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

auto_analysis_results_source_filename="${script_file_path}autoAnalysisResults.cpp"
auto_analysis_results_program="${script_file_path}autoAnalysisResults"

# 参数1: 构建地址
# 参数2: "CMakeLists.txt"文件路径
build_program(){
  local build_path=${1}
  local cmake_file_path=${2}

  echo "* Building the program..."
  if [ ! -e ${build_path} ]; then
      mkdir ${build_path}
  fi
  cmake -S ${cmake_file_path} -B ${build_path} -DCMAKE_BUILD_TYPE=Release > /dev/null
  cmake --build ${build_path} > /dev/null
  echo "* Build complete : ${build_path}"
}

build_program ${zcx_build_folder_path} ${zcx_cmake_file_path}
build_program ${isratnisa_build_folder_path} ${isratnisa_cmake_file_path}

# 创建不重名的日志文件, 并且将日志文件名更新在全局变量`log_file`中
# 参数1 : 原始日志文件名
log_file_suffix=".log"
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
  echo "* Create file: ${log_file}"
  touch ${log_file}
}

create_log_file ${zcx_test_log_filename}
zcx_test_log_file=${log_file}
create_log_file ${isratnisa_test_log_filename}
isratnisa_test_log_file=${log_file}

create_log_file ${analysis_results_log_filename}
analysis_results_log_file=${log_file}

# 参数1 : 进行测试的程序
# 参数2 : 测试日志文件
autoTest(){
  local autoTest_program=${1}
  local autoTest_autoTestlog_file=${2}

  # 使用 find 命令读取目录中的所有文件名，并存储到数组中
  local filesList=($(find "${test_file_folder_path}" -maxdepth 1 -type f -printf '%f\n'))

  local numTestFiles=${#filesList[@]}
  echo "* Number of test files: = ${numTestFiles}"

  local numResultData=$((${numTestFiles} * ${num_K}))
  echo "* Number of test data : ${numResultData}"

  echo "* Start test..."
  echo -e "@numTestFiles : ${numTestFiles} @\n" >> ${autoTest_autoTestlog_file}
  echo -e "@num_K : ${num_K} @\n" >> ${autoTest_autoTestlog_file}
  echo -e "@numResultData : ${numResultData} @\n" >> ${autoTest_autoTestlog_file}

  local file_id=1
  for file in "${filesList[@]}"; do
    echo -e "* \t${test_file_folder_path}$file start testing... [Remaining: $((${numTestFiles} - ${file_id}))]"

    local k_id=1
    for k in "${K[@]}"; do
      echo -e "* \t\tK = ${k} start testing... [Remaining: $((${num_K} - ${k_id}))]"
      echo -e ${data_split_symbol} >> ${autoTest_autoTestlog_file}
      local start_time=$(date +%s.%N)
      ${autoTest_program} "${test_file_folder_path}${file}" ${k} 192 50000 >> ${autoTest_autoTestlog_file}
      local end_time=$(date +%s.%N)
      echo -e "* \t\tExecution time: $(echo "$end_time - $start_time" | bc) seconds"
      ((k_id++))
    done

    ((file_id++))
  done

  echo -e ${test_done_symbol} >> ${autoTest_autoTestlog_file}

  echo "* End test"
  echo "* Test information file: ${autoTest_autoTestlog_file}"
}

autoTest ${isratnisa_program_path}${isratnisa_program_name} ${isratnisa_test_log_file}
autoTest ${zcx_program_path}${zcx_program_name} ${zcx_test_log_file}

# 编译分析结果程序
g++ ${auto_analysis_results_source_filename} -o ${auto_analysis_results_program}
echo "* Auto analysis results program : ${auto_analysis_results_source_filename}"

echo "* Start analyzing results..."
${auto_analysis_results_program} ${zcx_test_log_file} ${isratnisa_test_log_file} >> ${analysis_results_log_file}
echo "* Results analysis completed: ${analysis_results_log_file}"