#!/bin/bash

test_log_file_name="autoTestLog"

# 避免有同名文件
if [ -e "${test_log_file_name}" ]; then
    file_id=1
    while [ -e "${test_log_file_name}${file_id}" ]
    do
      ((file_id++))
    done
    test_log_file_name="autoTestLog${file_id}"
fi

echo "* Create file: ${test_log_file_name}"
touch ${test_log_file_name}

test_file_folder="$(pwd)/../dataset/"

program_path="$(pwd)/../cmake-build-release/"
program_name="sddmm-gpu"

autoTest(){
    echo "* Start test..."

    # 使用 find 命令读取目录中的所有文件名，并存储到数组中
    filesList=($(find "${test_file_folder}" -maxdepth 1 -type f -printf '%f\n'))

#    for file in "${filesList[@]}"; do
#          echo "* ${test_file_folder}$file start testing..."
#          ${program_path}${program_name} ${test_file_folder}${file} >> ${test_log_file_name}
#          echo "* ${test_file_folder}$file end test"
#    done

    echo "* End test"
    echo "* Test information file: $(pwd)${test_log_file_name}"
}

autoTest

g++ autoAnalysisResults.cpp -o autoAnalysisResults

"$(pwd)/autoAnalysisResults" "$(pwd)/${test_log_file_name}"