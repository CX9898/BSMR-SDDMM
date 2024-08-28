#!/bin/bash

test_file_folder="$(pwd)/dataset"

test_file1="${test_file_folder}/matrix_3000_7000_313110.mtx"
test_file2="${test_file_folder}/matrix_2000_12000_746000.mtx"

test_file4="${test_file_folder}/matrix_35000_35000_422000.mtx"


test_file7="${test_file_folder}/matrix_37000_37000_368000.mtx"
test_file8="${test_file_folder}/matrix_4000_4000_88000.mtx"







test_file17="${test_file_folder}/matrix_36000_36000_4000000.mtx"

program_path="$(pwd)/cmake-build-debug/"
program_name="sddmm-gpu"

test_log_file_name="autoTestLog"

if [ -e "${test_log_file_name}" ]; then
    file_id=1
    while [ -e "${test_log_file_name}${file_id}" ]
    do
      ((file_id++))
    done
    test_log_file_name="autoTestLog${file_id}"
fi

echo "Start test"

touch ${test_log_file_name}

echo "${test_file1} test finish"
${program_path}${program_name} ${test_file2} >> ${test_log_file_name}
echo "------------next------------" >> ${test_log_file_name}
echo "${test_file2} test finish"

${program_path}${program_name} ${test_file4} >> ${test_log_file_name}
echo "------------next------------" >> ${test_log_file_name}
echo "${test_file4} test finish"

${program_path}${program_name} ${test_file7} >> ${test_log_file_name}
echo "------------next------------" >> ${test_log_file_name}
echo "${test_file7} test finish"

${program_path}${program_name} ${test_file8} >> ${test_log_file_name}
echo "------------next------------" >> ${test_log_file_name}
echo "${test_file8} test finish"

${program_path}${program_name} ${test_file17} >> ${test_log_file_name}
echo "------------next------------" >> ${test_log_file_name}
echo "${test_file17} test finish"

echo "End test"
echo "Test information file: $(pwd)${test_log_file_name}"