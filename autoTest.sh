#!/bin/bash

test_file1="dataset/matrix_3000_7000_313110.mtx"
test_file2="dataset/matrix_2000_12000_746000.mtx"

test_file4="dataset/matrix_35000_35000_422000.mtx"


test_file7="dataset/matrix_37000_37000_368000.mtx"
test_file8="dataset/matrix_4000_4000_88000.mtx"








test_file17="dataset/matrix_36000_36000_4000000.mtx"

program_path="./cmake-build-debug/"
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
${program_path}${program_name} ${test_file1} >> ${test_log_file_name}
echo "------next------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file2} >> ${test_log_file_name}
echo "------next------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file4} >> ${test_log_file_name}
echo "------next------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file7} >> ${test_log_file_name}
echo "------next------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file8} >> ${test_log_file_name}
echo "------next------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file17} >> ${test_log_file_name}

echo "End test"
echo "Test information filename: ${test_log_file_name}"