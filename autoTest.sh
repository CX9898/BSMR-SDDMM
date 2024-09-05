#!/bin/bash

test_file_folder="$(pwd)/dataset"

test_file1="${test_file_folder}/matrix_3000_7000_313110.mtx" # small
test_file2="${test_file_folder}/matrix_2000_12000_746000.mtx" # small
test_file3="${test_file_folder}/matrix_300000_103000_69000000.mtx"
test_file4="${test_file_folder}/matrix_35000_35000_422000.mtx" # small
test_file5="${test_file_folder}/matrix_549000_549000_926000.mtx"
test_file6="${test_file_folder}/matrix_426000_426000_1000000.mtx"
test_file7="${test_file_folder}/matrix_37000_37000_368000.mtx" # small
test_file8="${test_file_folder}/matrix_4000_4000_88000.mtx" # small
test_file9="${test_file_folder}/matrix_106000_106000_3000000.mtx"
test_file10="${test_file_folder}/matrix_685000_685000_8000000.mtx"
test_file11="${test_file_folder}/matrix_916000_916000_5000000.mtx"
test_file12="${test_file_folder}/matrix_326000_326000_1000000.mtx"
test_file13="${test_file_folder}/matrix_197000_197000_2000000.mtx"
test_file14="${test_file_folder}/matrix_390000_390000_2000000.mtx"
test_file15="${test_file_folder}/matrix_260000_260000_4000000.mtx"
test_file16="${test_file_folder}/matrix_241000_241000_561000.mtx"
test_file17="${test_file_folder}/matrix_36000_36000_4000000.mtx" # small

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

${program_path}${program_name} ${test_file1} >> ${test_log_file_name}
echo "${test_file1} test finish"

echo "------------next------------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file2} >> ${test_log_file_name}
echo "${test_file2} test finish"

echo "------------next------------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file3} >> ${test_log_file_name}
echo "${test_file2} test finish"

echo "------------next------------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file4} >> ${test_log_file_name}
echo "${test_file4} test finish"

echo "------------next------------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file5} >> ${test_log_file_name}
echo "${test_file4} test finish"

echo "------------next------------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file6} >> ${test_log_file_name}
echo "${test_file4} test finish"

echo "------------next------------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file7} >> ${test_log_file_name}
echo "${test_file7} test finish"

echo "------------next------------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file8} >> ${test_log_file_name}
echo "${test_file8} test finish"

echo "------------next------------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file9} >> ${test_log_file_name}
echo "${test_file4} test finish"

echo "------------next------------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file10} >> ${test_log_file_name}
echo "${test_file4} test finish"

echo "------------next------------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file11} >> ${test_log_file_name}
echo "${test_file4} test finish"

echo "------------next------------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file12} >> ${test_log_file_name}
echo "${test_file4} test finish"

echo "------------next------------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file13} >> ${test_log_file_name}
echo "${test_file4} test finish"

echo "------------next------------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file14} >> ${test_log_file_name}
echo "${test_file4} test finish"

echo "------------next------------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file15} >> ${test_log_file_name}
echo "${test_file4} test finish"

echo "------------next------------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file16} >> ${test_log_file_name}
echo "${test_file4} test finish"

echo "------------next------------" >> ${test_log_file_name}
${program_path}${program_name} ${test_file17} >> ${test_log_file_name}
echo "${test_file17} test finish"

echo "End test"
echo "Test information file: $(pwd)${test_log_file_name}"