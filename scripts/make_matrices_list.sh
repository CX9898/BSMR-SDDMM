#!/bin/bash

#
# Script Name: make_matrices_list.sh
# Description: 创建包含所有 .mtx 和 .smtx 矩阵文件路径的两个列表文件。
#              会遍历指定目录下的所有 .mtx 和 .smtx 文件，并将它们的相对路径
#              分别保存到 matrix_file_list_mtx.txt 和 matrix_file_list_smtx.txt。
# Usage: ./make_matrices_list.sh [目标文件夹]

##############################################################################################

# 获取目标文件夹
target_folder=$1

# 检查是否指定了目标文件夹
if [ -z "$target_folder" ]; then
  echo "用法: $0 [目标文件夹]"
  exit 1
fi

# 移除末尾斜杠（如果有）
target_folder="${target_folder%/}"

# 输出文件路径
output_file_mtx="${target_folder}/matrix_file_list_mtx.txt"
output_file_smtx="${target_folder}/matrix_file_list_smtx.txt"

# 创建并清空输出文件
> "$output_file_mtx"
> "$output_file_smtx"

# 处理 .mtx 文件
find "$target_folder" -type f -name "*.mtx" | sed "s|^$target_folder/||" > "$output_file_mtx"

# 处理 .smtx 文件
find "$target_folder" -type f -name "*.smtx" | sed "s|^$target_folder/||" > "$output_file_smtx"

echo ".mtx 文件路径已保存到 $output_file_mtx"
echo ".smtx 文件路径已保存到 $output_file_smtx"
