#!/bin/bash

#
# Script Name: make_matrices_list.sh
# Description: 创建一个包含所有矩阵文件路径的列表文件. 该脚本会遍历指定目录下的所有 .mtx 和 .smtx 文件, 并将它们的相对路径保存到一个文本文件中.
# Usage: ./make_matrices_list.sh [目标文件夹]

##############################################################################################

# 目标文件夹
target_folder=$1

# 移除 target_folder 末尾的斜杠(/)
target_folder="${target_folder%/}"

output_file="${target_folder}/matrix_file_list.txt"   # 保存文件路径的文件

touch "$output_file"
# 清空输出文件(如果已存在)
> "$output_file"

# 遍历所有 .mtx 或 .smtx 结尾的文件并保存路径
find "$target_folder" -type f \( -name "*.mtx" -o -name "*.smtx" \) | sed "s|^$target_folder/||" > "$output_file"

echo "所有文件路径已保存到 $output_file"