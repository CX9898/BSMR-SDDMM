#!/bin/bash

# 解析参数
target_folder="results"
while getopts "f:p:n:" opt; do
    case ${opt} in
        n) target_folder="$OPTARG" ;;  # 处理 -n 选项(目标文件夹)
        ?) echo "用法: $0 -n <目标文件夹>"
           exit 1 ;;
    esac
done

# 移除 target_folder 末尾的斜杠(/)
target_folder="${target_folder%/}"

output_file="${target_folder}/matrix_file_list.txt"   # 保存文件路径的文件

# 清空输出文件(如果已存在)
> "$output_file"

# 遍历所有 .mtx 或 .smtx 结尾的文件并保存路径
find "$target_folder" -type f \( -name "*.mtx" -o -name "*.smtx" \) > "$output_file"

echo "所有文件路径已保存到 $output_file"