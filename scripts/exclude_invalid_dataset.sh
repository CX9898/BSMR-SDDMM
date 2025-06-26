#!/bin/bash

dataset_path="${1%/}/"
excluded_dataset_path="excluded_dataset/${dataset_path}"
echo "正在处理数据集路径: $dataset_path"
echo "排除无效数据集的路径: $excluded_dataset_path"

mkdir -p "${excluded_dataset_path}"

log_file="${excluded_dataset_path}excluded_files.log"

bash make_matrices_list.sh "${dataset_path}"
list_file="${dataset_path}/matrix_file_list.txt"

if [ ! -f "$list_file" ]; then
  echo "找不到文件: $list_file"
  exit 1
fi

while read -r filepath; do
  # 跳过空行
  [ -z "${filepath}" ] && continue

  full_path="${dataset_path}${filepath}"

  # 检查文件是否存在
  if [ ! -f "$full_path" ]; then
    echo "文件不存在: $full_path"
    continue
  fi

  # 获取第一行非注释
  first_valid_line=$(grep -v '^%' "$full_path" | head -n 1)

  # 检查是否是合法三元组
  if echo "$first_valid_line" | awk 'NF==3 && $1 ~ /^[0-9]+$/ && $2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ { exit 0 } { exit 1 }'; then
    read m n nnz <<< "$first_valid_line"

    if (( m < 5000 || n < 5000 )); then
      echo "$full_path -> 尺寸过小 (m=$m, n=$n)，将移除"
    else
      echo "$full_path -> m=$m, n=$n, nnz=$nnz"
      continue
    fi
  else
    echo "$full_path -> 第一有效行不是合法三元组: $first_valid_line"
  fi

  # 构造目标路径
  target_path="${excluded_dataset_path}${filepath}"
  target_dir=$(dirname "$target_path")

  # 创建目标目录
  mkdir -p "$target_dir"

  # 移动文件
  mv "$full_path" "$target_path"
  echo "$full_path -> 已移动到 $target_path"

  {
    echo "original: $full_path"
    echo "moved_to: $target_path"
    echo "---"
  } >> "$log_file"

done < "$list_file"
