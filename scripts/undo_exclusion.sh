#!/bin/bash

dataset_name=$1

if [ -z "$dataset_name" ]; then
  echo "❗ 请提供原始数据集路径（如: buchong）"
  exit 1
fi

log_file="excluded_dataset/${dataset_name}/excluded_files.log"

if [ ! -f "$log_file" ]; then
  echo "❗ 找不到日志文件: $log_file"
  exit 1
fi

echo "🔄 正在撤销: $log_file"

# 状态变量
orig=""
moved=""
declare -A processed_files

while IFS= read -r line; do
  if [[ "$line" == original:* ]]; then
    orig="${line#original: }"
  elif [[ "$line" == moved_to:* ]]; then
    moved="${line#moved_to: }"
  elif [[ "$line" == ---* ]]; then
    if [[ -n "$orig" && -n "$moved" ]]; then
      # 避免重复处理
      if [[ -n "${processed_files[$moved]}" ]]; then
        continue
      fi
      processed_files["$moved"]=1

      if [ -f "$moved" ]; then
        mkdir -p "$(dirname "$orig")"
        mv "$moved" "$orig"
        echo "✔ 已还原: $moved -> $orig"
      else
        echo "⚠ 文件不存在，跳过: $moved"
      fi
    fi
    orig=""
    moved=""
  fi
done < "$log_file"
