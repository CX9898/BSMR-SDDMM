#!/usr/bin/env python3
import argparse
import numpy as np
import scipy.io
import scipy.sparse as sp
import os
import csv
import sys

def mtx_to_npz(mtx_path, output_dir, graph_name=None):
    # 加载 .mtx 文件为 scipy 稀疏矩阵（COO格式）
    mat = scipy.io.mmread(mtx_path).tocoo()

    # 获取边信息
    src_li = mat.row.astype(np.int32)
    dst_li = mat.col.astype(np.int32)
    num_edges = mat.nnz
    num_nodes_src = mat.shape[0]
    num_nodes_dst = mat.shape[1]

    # 使用文件名作为图名称（如果未显式提供）
    if graph_name is None:
        graph_name = os.path.splitext(os.path.basename(mtx_path))[0]

    # 创建统一保存目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 生成 .npz 文件完整路径
    npz_path = os.path.join(output_dir, f"{graph_name}.npz")

    # 保存为 .npz 文件
    np.savez(
        npz_path,
        src_li=src_li,
        dst_li=dst_li,
        num_nodes_src=num_nodes_src,
        num_nodes_dst=num_nodes_dst,
        num_edges=num_edges
    )
    print(f"Saved: {graph_name}.npz to {output_dir}")

    return {
        'graph': graph_name,
        'num_nodes_src': num_nodes_src,
        'num_edges': num_edges,
        'num_nodes_dst': num_nodes_dst,
        'npz_path': npz_path
    }

def batch_convert(list_file):
    base_prefix = os.path.dirname(list_file)
    output_dir = os.path.join(base_prefix, "converted_npz")
    os.makedirs(output_dir, exist_ok=True)

    with open(list_file, 'r') as f:
        lines = f.readlines()

    summary = []
    for line in lines:
        relative_path = line.strip()
        mtx_path = os.path.join(base_prefix, relative_path)
        if not mtx_path.endswith(".mtx") or not os.path.isfile(mtx_path):
            print(f"Skipping invalid path: {mtx_path}")
            continue

        result = mtx_to_npz(mtx_path, output_dir)
        summary.append(result)

    # 写入 data_filter.csv（确保 npz_path 保留完整相对路径）
    csv_path = os.path.join(base_prefix, "data_filter.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['graph', 'num_nodes_src', 'num_edges', 'num_nodes_dst', 'npz_path'])
        writer.writeheader()
        writer.writerows(summary)
    print(f"Summary written to {csv_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_smtx_to_mtx.py <matrix_file_list_mtx.txt>")
        sys.exit(1)

    file_list_path = sys.argv[1]
    # parser = argparse.ArgumentParser(description="Convert .mtx files to .npz format for GAT/SDDMM input")
    # parser.add_argument("list_file", help="Path to file listing .mtx file paths (one per line)")
    # args = parser.parse_args()

    batch_convert(file_list_path)
