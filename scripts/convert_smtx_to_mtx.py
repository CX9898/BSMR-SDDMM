import os
import sys

def parse_smtx_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 第一行：读取矩阵维度信息
    nrows, ncols, nnz = map(int, lines[0].strip().split(','))

    # 第二行：rowOffsets，长度应为 nrows+1
    row_offsets = list(map(int, lines[1].strip().split()))
    if len(row_offsets) != nrows + 1:
        raise ValueError(f"rowOffsets 长度错误，应为 {nrows+1}，但得到 {len(row_offsets)}")

    # 第三行：colIndices，总数应为 nnz
    col_indices = list(map(int, lines[2].strip().split()))
    if len(col_indices) != nnz:
        raise ValueError(f"colIndices 数量错误，应为 {nnz}，但得到 {len(col_indices)}")

    matrix_data = []
    for row in range(nrows):
        start = row_offsets[row]
        end = row_offsets[row + 1]
        for idx in range(start, end):
            col = col_indices[idx]
            matrix_data.append((row + 1, col + 1, 1.0))  # Matrix Market 使用1-based索引

    return nrows, ncols, matrix_data

def write_mtx_file(nrows, ncols, matrix_data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("%%MatrixMarket matrix coordinate real general\n")
        f.write("%-------------------------------------------------------------------------------\n")
        f.write("% Converted from .smtx format\n")
        f.write("%-------------------------------------------------------------------------------\n")
        f.write(f"{nrows} {ncols} {len(matrix_data)}\n")
        for row, col, val in matrix_data:
            f.write(f"{row} {col} {val}\n")

def convert_smtx_to_mtx(input_path, base_dir):
    input_full_path = os.path.join(base_dir, input_path)
    output_path = os.path.splitext(input_full_path)[0] + '.mtx'

    if not os.path.exists(input_full_path):
        print(f"[SKIP] File not found: {input_full_path}")
        return

    print(f"Converting: {input_full_path} -> {output_path}")
    nrows, ncols, matrix_data = parse_smtx_file(input_full_path)
    write_mtx_file(nrows, ncols, matrix_data, output_path)

def batch_convert(file_list_path, base_dir):
    if not os.path.exists(file_list_path):
        print(f"[ERROR] File list not found: {file_list_path}")
        return

    with open(file_list_path, 'r') as f:
        for line in f:
            rel_path = line.strip()
            if rel_path.endswith('.smtx'):
                convert_smtx_to_mtx(rel_path, base_dir)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python convert_smtx_to_mtx.py <matrix_file_list_smtx.txt>")
        sys.exit(1)

    file_list_path = sys.argv[1]
    base_dir = os.path.dirname(os.path.abspath(file_list_path)) # Base directory for relative paths
    batch_convert(file_list_path, base_dir)
