import os
import sys
import shutil
import subprocess

def is_valid_triplet(line):
    parts = line.strip().split()
    return len(parts) == 3 and all(p.isdigit() for p in parts[:2])

def clean_data_lines(lines):
    cleaned = []
    for line in lines:
        parts = line.strip().split()
        row, col = parts[0], parts[1]
        cleaned.append(f"{row} {col} 1")
    return cleaned

def process_matrix_file(filepath, excluded_path, log_file):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # 修改文件头，如果包含 complex 则替换为 real（可选）
    if lines and "complex" in lines[0]:
        lines[0] = lines[0].replace("complex", "real")

    comments = [line for line in lines if line.strip().startswith('%')]
    content = [line for line in lines if not line.strip().startswith('%') and line.strip()]

    if not content:
        reason = "无有效数据行"
        move_file(filepath, excluded_path, log_file, reason)
        return

    first_line = content[0].strip()
    if not is_valid_triplet(first_line):
        reason = f"第一有效行不是合法三元组: {first_line}"
        move_file(filepath, excluded_path, log_file, reason)
        return

    m, n, nnz = map(int, first_line.split())
    data_lines = content[1:]

    if m < 5000 or n < 5000:
        reason = f"尺寸过小 m={m}, n={n}"
        move_file(filepath, excluded_path, log_file, reason)
        return

    if len(data_lines) != nnz:
        reason = f"nnz 不匹配: expected={nnz}, actual={len(data_lines)}"
        move_file(filepath, excluded_path, log_file, reason)
        return

    # 检查每行是否至少包含两个字段（行列）
    for i, line in enumerate(data_lines):
        parts = line.strip().split()
        if len(parts) < 2:
            reason = f"第 {i+2} 行非法数据: 少于两个字段 -> {line.strip()}"
            move_file(filepath, excluded_path, log_file, reason)
            return

    # 清理数据行：仅保留前两个字段 + 设置值为 1
    cleaned_data = clean_data_lines(data_lines)
    with open(filepath, 'w') as f:
        f.writelines(comments)
        f.write(f"{m} {n} {nnz}\n")
        f.writelines(line + '\n' for line in cleaned_data)

    print(f"{filepath} -> 合法，已将所有数值设置为: row col 1")

def move_file(src, excluded_base, log_file, reason):
    rel_path = os.path.relpath(src, start=dataset_path)
    dst = os.path.join(excluded_base, rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.move(src, dst)
    print(f"{src} -> 已移动: {reason}")
    with open(log_file, 'a') as log:
        log.write(f"original: {src}\n")
        log.write(f"moved_to: {dst}\n")
        log.write(f"reason: {reason}\n")
        log.write("---\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python clean_dataset.py /path/to/dataset/")
        sys.exit(1)

    dataset_path = os.path.abspath(sys.argv[1]).rstrip('/')
    parent_path = os.path.dirname(dataset_path)
    dataset_name = os.path.basename(dataset_path)
    excluded_path = os.path.join(parent_path, "excluded_dataset", dataset_name)
    os.makedirs(excluded_path, exist_ok=True)
    log_file = os.path.join(excluded_path, "excluded_files.log")

    print(f"正在处理数据集路径: {dataset_path}")
    print(f"排除无效数据集的路径: {excluded_path}")

    # 调用 make_matrices_list.sh
    subprocess.run(["bash", "make_matrices_list.sh", dataset_path], check=True)

    list_file = os.path.join(dataset_path, "matrix_file_list_mtx.txt")
    if not os.path.isfile(list_file):
        print(f"找不到文件: {list_file}")
        sys.exit(1)

    with open(list_file, 'r') as f:
        files = [line.strip() for line in f if line.strip()]

    for rel_path in files:
        abs_path = os.path.join(dataset_path, rel_path)
        if not os.path.isfile(abs_path):
            print(f"文件不存在: {abs_path}")
            continue
        process_matrix_file(abs_path, excluded_path, log_file)
