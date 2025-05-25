import argparse
import ssgetpy
import tarfile
import os
import time
import subprocess
import random

def extract_tar_gz(file_path, output_dir):
    """解压 .tar.gz 文件到指定目录"""
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=output_dir)
            print(f"解压完成, 文件已存放到 {output_dir}")
        os.remove(file_path)
        print(f"已删除压缩包: {file_path}")
    except (tarfile.TarError, EOFError):
        print(f"文件损坏: {file_path}")

def wget_download(matrix, output_dir):
    """使用 wget 下载矩阵 .tar.gz 文件（使用 group+name 构造 URL）"""
    url = f"https://sparse.tamu.edu/MM/{matrix.group}/{matrix.name}.tar.gz"
    filename = os.path.join(output_dir, matrix.name + ".tar.gz")

    try:
        subprocess.run(["wget", "--show-progress", "-O", filename, url], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"wget 下载失败: {matrix.name} | {e}")
        return False

# 解析命令行参数
parser = argparse.ArgumentParser(description="从SuiteSparse_Matrix_Collection下载矩阵脚本")

parser.add_argument("-num", type=int, default=10, help="矩阵数量")
parser.add_argument("-row_min", type=int, default=1000, help="最小行数")
parser.add_argument("-row_max", type=int, default=10000, help="最大行数")
parser.add_argument("-col_min", type=int, default=1000, help="最小列数")
parser.add_argument("-col_max", type=int, default=10000, help="最大列数")
parser.add_argument("-outdir", type=str, default="dataset/ssgetpy/", help="输出目录")

args = parser.parse_args()

numMatrix = args.num
row_min = args.row_min
row_max = args.row_max
col_min = args.col_min
col_max = args.col_max
output_path = args.outdir

# 创建输出目录（如果不存在）
os.makedirs(output_path, exist_ok=True)

# 搜索满足条件的矩阵
result = ssgetpy.search(rowbounds=(row_min, row_max),
                        colbounds=(col_min, col_max),
                        limit=numMatrix)

print(f"准备下载 {numMatrix} 个矩阵...")

matrix_file_list = []
failed_matrix_list = []

# 依次下载每个矩阵
for i, matrix in enumerate(result[:numMatrix]):
    remaining = numMatrix - i - 1
    print(f"\n下载矩阵: {matrix.name} | 剩余: {remaining}")

    # 每次下载之间随机 sleep，减少被限速风险
    time.sleep(random.uniform(1.5, 3.5))

    success = False
    for attempt in range(3):
        success = wget_download(matrix, output_path)
        if success:
            break
        print(f"[{matrix.name}] 第 {attempt + 1} 次 wget 下载失败，重试中...")
        time.sleep(2)

    if not success:
        print(f"[{matrix.name}] 最终 wget 下载失败，跳过")
        failed_matrix_list.append(matrix.name)
        continue

    # 解压下载的文件
    tar_gz_file = os.path.join(output_path, matrix.name + ".tar.gz")
    print("开始解压: " + tar_gz_file)
    extract_tar_gz(tar_gz_file, output_path)

    matrix_file = os.path.join(output_path, matrix.name, matrix.name + ".mtx")
    matrix_file_list.append(matrix_file)

# 写入失败矩阵列表
if failed_matrix_list:
    fail_list_path = os.path.join(output_path, "failed_downloads.txt")
    with open(fail_list_path, "w") as f:
        for name in failed_matrix_list:
            f.write(name + "\n")
    print(f"\n下载失败的矩阵已记录: {fail_list_path}")
else:
    print("\n所有矩阵均成功下载 ✅")
