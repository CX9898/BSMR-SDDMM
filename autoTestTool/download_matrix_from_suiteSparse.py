import argparse
import ssgetpy
import tarfile


def extract_tar_gz(file_path, output_dir):
    """解压 .tar.gz 文件到指定目录"""
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=output_dir)
            print(f"解压完成, 文件已存放到 {output_dir}")
    except (tarfile.TarError, EOFError):
        print(f"文件损坏: {file_path}")


parser = argparse.ArgumentParser(description="从SuiteSparse_Matrix_Collection下载矩阵脚本")

parser.add_argument("-num", type=int, default=10, help="矩阵数量")
parser.add_argument("-row_min", type=int, default=1000, help="矩阵数量")
parser.add_argument("-row_max", type=int, default=10000, help="矩阵数量")
parser.add_argument("-col_min", type=int, default=1000, help="矩阵数量")
parser.add_argument("-col_max", type=int, default=10000, help="矩阵数量")
# parser.add_argument("-nnz_min", type=int, default=1000, help="矩阵数量")
# parser.add_argument("-nnz_max", type=int, default=10000, help="矩阵数量")

args = parser.parse_args()

numMatrix = args.num
row_min = args.row_min
row_max = args.row_max
col_min = args.col_min
col_max = args.col_max

result = ssgetpy.search(rowbounds=(row_min, row_max),
                        colbounds=(col_min, col_max),
                        limit=numMatrix)

print("Download " + str(numMatrix) + " matrices from suite sparse matrix collection")

output_path = "../dataset/ssgetpy/"
matrix_file_list = []
i = 0
for matrix in result[:numMatrix]:
    i += 1
    remaining = numMatrix - i  # 计算剩余次数
    print(f"下载矩阵: {matrix.name}" + f" remaining: {remaining}")
    matrix.download(format="MM", destpath=output_path)
    tar_gz_file = output_path + matrix.name + ".tar.gz"
    print("开始解压: " + tar_gz_file)
    extract_tar_gz(tar_gz_file, output_path)
    matrix_file = output_path + matrix.name + "/" + matrix.name + ".mtx"
    matrix_file_list.append(matrix_file)

matrix_file_list_file = output_path + "matrix_file_list.txt"
with open(matrix_file_list_file, "w") as file:
    for matrix_file in matrix_file_list:
        file.write(matrix_file + "\n")
    print("矩阵文件列表: " + matrix_file_list_file)
