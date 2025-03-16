import argparse
import ssgetpy
import tarfile

def extract_tar_gz(file_path, output_dir):
    """解压 .tar.gz 文件到指定目录"""
    with tarfile.open(file_path, "r:gz") as tar:
        tar.extractall(path=output_dir)
        print(f"解压完成, 文件已存放到 {output_dir}")

parser = argparse.ArgumentParser(description="从SuiteSparse_Matrix_Collection下载矩阵脚本")

parser.add_argument("-n", type=int, help="矩阵数量")

args = parser.parse_args()

numMatrix = args.n

result = ssgetpy.search(limit=numMatrix)

output_path = "../dataset/ssgetpy/"
matrix_file_list = []
for matrix in result[:numMatrix]:
    print(f"下载矩阵: {matrix.name}")
    matrix.download(format="MM", destpath=output_path)
    tar_gz_file = output_path + matrix.name + ".tar.gz"
    extract_tar_gz(tar_gz_file, output_path)
    matrix_file = output_path + matrix.name + "/" + matrix.name + ".mtx"
    matrix_file_list.append(matrix_file)

matrix_file_list_file = output_path + "matrix_file_list.txt"
with open(matrix_file_list_file, "w") as file:
    for matrix_file in matrix_file_list:
        file.write(matrix_file + "\n")
    print("矩阵文件列表: " + matrix_file_list_file)
