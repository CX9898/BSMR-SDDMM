import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time

import torch
import numpy as np
import TCGNN
# import TCGNN_kernel
from scipy.sparse import coo_matrix, csr_matrix
from scipy.io import mmread
import argparse
from pathlib import Path

BLK_H = 16
BLK_W = 8
WARP_SIZE = 32


def func(x):
    if x > 0:
        return x
    else:
        return 1


def is_symmetric(sparse_matrix):
    transposed_matrix = sparse_matrix.transpose(copy=True)
    return (sparse_matrix != transposed_matrix).nnz == 0


class MGCN_dataset(torch.nn.Module):
    """
    Data loading for multiple graph formats (.npz and .mtx)
    """

    def __init__(self, data_path, from_mtx=False):
        super(MGCN_dataset, self).__init__()

        ext = os.path.splitext(data_path)[1].lower()
        if ext == '.mtx':
            self.from_mtx = True
            self.graph = mmread(data_path).tocsr()  # Convert to CSR for efficient processing
        elif ext == '.npz':
            self.from_mtx = False
            self.graph = np.load(data_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        self.init_edges()
        self.init_tcgnn()

    def init_edges(self):
        if self.from_mtx:
            adj = self.graph
            self.num_nodes = adj.shape[0]
            self.num_nodes_dst = adj.shape[1]
            self.num_edges = adj.nnz
            self.column_index = torch.IntTensor(adj.indices)
            self.row_pointers = torch.IntTensor(adj.indptr)
            self.values = torch.tensor(adj.data, dtype=torch.float32)
        else:
            self.num_nodes = int(self.graph['num_nodes_src'])
            self.num_nodes_dst = int(self.graph['num_nodes_dst'])
            self.num_edges = int(self.graph['num_edges'])

            src_li = self.graph['src_li']
            dst_li = self.graph['dst_li']
            self.edge_index = np.stack([src_li, dst_li])

            val = [1] * self.num_edges
            scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes))
            adj = scipy_coo.tocsr()

            self.column_index = torch.IntTensor(adj.indices)
            self.row_pointers = torch.IntTensor(adj.indptr)
            self.values = torch.tensor(adj.data, dtype=torch.float32)

    def init_tcgnn(self):
        self.num_row_windows = (self.num_nodes + 16 - 1) // BLK_H
        self.edgeToColumn = torch.zeros(self.num_edges, dtype=torch.int)
        self.edgeToRow = torch.zeros(self.num_edges, dtype=torch.int)
        self.blockPartition = torch.zeros(self.num_row_windows, dtype=torch.int)

        TCGNN.preprocess(self.column_index, self.row_pointers, self.num_nodes,
                         16, 8, self.blockPartition, self.edgeToColumn, self.edgeToRow)

    def init_embedding(self, dimN):
        self.x1 = torch.randn(self.num_nodes, dimN)
        self.x = self.x1.cuda()

    def to(self, device):
        self.column_index = self.column_index.cuda()
        self.row_pointers = self.row_pointers.cuda()
        self.values = self.values.cuda()
        self.blockPartition = self.blockPartition.cuda()
        self.edgeToColumn = self.edgeToColumn.cuda()
        self.edgeToRow = self.edgeToRow.cuda()
        return self

    def get_values_numel(self):
        return self.values.numel()


def kernel(inputInfo, epoches):
    X_prime, sddmm_ms_avg = TCGNN.forward_ef(inputInfo.x, inputInfo.row_pointers, inputInfo.column_index,
                                             inputInfo.blockPartition, inputInfo.edgeToColumn,
                                             inputInfo.edgeToRow, epoches)
    return round(sddmm_ms_avg.item(), 4)


'''
TCGNN
'''


def tcgnn_test(dimN, epoches, data_path):
    # 记录程序开始时间
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputInfo = MGCN_dataset(data_path)
    inputInfo.init_embedding(dimN)
    inputInfo = inputInfo.to(device)

    execution_time = kernel(inputInfo, epoches)
    print(str(execution_time) + ' ms')
    return execution_time, inputInfo.num_edges


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="TCGNN SDDMM Test Script")
    parser.add_argument('--matrix_list', type=str, help="矩阵文件列表路径")
    parser.add_argument('-K', type=int, default=32, help="矩阵文件列表路径")
    parser.add_argument('--log_file', type=str, default='TCGNN', help="日志文件")
    args = parser.parse_args()

    gpu_device = torch.cuda.current_device()
    gpu = torch.cuda.get_device_name(gpu_device)
    print(gpu)

    matrix_lists_file = args.matrix_list
    matrix_dir_path = os.path.dirname(matrix_lists_file) + '/'
    K = args.K
    print(matrix_lists_file)
    print(matrix_dir_path)

    log_file = args.log_file + '.log'
    print("Log file: " + log_file)

    print('K: ' + str(K))
    epoches = 10

    start_time = time.time()

    with open(matrix_lists_file, 'r') as f_log:
        line_count = sum(1 for _ in f_log)
        with open(log_file, 'w', newline='') as write_file:
            write_file.write('[numTestFiles : ' + str(line_count) + ' ]\n')

    with open(matrix_lists_file, 'r', encoding='utf-8') as f_log:
        lines = [line.strip() for line in f_log if line.strip()]
        total_lines = len(lines)

        for idx, line in enumerate(lines):
            remaining = total_lines - idx - 1
            print(f'TCGNN SDDMM Test: [Remaining: {remaining} ]')

            file_path = matrix_dir_path + line
            print('Loading file: ' + file_path)
            print('K: ' + str(K))

            # tcgnn
            sdmm, nnz = tcgnn_test(K, epoches, file_path)

            gflops = (nnz * K * 2) / (sdmm * 1e6)

            with open(log_file, 'a', newline='') as f_log:
                f_log.write('---New data---\n')
                f_log.write(f'[Remaining: {remaining} ]\n')
                f_log.write('[File : ' + file_path + ']\n')
                f_log.write('[K : ' + str(K) + ' ]\n')
                f_log.write('[TCGNN_gflops : ' + str(gflops) + ' ]\n')

            print('success\n')

    with open(log_file, 'a', newline='') as write_file:
        write_file.write('\n---Test done---\n')

    print('All is success')

    end_time = time.time()
    execution_time = end_time - start_time

    # Record execution time.
    with open(log_file, 'a', newline='') as write_file:
        write_file.write(str(round(execution_time / 60, 2)) + " minutes\n")
