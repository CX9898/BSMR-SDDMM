import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import time

import torch
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.io import mmread
import argparse
import FS_SDDMM
import FS_Block_gpu


# tf32
class dataSet_tf32(torch.nn.Module):

    def __init__(self, K, partsize, data_path, window, wide):

        """
        Data loading for multiple graph formats (.npz and .mtx)
        """

        super(dataSet_tf32, self).__init__()

        ext = os.path.splitext(data_path)[1].lower()
        if ext == '.mtx':
            self.from_mtx = True
            self.graph = mmread(data_path).tocsr()  # Convert to CSR for efficient processing
        elif ext == '.npz':
            self.from_mtx = False
            self.graph = np.load(data_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        self.num_features = K
        self.init_edges(partsize, window, wide)
        self.init_embedding()

    def init_edges(self, partSize, window, wide):
        if self.from_mtx:
            adj = self.graph.tocsr()
            self.num_nodes = adj.shape[0]
            self.num_nodes_dst = adj.shape[1]
            self.num_edges = adj.nnz
            self.column_index = torch.IntTensor(adj.indices)
            self.row_pointers = torch.IntTensor(adj.indptr)
            self.values = torch.tensor(adj.data, dtype=torch.float32)

            coo = adj.tocoo()
            self.edge_index = np.stack([coo.row, coo.col])
        else:
            self.num_nodes = int(self.graph['num_nodes_src'])
            self.num_nodes_dst = int(self.graph['num_nodes_dst'])
            self.num_edges = int(self.graph['num_edges'])

            src_li = self.graph['src_li']
            dst_li = self.graph['dst_li']
            self.edge_index = np.stack([src_li, dst_li])

            val = [1] * self.num_edges
            scipy_coo = coo_matrix((val, self.edge_index), shape=(self.num_nodes, self.num_nodes_dst))
            adj = scipy_coo.tocsr()

            self.column_index = torch.IntTensor(adj.indices)
            self.row_pointers = torch.IntTensor(adj.indptr)
            self.values = torch.tensor(adj.data, dtype=torch.float32)

        self.row_pointers, \
            self.column_index, \
            self.degrees, \
            self.t_window_rowTensor, _, _ = FS_Block_gpu.preprocess_gpu_fs_balance(self.row_pointers, self.column_index,
                                                                                   self.num_nodes, self.num_edges,
                                                                                   window, wide, partSize)

        max_vectors = torch.max(self.row_pointers[1:] - self.row_pointers[:-1])
        if max_vectors % wide > 0:
            max_vectors += (wide - (max_vectors % wide))
        self.max = max_vectors / wide

        if self.max % 4 > 0:
            self.max += 4 - self.max % 4

    def init_embedding(self):
        '''
        Generate node embedding for nodes.
        Called from __init__.
        '''
        self.x = torch.randn(self.num_nodes_dst, self.num_features)
        self.x = self.x


# 8x1
def fs_tf32_16_1(epoches, K, partsize_t, data_path, window, wide):
    inputInfo = dataSet_tf32(K, partsize_t, data_path, window, wide)

    X_prime, sddmm_ms_avg = FS_SDDMM.forward_gen_tf32_16(
        inputInfo.x.size(1),
        inputInfo.row_pointers,
        inputInfo.column_index,
        inputInfo.degrees.int(),
        inputInfo.t_window_rowTensor,
        inputInfo.x,
        epoches, inputInfo.max)
    sddmm_ms_avg = round((sddmm_ms_avg.item()), 4)
    # print(str(K) + '-' + data_path + 'tcu_16_1' + '-' + str(sddmm_ms_avg))
    return sddmm_ms_avg, inputInfo.num_edges


# 8x1
def fs_tf32_8_1(epoches, dimN, partsize_t, data_path, window, wide):
    inputInfo = dataSet_tf32(dimN, partsize_t, data_path, window, wide)

    X_prime, sddmm_ms_avg = FS_SDDMM.forward_gen_tf32(
        inputInfo.x.size(1),
        inputInfo.row_pointers,
        inputInfo.column_index,
        inputInfo.degrees.int(),
        inputInfo.t_window_rowTensor,
        inputInfo.x,
        epoches, inputInfo.max)
    sddmm_ms_avg = round((sddmm_ms_avg.item()), 4)
    # print(str(dimN) + '-' + data_path + 'tcu_8_1' + '-' + str(sddmm_ms_avg))
    return sddmm_ms_avg, inputInfo.num_edges


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="FlashSparse SDDMM Test Script")
    parser.add_argument('--matrix_list', type=str, help="矩阵文件列表路径")
    parser.add_argument('-K', type=int, default=32, help="矩阵文件列表路径")
    parser.add_argument('--log_file', type=str, default='FlashSparse', help="日志文件")
    args = parser.parse_args()

    gpu_device = torch.cuda.current_device()
    gpu = torch.cuda.get_device_name(gpu_device)
    print(gpu)

    matrix_lists_file = args.matrix_list
    matrix_dir_path = os.path.dirname(matrix_lists_file) + '/'
    K = args.K

    print("Matrix lists file: " + matrix_lists_file)

    log_file = args.log_file + '.log'
    print("Log file: " + log_file)

    print('K: ' + str(K))

    epoches = 10
    partsize_t = 32

    start_time = time.time()

    with open(matrix_lists_file, 'r') as f:
        line_count = sum(1 for _ in f)
        with open(log_file, 'w', newline='') as write_file:
            write_file.write('[numTestFiles : ' + str(line_count) + ' ]\n')

    with open(matrix_lists_file, 'r', encoding='utf-8') as f:
        for line in f:
            file_path = matrix_dir_path + line.strip()
            # file_name = os.path.splitext(file_path.split('/')[-1])[0]
            print('Loading file: ' + file_path)
            print('K: ' + str(K))

            # 16x1
            sddmm_tcu_16_1, nnz = fs_tf32_16_1(epoches, K, partsize_t, file_path, 16, 8)

            # 8x1
            sddmm_tcu_8_1, nnz = fs_tf32_8_1(epoches, K, partsize_t, file_path, 8, 16)

            gflops = (nnz * K * 2) / (sddmm_tcu_8_1 * 1e6)

            with open(log_file, 'a', newline='') as f:
                f.write('---New data---\n')
                f.write('[File : ' + file_path + ']\n')
                f.write('[K : ' + str(K) + ' ]\n')
                f.write('[FlashSparse_gflops : ' + str(gflops) + ' ]\n')

            print('success')
            print()

    print('All is success')

    end_time = time.time()
    execution_time = end_time - start_time

    # Record execution time.
    with open(args.outdir + "/execution_time_base.txt", "a") as file_path:
        file_path.write("Baseline-FlashSparse-" + str(K) + "-" + str(round(execution_time / 60, 2)) + " minutes\n")
