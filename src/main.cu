#include <iostream>
#include <string>

#include "Matrix.hpp"
#include "TensorCoreConfig.cuh"
#include "util.hpp"
#include "cuSparseSDDMM.hpp"
#include "sddmm.hpp"

const std::string folderPath("../dataset/test/matrix_20000_20000_/");
//const std::string folderPath("./");
//const std::string fileName = ("nips");
//const std::string fileName = ("test2");
const std::string fileName("matrix_20000_20000_4000000");
const std::string fileFormat(".mtx");
const std::string filePath = folderPath + fileName + fileFormat;

// TODO :
//      测试矩阵的尺寸按照论文中的尺寸
//      1: 将 comp_sddmm_gpu 全部使用 Tensor core 执行          OK
//      2: 测试不同尺寸的 wmma 的速度表现                         OK
//      3: 测试使用稀疏度比较器的速度表现
//              稀疏度大于50%使用 isratnisa 的方法
//              稀疏度小于50%使用 Tensor core 方法
//      4: 全部数据放在device内存                                OK
//      5: 优化openTensorCoreModeForSampled()                  OK
//      6: 测试更大的K(<5k)的结果                                OK
//      7: 优化positionCalculator(),                           OK
//                  支持 WMMA 维度 : 16×16×16                   OK
//                  支持 WMMA 维度 : 32×8×16                    OK
//                  支持 WMMA 维度 : 8×32×16                    OK
//      8: sddmm函数中支持各种矩阵储存维度
//                    matrixA: row_major matrixB: row_major    OK
//                    matrixA: row_major matrixB: col_major    OK
//                    matrixA: col_major matrixB: row_major
//                    matrixA: col_major matrixB: col_major
//      9: TensorCoreConfig 支持 WarpOrder
//      10: 在sddmm函数中使用共享内存                             OK
//      11: 试验新方法(行列重排序)
//                   1) 代码实现
//                   2) 测试
//                   3) 比较数据
//      12: 核函数中, 将K迭代放在调用核函数的外部. 增加矩阵A的数据重用性. 但是写回全局内存的次数将会增加. 具体效果还需要测试

//#define MAKE_MATRIX_DATA

int main(int argc, char *argv[]) {

#ifdef MAKE_MATRIX_DATA
    // make sparse matrix data
    {
        SparseMatrix<int> matrixTmp;
        const size_t thousand = 1000;
        const size_t million = 1000000;
//    const size_t makeDataRow = 3 * thousand;
//    const size_t makeDataCol = 7 * thousand;
        const size_t makeDataRow = 10000;
        const size_t makeDataCol = 15000;
//    const float density = 4.006f;
//    const size_t makeDataNNZ = static_cast<int> (makeDataRow * makeDataCol * density / 100);
//    const float sparsity = 0.80;
//    const size_t makeDataNNZ = makeDataRow * makeDataCol * (1 - sparsity);
//    const size_t makeDataNNZ = 1 * million;
        const size_t makeDataNNZ = 7500000;
        matrixTmp.makeData(makeDataRow, makeDataCol, makeDataNNZ);
        matrixTmp.outputToMarketMatrixFile();
        std::cout << "makeData : M : " << makeDataRow
                  << ", N : " << makeDataCol
                  << ", K : " << 256
                  << ", nnz : " << makeDataNNZ
                  << ", sparsity : "
                  << (float) (makeDataRow * makeDataCol - makeDataNNZ) / (makeDataRow * makeDataCol) * 100 << "%"
                  << std::endl;
        exit(0);
    }
#endif // MAKE_MATRIX_DATA

    size_t K = 16;
    sparseMatrix::COO<float> matrixS;

    if (argc > 2) {
        if (!matrixS.initializeFromMatrixMarketFile(argv[1])) {
            exit(1);
        }
        K = std::stol(argv[2]);
    } else if (argc == 2) {
        if (!matrixS.initializeFromMatrixMarketFile(argv[1])) {
            exit(1);
        }
    } else {
        if (!matrixS.initializeFromMatrixMarketFile(util::getParentFolderPath(argv[0]) + filePath)) {
            exit(1);
        }
    }

#ifdef NDEBUG
    printf("[Build type : Release]\n");
#endif

#ifndef NDEBUG
    printf("[Build type : Debug]\n");
#endif

    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("[Device : %s]\n", deviceProp.name);

    printf("[M : %d], ", matrixS.row());
    printf("[N : %d], ", matrixS.col());
    printf("[K : %ld], ", K);
    printf("[NNZ : %d], ", matrixS.nnz());
    printf("[sparsity : %.2f%%]\n", matrixS.getSparsity() * 100);

    printf("[matrixA type : %s]\n", typeid(MATRIX_A_TYPE).name());
    printf("[matrixB type : %s]\n", typeid(MATRIX_B_TYPE).name());
    printf("[matrixC type : %s]\n", typeid(MATRIX_C_TYPE).name());

    Matrix<float> matrixA(matrixS.row(), K, MatrixStorageOrder::row_major);
    matrixA.makeData(matrixA.row(), K);

    Matrix<float> matrixB(K, matrixS.col(), MatrixStorageOrder::row_major);
    matrixB.makeData(K, matrixS.col());

//    matrixA.changeStorageOrder();
//    matrixB.changeStorageOrder();

    if (matrixA.storageOrder() == MatrixStorageOrder::row_major) { printf("[matrixA storageOrder : row_major]\n"); }
    else { printf("[matrixA storageOrder : col_major]\n"); }
    if (matrixB.storageOrder() == MatrixStorageOrder::row_major) { printf("[matrixB storageOrder : row_major]\n"); }
    else { printf("[matrixB storageOrder : col_major]\n"); }

    const sparseMatrix::CSR<float> matrixS_csr(matrixS.getCsrData());

    // cuSparse library
    sparseMatrix::CSR<float> matrixP_cuSparse(matrixS_csr);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cuSparseSDDMM(matrixA, matrixB, matrixS_csr, alpha, beta, matrixP_cuSparse);

    // sddmm
    sparseMatrix::CSR<float> matrixP_csr(matrixS_csr);
    sddmm(matrixA, matrixB, matrixS_csr, matrixP_csr);

    // Error check
    printf("check cuSparseSDDMM and sddmm : \n");
    size_t numError = 0;
    if (!checkData(matrixP_cuSparse.values(), matrixP_csr.values(), numError)) {
        printf("[checkData : NO PASS Error rate : %2.2f%%]\n",
               static_cast<float>(numError) / static_cast<float>(matrixP_csr.values().size()) * 100);
        return -1;
    }

    return 0;
}