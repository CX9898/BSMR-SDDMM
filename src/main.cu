#include <iostream>
#include <string>

#include "Matrix.hpp"
#include "TensorCoreConfig.cuh"
#include "util.hpp"
#include "cuSparseSDDMM.hpp"
#include "sddmm.hpp"
#include "Logger.hpp"
#include "Options.hpp"

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
//      11: 试验新方法(行列重排序)                                OK
//                   1) 代码实现                                OK
//                   2) 测试                                   OK
//                   3) 比较数据                                OK
//      12: 核函数中, 将K迭代放在调用核函数的外部. 增加矩阵A的数据重用性. 但是写回全局内存的次数将会增加. 具体效果还需要测试  OK
//      13: 1) 增加线程块大小
//          2) 测试更小稀疏度的矩阵
//          3) 测试论文中的数据集, 因为随机的数据集会让数据均匀分布, 重排序的作用不明显
//          4) 测试一个warp计算两个block的数据, 创建两个cFragment

int main(int argc, char *argv[]) {

    Options options(argc, argv);

    const size_t K = options.K();
    const float alpha = options.alpha(), beta = options.beta();

    sparseMatrix::COO<float> matrixS;
    matrixS.initializeFromMatrixMarketFile(options.inputFile());

    Logger logger;
    logger.getInformation(matrixS);

    Matrix<float> matrixA(matrixS.row(), K, MatrixStorageOrder::row_major);
    matrixA.makeData();

    Matrix<float> matrixB(K, matrixS.col(), MatrixStorageOrder::col_major);
    matrixB.makeData();

    logger.getInformation(matrixA, matrixB);

    const sparseMatrix::CSR<float> matrixS_csr(matrixS.getCsrData());

    // cuSparse library
    sparseMatrix::CSR<float> matrixP_cuSparse(matrixS_csr);
    cuSparseSDDMM(matrixA, matrixB, matrixS_csr, alpha, beta, matrixP_cuSparse, logger);

//    // old method
//    sparseMatrix::COO<float> matrixP_oldMethod;
//    sddmm(matrixA, matrixB, matrixS, matrixP_oldMethod);
//    // Error check
//    printf("check cuSparseSDDMM and sddmm : \n");
//    size_t numError_old = 0;
//    if (!checkData(matrixP_cuSparse.values(), matrixP_oldMethod.values(), numError_old)) {
//        printf("[checkData : NO PASS Error rate : %2.2f%%]\n",
//               static_cast<float>(numError_old) / static_cast<float>(matrixP_oldMethod.values().size()) * 100);
//    }

    // sddmm
    sparseMatrix::CSR<float> matrixP(matrixS_csr);
    sddmm(matrixA, matrixB, alpha, beta, matrixS_csr, matrixP, logger);

    // Error check
    printf("check cuSparseSDDMM and sddmm : \n");
    size_t numError = 0;
    if (!checkData(matrixP_cuSparse.values(), matrixP.values(), numError)) {
        printf("[checkData : NO PASS Error rate : %2.2f%%]\n",
               static_cast<float>(numError) / static_cast<float>(matrixP.values().size()) * 100);
    }
    logger.numError_ = numError;

    logger.printLogInformation();

    return 0;
}