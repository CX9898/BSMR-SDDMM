#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include "Matrix.hpp"
//#include "sddmm_isratnisa.h"
#include "kernel.cuh"
//#include "util_isratnisa.h"
#include "wmmaSetting.hpp"
#include "cudaErrorCheck.cuh"
#include "cudaUtil.cuh"
#include "CudaTimeCalculator.cuh"
#include "host.hpp"
#include "checkData.hpp"
#include "devVector.cuh"

const std::string folderPath("../dataset/");
//const std::string fileName = ("nips");
const std::string fileName = ("test");
//const std::string fileName = ("matrixTmp_8000_8000_2560000");
const std::string fileFormat(".mtx");
const std::string filePath = folderPath + fileName + fileFormat;

// TODO :
//      测试矩阵的尺寸按照论文中的尺寸
//      1: 将 comp_sddmm_gpu 全部使用 Tensor core 执行
//      2: 测试不同尺寸的 wmma 的速度表现
//      3: 测试使用稀疏度比较器的速度表现
//              稀疏度大于50%使用 israt 的方法
//              稀疏度小于50%使用 Tensor core 方法
int main() {
//    // make sparse matrix data
//    SparseMatrix<int> matrixTmp;
//    const int makeDataRow = 500 * WMMA_M;
//    const int makeDataCol = 500 * WMMA_N;
//    const float sparsity = 0.04f;
//    const int makeDataNNZ = (int) ((makeDataRow * makeDataCol) * sparsity);
////    const int makeDataNNZ = 746316;
//    matrixTmp.makeData(makeDataRow, makeDataCol, makeDataNNZ);
//    matrixTmp.outputToMarketMatrixFile("matrixTmp");

    SparseMatrix<float> matrixS(filePath);

//    const int K = 1 * WMMA_K;
    const int K = 32;
    const int M = matrixS.row();
    const int N = matrixS.col();
    const int MATRIX_A_SIZE = M * K;
    const int MATRIX_B_SIZE = K * N;

    const float matrixSSparsity = 1 / (M * N / (float) matrixS.nnz());
    std::cout << "M : " << M << ", N : " << N << ", K : " << K << ", nnz : " << matrixS.nnz() << ", sparsity : "
              << (1 - matrixSSparsity) * 100 << "%" << std::endl;

//    std::cout << "matrixS : " << std::endl;
//    matrixS.print();

    Matrix<float> matrixS2D(matrixS);

    Matrix<float> matrixA(M, K, MATRIX_A_SIZE, MatrixStorageOrder::row_major, K);
    matrixA.makeData(M, K, MatrixStorageOrder::row_major);
//    initial(matrixA.setValues(), M, K);
//    std::cout << "matrixA : ";
//    matrixA.print();

    Matrix<float> matrixB(K, N, MATRIX_B_SIZE, MatrixStorageOrder::row_major, N);
    matrixB.makeData(K, N, MatrixStorageOrder::row_major);
//    initial(matrixB.setValues(), N, K);
//    std::cout << "matrixB : ";
//    matrixB.print();

//    matrixA.changeStorageOrder();
//    matrixB.changeStorageOrder();

    SparseMatrix<float> matrixP_cpu_res(matrixS.row(), matrixS.col(), matrixS.nnz(),
                                        matrixS.rowIndex(), matrixS.colIndex());

    // comp by cpu
    sddmm_cpu_coo(matrixA, matrixB, matrixS, matrixP_cpu_res);

//    std::cout << "matrixP_cpu_res.values() : " << std::endl;
//    matrixP_cpu_res.print();

    dev::vector<float> valuesA_d(matrixA.size());
    dev::vector<half> valuesAfp16_d(matrixA.size());
    dev::vector<float> valuesB_d(matrixB.size());
    dev::vector<half> valuesBfp16_d(matrixB.size());
    dev::vector<float> valuesS_d(matrixS2D.size());
    dev::vector<float> valuesP_d(matrixS2D.size());

    cuUtil::H2D(valuesA_d.data(), matrixA.values().data(), matrixA.size());
    cuUtil::H2D(valuesB_d.data(), matrixB.values().data(), matrixA.size());

    const int numThreadPerBlock = 1024;
    convertFp32ToFp16<<< (matrixA.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
        matrixA.size(), valuesA_d.data(), valuesAfp16_d.data());
    convertFp32ToFp16<<< (matrixB.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
        matrixB.size(), valuesB_d.data(), valuesBfp16_d.data());

    dim3 grid;
    dim3 block;
    block.x = WARP_SIZE;
    block.y = WARP_SIZE;
    const int numCountRowOfOutputMatrixPerBlock = (int) (WMMA_M * block.x / 32);
    const int numCountColOfOutputMatrixPerBlock = (int) (WMMA_N * block.y);
    grid.x = (M + numCountRowOfOutputMatrixPerBlock - 1) / numCountRowOfOutputMatrixPerBlock;
    grid.y = (N + numCountColOfOutputMatrixPerBlock - 1) / numCountColOfOutputMatrixPerBlock;
//    printf("grid : [%d %d %d] block : [%d %d %d]\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();

    comp_sddmm_gpu<<<grid, block>>>(M, N, K,
                                    valuesAfp16_d.data(), valuesBfp16_d.data(), valuesS_d.data(), valuesP_d.data());

    timeCalculator.endClock();
    std::cout << "Func comp_sddmm_gpu time : " << timeCalculator.getTime() << " ms" << std::endl;
    std::cout << "sddmm_zcx time : " << timeCalculator.getTime() << " ms" << std::endl;

    Matrix<float> matrixP_gpu_res_tmp(M, N, M * N, MatrixStorageOrder::row_major, N);
    matrixP_gpu_res_tmp.initializeValue(cuUtil::D2H(valuesP_d.data(), matrixP_gpu_res_tmp.size()));

    SparseMatrix<float> matrixP_gpu_res(matrixS.row(), matrixS.col(), matrixS.nnz(),
                                        matrixS.rowIndex(), matrixS.colIndex());
    matrixP_gpu_res.setValuesFromMatrix(matrixP_gpu_res_tmp);

//    matrixP_gpu_res.outputToMarketMatrixFile("matrixP_gpu_res");

    checkData(matrixP_cpu_res.values(), matrixP_gpu_res.values());

//    dmm_cpu(matrixA,matrixB,matrixS2D);

//    std::cout << "matrixP_gpu_res : " << std::endl;
//    matrixP_gpu_res.print();

//    isratnisa::Matrix isratnisaMatrixS;
//    isratnisaMatrixS.copyFromMatrix(matrixS);

//    float *valuesP_isratnisa = nullptr;
//    preprocessing(isratnisaMatrixS, matrixA.values(), matrixB.values(), valuesP_isratnisa);

    return 0;
}