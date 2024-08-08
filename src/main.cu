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

const std::string folderPath("../dataset/");
//const std::string fileName = ("nips");
//const std::string fileName = ("nips_wmma");
const std::string fileName = ("test3");
const std::string fileFormat(".mtx");
const std::string filePath = folderPath + fileName + fileFormat;

int main() {
    SparseMatrix<float> matrixS;
    matrixS.makeData(1504, 1504, 746316);
    matrixS.outputToMarketMatrixFile("matrixS");
//    matrixS.initializeFromMatrixMarketFile(filePath);

    const int K = 16 * WMMA_K;
//    const int K = 17;
    const int M = matrixS.row();
    const int N = matrixS.col();
    const int MATRIX_A_SIZE = M * K;
    const int MATRIX_B_SIZE = K * N;

    std::cout << "M : " << M << ", N : " << N << ", K : " << K << std::endl;

//    std::cout << "matrixS : " << std::endl;
//    matrixS.print();

    Matrix<float> matrixS2D;
    matrixS2D.initializeFromSparseMatrix(matrixS);

    Matrix<float> matrixA(M, K, MATRIX_A_SIZE, MatrixStorageOrder::row_major, K);
    matrixA.makeData(M, K);
//    initial(matrixA.setValues(), M, K);
//    std::cout << "matrixA : ";
//    matrixA.print();

    Matrix<float> matrixB(K, N, MATRIX_B_SIZE, MatrixStorageOrder::row_major, N);
    matrixB.makeData(K, N);
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

    float *valuesA_d;
    half *valuesAfp16_d;
    float *valuesB_d;
    half *valuesBfp16_d;
    float *valuesS_d;
    float *valuesP_d;

    cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&valuesA_d), matrixA.size() * sizeof(float)));
    cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&valuesAfp16_d), matrixA.size() * sizeof(half)));
    cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&valuesB_d), matrixB.size() * sizeof(float)));
    cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&valuesBfp16_d), matrixB.size() * sizeof(half)));
    cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&valuesS_d), matrixS2D.size() * sizeof(float)));
    cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&valuesP_d), matrixS2D.size() * sizeof(float)));

    dev::H2D(valuesA_d, matrixA.values().data(), matrixA.size());
    dev::H2D(valuesB_d, matrixB.values().data(), matrixA.size());

    const int numThreadPerBlock = 1024;
    convertFp32ToFp16<<< (matrixA.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
        matrixA.size(), valuesA_d, valuesAfp16_d);
    convertFp32ToFp16<<< (matrixB.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
        matrixB.size(), valuesB_d, valuesBfp16_d);

    dim3 grid;
    dim3 block;
    block.x = WARP_SIZE;
    block.y = WARP_SIZE;
    const int numCountRowOfOutputMatrixPerBlock = (int) (WMMA_M * block.x / 32);
    const int numCountColOfOutputMatrixPerBlock = (int) (WMMA_N * block.y);
    grid.x = (M + numCountRowOfOutputMatrixPerBlock - 1) / numCountRowOfOutputMatrixPerBlock;
    grid.y = (N + numCountColOfOutputMatrixPerBlock - 1) / numCountColOfOutputMatrixPerBlock;
    printf("grid : [%d %d %d] block : [%d %d %d]\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();

    comp_sddmm_gpu<<<grid, block>>>(M, N, K, valuesAfp16_d, valuesBfp16_d, valuesS_d, valuesP_d);

    timeCalculator.endClock();
    std::cout << "Func comp_sddmm_gpu time : " << timeCalculator.getTime() << "ms" << std::endl;

    Matrix<float> matrixP_gpu_res_tmp(M, N, M * N, MatrixStorageOrder::row_major, N);
    dev::D2H(matrixP_gpu_res_tmp.setValues().data(), valuesP_d, matrixP_gpu_res_tmp.size());

    SparseMatrix<float> matrixP_gpu_res(matrixS.row(), matrixS.col(), matrixS.nnz(),
                                        matrixS.rowIndex(), matrixS.colIndex());
    matrixP_gpu_res.setValuesFromMatrix(matrixP_gpu_res_tmp);

//    matrixP_gpu_res.outputToMarketMatrixFile("matrixP_gpu_res");

    checkData(matrixP_cpu_res.values(), matrixP_gpu_res.values());

//    std::cout << "matrixP_gpu_res : " << std::endl;
//    matrixP_gpu_res.print();

//    isratnisa::Matrix isratnisaMatrixS;
//    isratnisaMatrixS.copyFromMatrix(matrixS);

//    float *valuesP_isratnisa = nullptr;
//    preprocessing(isratnisaMatrixS, matrixA.values(), matrixB.values(), valuesP_isratnisa);

    cudaFree(valuesA_d);
    cudaFree(valuesAfp16_d);
    cudaFree(valuesB_d);
    cudaFree(valuesBfp16_d);
    cudaFree(valuesS_d);
    cudaFree(valuesP_d);

    return 0;
}