#include <iostream>
#include <string>

#include "Matrix.hpp"
#include "sddmm.h"
#include "kernel.cuh"
#include "util.h"
#include "wmmaSetting.hpp"
#include "cudaErrorCheck.cuh"
#include "cudaUtil.cuh"
#include "CudaTimeCalculator.cuh"
#include "host.hpp"

const std::string folderPath("../dataset/");
const std::string fileName = ("nips.mtx");
//const std::string fileName = ("test1.mtx");
const std::string filePath = folderPath + fileName;

int main() {
    SparseMatrix<float> matrixS;
    matrixS.initializeFromMatrixMarketFile(filePath);

    const int K = 256;
    const int M = matrixS.row();
    const int N = matrixS.col();
    const int MATRIX_A_SIZE = M * K;
    const int MATRIX_B_SIZE = K * N;

//    std::cout << "matrixS : " << std::endl;
//    matrixS.printfValue();

    Matrix<float> matrixS2D;
    matrixS2D.initializeFromSparseMatrix(matrixS);
//    matrixS2D.changeStorageOrder();

    Matrix<float> matrixA(M, K, MATRIX_A_SIZE, MatrixStorageOrder::row_major, K);
    initial(matrixA.setValues(), M, K);
    Matrix<float> matrixB(K, N, MATRIX_B_SIZE, MatrixStorageOrder::row_major, N);
    initial(matrixB.setValues(), N, K);
//    matrixA.changeStorageOrder();
    matrixB.changeStorageOrder();

    SparseMatrix<float> matrixP_cpu_res(matrixS.row(), matrixS.col(), matrixS.nnz(),
                                        matrixS.rowIndex(), matrixS.colIndex());
    sddmm_cpu_coo(matrixA, matrixB, matrixS, matrixP_cpu_res);
//    std::cout << "matrixP.values() : " << std::endl;
//    matrixP_host.printfValue();

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
    block.x = 128;
    block.y = 4;
    grid.x = (matrixS2D.row() + (WMMA_M * block.x / 32 - 1)) / (WMMA_M * block.x / 32);
    grid.y = (matrixS2D.col() + WMMA_N * block.y - 1) / (WMMA_N * block.y);

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();
    comp_sddmm_gpu<<<grid, block>>>(M, N, K, valuesAfp16_d, valuesBfp16_d, valuesS_d, valuesP_d);
    timeCalculator.endClock();
    std::cout << "Func compSddmm time : " << timeCalculator.getTime() << "ms" << std::endl;

    SparseMatrix<float> matrixP_gpu_res(matrixS.row(), matrixS.col(), matrixS.nnz(),
                                        matrixS.rowIndex(), matrixS.colIndex());

    dev::D2H(matrixP_gpu_res.setValues().data(), valuesP_d, matrixP_gpu_res.values().size());


    std::cout << "matrixP_gpu_res : " << std::endl;
    matrixP_gpu_res.printfValue();

    isratnisa::Matrix isratnisaMatrixS;
    isratnisaMatrixS.copyFromMatrix(matrixS);

    float *valuesP_isratnisa = nullptr;
    preprocessing(isratnisaMatrixS, matrixA.values(), matrixB.values(), valuesP_isratnisa);

    return 0;
}