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
//const std::string fileName = ("nips.mtx");
const std::string fileName = ("test1.mtx");
const std::string filePath = folderPath + fileName;

int main() {
    const int K = 256;

    SparseMatrix<float> matrixS;
    matrixS.initializeFromMatrixMarketFile(filePath);

    std::cout << "matrixS : " << std::endl;
    matrixS.printfValue();

    Matrix<float> matrixS2D;
    matrixS2D.initializeFromSparseMatrix(matrixS);
    matrixS2D.changeMajorOrder();

    std::vector<float> valuesA(matrixS2D.row() * K);
    initial(valuesA, matrixS2D.row(), K);
    std::vector<float> valuesB(K * matrixS2D.col());
    initial(valuesB, matrixS2D.col(), K);

    Matrix<float> matrixA(matrixS2D.row(), K, matrixS2D.row() * K, MatrixOrder::row_major, K, valuesA);
    Matrix<float> matrixB(K, matrixS2D.col(), K * matrixS2D.col(), MatrixOrder::row_major, matrixS2D.col(), valuesB);

    SparseMatrix<float> matrixP0
        (matrixS.row(), matrixS.col(), matrixS.nnz(), matrixS.rowIndex(), matrixS.colIndex());
    sddmm_cpu_coo(matrixA, matrixB, matrixS, matrixP0);
    std::cout << "matrixP0.values() : " << std::endl;
    matrixP0.printfValue();

    SparseMatrix<float> matrixP
        (matrixS.row(), matrixS.col(), matrixS.nnz(), matrixS.rowIndex(), matrixS.colIndex());
    sddmm_cpu_coo2(matrixA, matrixB, matrixS, matrixP);
    std::cout << "matrixP.values() : " << std::endl;
    matrixP.printfValue();

    float *valuesA_d;
    half *valuesAfp16_d;
    float *valuesB_d;
    half *valuesBfp16_d;
    float *valuesS_d;
    float *valuesP_d;

    cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&valuesA_d), valuesA.size() * sizeof(float)));
    cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&valuesAfp16_d), valuesA.size() * sizeof(half)));
    cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&valuesB_d), valuesB.size() * sizeof(float)));
    cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&valuesBfp16_d), valuesB.size() * sizeof(half)));
    cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&valuesS_d), matrixS2D.size() * sizeof(float)));
    cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&valuesP_d), matrixS2D.size() * sizeof(float)));

    dev::H2D(valuesA_d, valuesA.data(), valuesA.size());
    dev::H2D(valuesB_d, valuesB.data(), valuesB.size());

    const int numThreadPerBlock = 1024;
    convertFp32ToFp16<<< (valuesA.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
        valuesA.size(), valuesA_d, valuesAfp16_d);
    convertFp32ToFp16<<< (valuesB.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
        valuesB.size(), valuesB_d, valuesBfp16_d);

    const int M = matrixS2D.row();
    const int N = matrixS2D.col();

    dim3 grid;
    dim3 block;

    block.x = 128;
    block.y = 4;

    grid.x = (matrixS2D.row() + (WMMA_M * block.x / 32 - 1)) / (WMMA_M * block.x / 32);
    grid.y = (matrixS2D.col() + WMMA_N * block.y - 1) / (WMMA_N * block.y);

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();
    compSddmm<<<grid, block>>>(M, N, K, valuesAfp16_d, valuesBfp16_d, valuesS_d, valuesP_d);
    timeCalculator.endClock();
    std::cout << "Func compSddmm time : " << timeCalculator.getTime() << "ms" << std::endl;

    std::vector<float> valuesP(matrixS2D.size());
    dev::D2H(valuesP.data(), valuesP_d, valuesP.size());
    std::cout << "valuesP_zcx : " << std::endl;
    for (int idx = 0; idx < valuesP.size(); ++idx) {
        std::cout << valuesP[idx] << " ";
    }
    std::cout << std::endl;

    isratnisa::Matrix isratnisaMatrixS;
    isratnisaMatrixS.copyFromMatrix(matrixS);

    float *valuesP_isratnisa = nullptr;
    preprocessing(isratnisaMatrixS, valuesA, valuesB, valuesP_isratnisa);
    return 0;
}