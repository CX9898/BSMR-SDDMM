#pragma once

#include <Options.hpp>
#include <cstdio>
#include <iostream>
#include <string>

#include "Matrix.hpp"

struct Logger {
    Logger() {
#ifdef NDEBUG
        buildType_ = "Release";
#endif

#ifndef NDEBUG
        buildType_ = "Debug";
#endif

        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, 0);
        gpu_ = deviceProp.name;

        matrixA_type_ = typeid(MATRIX_A_TYPE).name();
        matrixB_type_ = typeid(MATRIX_B_TYPE).name();
        matrixC_type_ = typeid(MATRIX_C_TYPE).name();

        wmma_m_ = WMMA_M;
        wmma_n_ = WMMA_N;
        wmma_k_ = WMMA_K;
    };

    inline void getInformation(const Options &options);

    inline void getInformation(const sparseMatrix::DataBase &matrix);

    template<typename T>
    inline void getInformation(const Matrix<T> &matrixA, const Matrix<T> &matrixB) {
        K_ = matrixA.col();
        matrixA_storageOrder_ = matrixA.storageOrder() == MatrixStorageOrder::row_major ? "row_major" : "col_major";
        matrixB_storageOrder_ = matrixB.storageOrder() == MatrixStorageOrder::row_major ? "row_major" : "col_major";
    }

    inline void printLogInformation();

    std::string inputFile_;

    std::string checkData_;
    float errorRate_ = 0.0f;

    std::string gpu_;
    std::string buildType_;

    size_t wmma_m_;
    size_t wmma_n_;
    size_t wmma_k_;

    std::string matrixA_type_;
    std::string matrixB_type_;
    std::string matrixC_type_;

    std::string matrixA_storageOrder_;
    std::string matrixB_storageOrder_;

    size_t M_;
    size_t N_;
    size_t K_;
    size_t NNZ_;
    float sparsity_;

    int numDenseBlock_;
    float averageDensity_;

    int numSparseBlock_;

    int numITER_;

    float alpha_;
    int beta_;

    float zcx_sddmm_time_ = 0.0f;
    float zcx_preprocessing_time_ = 0.0f;

    float cuSparse_sddmm_time_ = 0.0f;
};

void Logger::getInformation(const Options &options) {
    inputFile_ = options.inputFile();
    K_ = options.K();
    numITER_ = options.numIterations();
    alpha_ = options.similarityThresholdAlpha();
    beta_ = options.columnNonZeroThresholdBeta();
}

void Logger::getInformation(const sparseMatrix::DataBase &matrix) {
    M_ = matrix.row();
    N_ = matrix.col();
    NNZ_ = matrix.nnz();
    sparsity_ = matrix.getSparsity();
}

void Logger::printLogInformation() {
    printf("[File : %s]\n", inputFile_.c_str());

    printf("[Build type : %s]\n", buildType_.c_str());
    printf("[Device : %s]\n", gpu_.c_str());

    printf("[WMMA_M : %zu], [WMMA_N : %zu], [WMMA_K : %zu]\n", wmma_m_, wmma_n_, wmma_k_);

    printf("[K : %ld], ", K_);

    printf("[M : %ld], ", M_);
    printf("[N : %ld], ", N_);
    printf("[NNZ : %ld], ", NNZ_);
    printf("[sparsity : %.2f%%]\n", floor(sparsity_ * 10000) / 100.0);

    printf("[matrixA type : %s]\n", matrixA_type_.c_str());
    printf("[matrixB type : %s]\n", matrixB_type_.c_str());
    printf("[matrixC type : %s]\n", matrixC_type_.c_str());

    printf("[matrixA storageOrder : %s]\n", matrixA_storageOrder_.c_str());
    printf("[matrixB storageOrder : %s]\n", matrixB_storageOrder_.c_str());

    printf("[zcx_numDenseBlock : %d]\n", numDenseBlock_);
    printf("[zcx_averageDensity : %f]\n", averageDensity_);
    printf("[NumSparseBlock : %d]\n", numSparseBlock_);

    printf("[zcx_alpha : %.2f]\n", alpha_);
    printf("[zcx_beta : %d]\n", beta_);

    printf("[Num iterations : %d]\n", numITER_);

    cuSparse_sddmm_time_ /= numITER_;
    zcx_sddmm_time_ /= numITER_;

    const size_t flops = 2 * NNZ_ * K_;

    printf("[cuSparse_gflops : %.2f]\n", (flops / (cuSparse_sddmm_time_ * 1e6)));
    printf("[cuSparse_sddmm : %.2f]\n", cuSparse_sddmm_time_);

    printf("[zcx_gflops : %.2f]\n", (flops / (zcx_sddmm_time_ * 1e6)));
    printf("[zcx_sddmm : %.2f]\n", zcx_sddmm_time_);
    printf("[zcx_preprocessing : %.2f]\n", zcx_preprocessing_time_);

    if (errorRate_ > 0) {
        printf("[checkResults : NO PASS Error rate : %2.2f%%]\n", errorRate_);
    }
}
