#pragma once

#include <Options.hpp>
#include <cstdio>
#include <iostream>
#include <string>

#include "Matrix.hpp"

struct Logger{
    Logger(){
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

    inline void getInformation(const Options& options);

    inline void getInformation(const sparseMatrix::DataBase& matrix);

    template <typename T>
    inline void getInformation(const Matrix<T>& matrixA, const Matrix<T>& matrixB){
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

    dim3 gridDim_dense_;
    dim3 gridDim_sparse_;
    dim3 blockDim_dense_;
    dim3 blockDim_sparse_;

    int numRowPanels_;

    int numDenseBlock_;
    float averageDensity_;

    int originalNumDenseBlock_;
    float originalAverageDensity_;

    int numDenseThreadBlocks_;
    int numSparseThreadBlocks_;

    int numDenseData_;
    int numSparseData_;

    int numITER_;

    float alpha_;
    float delta_;

    int numClusters_ = 1;

    float sddmmTime_ = 0.0f;
    float rowReorderingTime_ = 0.0f;
    float colReorderingTime_ = 0.0f;
    float reorderingTime_ = 0.0f;

    float sddmmTime_cuSparse_ = 0.0f;
};

void Logger::getInformation(const Options& options){
    inputFile_ = options.inputFile();
    K_ = options.K();
    numITER_ = options.numIterations();
    alpha_ = options.similarityThresholdAlpha();
    delta_ = options.blockDensityThresholdDelta();
}

void Logger::getInformation(const sparseMatrix::DataBase& matrix){
    M_ = matrix.row();
    N_ = matrix.col();
    NNZ_ = matrix.nnz();
    sparsity_ = matrix.getSparsity();
}

void Logger::printLogInformation(){
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

    printf("[Num iterations : %d]\n", numITER_);

    printf("[NumRowPanel : %d]\n", numRowPanels_);

    printf("[original_numDenseBlock : %d]\n", originalNumDenseBlock_);
    printf("[original_averageDensity : %f]\n", originalAverageDensity_);

    printf("[bsmr_alpha : %.2f]\n", alpha_);
    printf("[bsmr_delta : %.2f]\n", delta_);

    printf("[bsmr_numClusters : %d]\n", numClusters_);

    printf("[bsmr_numDenseBlock : %d]\n", numDenseBlock_);
    printf("[bsmr_averageDensity : %f]\n", averageDensity_);

    printf("[bsmr_rowReordering : %.2f]\n", rowReorderingTime_);
    printf("[bsmr_colReordering : %.2f]\n", colReorderingTime_);
    printf("[bsmr_reordering : %.2f]\n", reorderingTime_);

    printf("[gridDim_dense : %d, %d, %d]\n", gridDim_dense_.x, gridDim_dense_.y, gridDim_dense_.z);
    printf("[blockDim_dense : %d, %d, %d]\n", blockDim_dense_.x, blockDim_dense_.y, blockDim_dense_.z);

    printf("[gridDim_sparse : %d, %d, %d]\n", gridDim_sparse_.x, gridDim_sparse_.y, gridDim_sparse_.z);
    printf("[blockDim_sparse : %d, %d, %d]\n", blockDim_sparse_.x, blockDim_sparse_.y, blockDim_sparse_.z);

    printf("[bsmr_numDenseThreadBlocks : %d]\n", numDenseThreadBlocks_);
    printf("[bsmr_numSparseThreadBlocks : %d]\n", numSparseThreadBlocks_);
    printf("[bsmr_threadBlockRatio : %.2f]\n", static_cast<float>(numDenseThreadBlocks_) / numSparseThreadBlocks_);

    printf("[bsmr_numDenseData : %d]\n", numDenseData_);
    printf("[bsmr_numSparseData : %d]\n", numSparseData_);
    printf("[bsmr_dataRatio: %.2f]\n", static_cast<float>(numDenseData_) / numSparseData_);

    const size_t flops = 2 * NNZ_ * K_;

    printf("[cuSparse_gflops : %.2f]\n", (flops / (sddmmTime_cuSparse_ * 1e6)));
    printf("[cuSparse_sddmm : %.2f]\n", sddmmTime_cuSparse_);

    printf("[bsmr_gflops : %.2f]\n", (flops / (sddmmTime_ * 1e6)));
    printf("[bsmr_sddmm : %.2f]\n", sddmmTime_);

    if (errorRate_ > 0){
        printf("[checkResults : NO PASS Error rate : %2.2f%%]\n", errorRate_);
    }
}
