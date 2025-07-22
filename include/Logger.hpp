#pragma once

#include <Options.hpp>
#include <cstdio>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#include <cmath>

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

    inline void printLogInformation(std::ostream& out = std::cout) const;

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

void Logger::printLogInformation(std::ostream& out) const{
    out << "[File : " << inputFile_ << "]\n";

    out << "[Build type : " << buildType_ << "]\n";
    out << "[Device : " << gpu_ << "]\n";

    out << "[WMMA_M : " << wmma_m_ << "], [WMMA_N : " << wmma_n_ << "], [WMMA_K : " << wmma_k_ << "]\n";

    out << "[K : " << K_ << "], ";
    out << "[M : " << M_ << "], ";
    out << "[N : " << N_ << "], ";
    out << "[NNZ : " << NNZ_ << "], ";
    out << "[sparsity : " << std::fixed << std::setprecision(2) << (std::floor(sparsity_ * 10000) / 100.0) << "%]\n";

    out << "[matrixA type : " << matrixA_type_ << "]\n";
    out << "[matrixB type : " << matrixB_type_ << "]\n";
    out << "[matrixC type : " << matrixC_type_ << "]\n";

    out << "[matrixA storageOrder : " << matrixA_storageOrder_ << "]\n";
    out << "[matrixB storageOrder : " << matrixB_storageOrder_ << "]\n";

    out << "[Num iterations : " << numITER_ << "]\n";

    out << "[NumRowPanel : " << numRowPanels_ << "]\n";

    out << "[original_numDenseBlock : " << originalNumDenseBlock_ << "]\n";
    out << "[original_averageDensity : " << originalAverageDensity_ << "]\n";

    out << "[bsmr_alpha : " << alpha_ << "]\n";
    out << "[bsmr_delta : " << delta_ << "]\n";

    out << "[bsmr_numClusters : " << numClusters_ << "]\n";
    out << "[bsmr_numDenseBlock : " << numDenseBlock_ << "]\n";
    out << "[bsmr_averageDensity : " << averageDensity_ << "]\n";

    out << "[bsmr_rowReordering : " << rowReorderingTime_ << "]\n";
    out << "[bsmr_colReordering : " << colReorderingTime_ << "]\n";
    out << "[bsmr_reordering : " << reorderingTime_ << "]\n";

    out << "[gridDim_dense : " << gridDim_dense_.x << ", " << gridDim_dense_.y << ", " << gridDim_dense_.z << "]\n";
    out << "[blockDim_dense : " << blockDim_dense_.x << ", " << blockDim_dense_.y << ", " << blockDim_dense_.z << "]\n";

    out << "[gridDim_sparse : " << gridDim_sparse_.x << ", " << gridDim_sparse_.y << ", " << gridDim_sparse_.z << "]\n";
    out << "[blockDim_sparse : " << blockDim_sparse_.x << ", " << blockDim_sparse_.y << ", " << blockDim_sparse_.z <<
        "]\n";

    out << "[bsmr_numDenseThreadBlocks : " << numDenseThreadBlocks_ << "]\n";
    out << "[bsmr_numSparseThreadBlocks : " << numSparseThreadBlocks_ << "]\n";
    out << "[bsmr_threadBlockRatio : " << std::fixed << std::setprecision(2)
        << static_cast<float>(numDenseThreadBlocks_) / numSparseThreadBlocks_ << "]\n";

    out << "[bsmr_numDenseData : " << numDenseData_ << "]\n";
    out << "[bsmr_numSparseData : " << numSparseData_ << "]\n";
    out << "[bsmr_dataRatio: " << std::fixed << std::setprecision(2)
        << static_cast<float>(numDenseData_) / numSparseData_ << "]\n";

    const size_t flops = 2 * NNZ_ * K_;

    out << "[cuSparse_gflops : " << std::fixed << std::setprecision(2)
        << (flops / (sddmmTime_cuSparse_ * 1e6)) << "]\n";
    out << "[cuSparse_sddmm : " << sddmmTime_cuSparse_ << "]\n";

    out << "[bsmr_gflops : " << (flops / (sddmmTime_ * 1e6)) << "]\n";
    out << "[bsmr_sddmm : " << sddmmTime_ << "]\n";

    if (errorRate_ > 0){
        out << "[checkResults : NO PASS Error rate : " << std::fixed << std::setprecision(2)
            << errorRate_ << "%]\n";
    }
}
