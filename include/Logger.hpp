#pragma once

#include <cstdio>
#include <iostream>
#include <string>

#include "Matrix.hpp"

class Logger {
public:
    Logger() {
#ifdef NDEBUG
        buildType_ = "Release";
#endif

#ifndef NDEBUG
        buildType_ = "Debug";
#endif

        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, 0);
        gpu_ =  deviceProp.name;

        matrixA_type_ = typeid(MATRIX_A_TYPE).name();
        matrixB_type_ = typeid(MATRIX_B_TYPE).name();
        matrixC_type_ = typeid(MATRIX_C_TYPE).name();

        wmma_m_ = WMMA_M;
        wmma_n_ = WMMA_N;
        wmma_k_ = WMMA_K;
    };

    template<typename T>
    inline void getInformation(const sparseMatrix::COO<T> &matrix);

    template<typename T>
    inline void getInformation(const Matrix<T> &matrixA, const Matrix<T> &matrixB){
        K_ = matrixA.col();
        matrixA_storageOrder_ = matrixA.storageOrder() == MatrixStorageOrder::row_major ? "row_major" : "col_major";
        matrixB_storageOrder_ = matrixB.storageOrder() == MatrixStorageOrder::row_major ? "row_major" : "col_major";

    }

    inline float &zcx_sddmm(){ return zcx_sddmm_;}
    inline float &zcx_other(){ return zcx_other_;}
    inline float &cuSparse(){ return cuSparse_;}

    inline void printLogInformation();

private:
    std::string checkData_;
    size_t numError;

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

    float isratnisa_sddmm_;
    float zcx_sddmm_;

    float isratnisa_other_;
    float zcx_other_;

    float isratnisa_;
    float zcx_;
    float cuSparse_;
};

template<typename T>
void Logger::getInformation(const sparseMatrix::COO<T> &matrix) {
    M_ = matrix.row();
    N_ = matrix.col();
    NNZ_ = matrix.nnz();
    sparsity_ = matrix.getSparsity();

}

void Logger::printLogInformation() {
    printf("[Build type : %s]\n", buildType_.c_str());
    printf("[Device : %s]\n", gpu_.c_str());

    printf("[M : %ld], ", M_);
    printf("[N : %ld], ", N_);
    printf("[K : %ld], ", K_);
    printf("[NNZ : %ld], ", NNZ_);
    printf("[sparsity : %.2f%%]\n", sparsity_ * 100);

    printf("[matrixA type : %s]\n", matrixA_type_.c_str());
    printf("[matrixB type : %s]\n", matrixB_type_.c_str());
    printf("[matrixC type : %s]\n", matrixC_type_.c_str());

    printf("[cuSparse : %.2f]\n", cuSparse_);

    printf("[WMMA_M : %d], [WMMA_N : %d], [WMMA_K : %d]\n", wmma_m_, wmma_n_, wmma_k_);

    printf("[zcx_sddmm : %.2f]\n", zcx_sddmm_);
    printf("[zcx_other : %.2f]\n", zcx_other_);
    zcx_ = zcx_other_ + zcx_sddmm_;
    printf("[zcx : %.2f]\n", zcx_);
}