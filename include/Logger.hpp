#pragma once

#include <cstdio>
#include <iostream>
#include <string>

class Logger {
 public:

  void printLogInformation();

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
  std::string matrixC_storageOrder_;

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

void Logger::printLogInformation() {
    printf("[Build type : %s]\n", buildType_.c_str());
    printf("[Device : %s]\n", gpu_.c_str());

    printf("[M : %ld], ", M_);
    printf("[N : %ld], ", N_);
    printf("[K : %ld], ", K_);
    printf("[NNZ : %ld], ", NNZ_);
    printf("[sparsity : %.2f%%]\n", sparsity_ * 100);

    printf("[matrixA type : %s]\n", typeid(MATRIX_A_TYPE).name());
    printf("[matrixB type : %s]\n", typeid(MATRIX_B_TYPE).name());
    printf("[matrixC type : %s]\n", typeid(MATRIX_C_TYPE).name());

    printf("[cuSparse : %.2f]\n", cuSparse_);

    printf("[WMMA_M : %d], [WMMA_N : %d], [WMMA_K : %d]\n", wmma_m_, wmma_n_, wmma_k_);

    printf("[zcx_sddmm : %.2f]\n", zcx_sddmm_);
    printf("[zcx_other : %.2f]\n", zcx_other_);
    printf("[zcx : %.2f]\n", zcx_);

    printf("[checkData : NO PASS Error rate : %2.2f%%]\n",
           static_cast<float>(numError) / static_cast<float>(NNZ_) * 100);
}