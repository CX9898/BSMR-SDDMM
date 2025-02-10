#include "sddmm.hpp"
#include "kernel.cuh"
#include "cudaErrorCheck.cuh"
#include "devVector.cuh"
#include "CudaTimeCalculator.cuh"
#include "host.hpp"
#include "checkData.hpp"
#include "ReBELL.hpp"

// The old method, directly uses TensorCore calculation
void sddmm(Matrix<float> &matrixA, Matrix<float> &matrixB, SparseMatrix<float> &matrixS, SparseMatrix<float> &matrixP) {

    TensorCoreConfig tensorCoreConfig(matrixS.row(), matrixS.col());

    printf("Kernel gridDim : [%d,%d,%d], blockDim : [%d,%d,%d]\n",
           tensorCoreConfig.gridDim().x, tensorCoreConfig.gridDim().y, tensorCoreConfig.gridDim().z,
           tensorCoreConfig.blockDim().x, tensorCoreConfig.blockDim().y, tensorCoreConfig.blockDim().z);
    printf("[WMMA_M : %d], [WMMA_N : %d], [WMMA_K : %d]\n", WMMA_M, WMMA_N, WMMA_K);

    matrixA.openTensorCoreMode(tensorCoreConfig, MatrixMultiplicationOrder::left_multiplication);
    printf("openTensorCoreMode matrixA : row = %d, col = %d\n", matrixA.row(), matrixA.col());
    matrixB.openTensorCoreMode(tensorCoreConfig, MatrixMultiplicationOrder::right_multiplication);
    printf("openTensorCoreMode matrixB : row = %d, col = %d\n", matrixB.row(), matrixB.col());

    //
    dev::vector<MATRIX_A_TYPE> matrixA_values_convertedType(matrixA.size());
    dev::vector<MATRIX_B_TYPE> matrixB_values_convertedType(matrixB.size());
    {
        dev::vector<float> matrixA_values_dev(matrixA.values());
        dev::vector<float> matrixB_values_dev(matrixB.values());

        const int numThreadPerBlock = 1024;
        kernel::convertDataType<<< (matrixA.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            matrixA.size(), matrixA_values_dev.data(), matrixA_values_convertedType.data());
        kernel::convertDataType<<< (matrixB.size() + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            matrixB.size(), matrixB_values_dev.data(), matrixB_values_convertedType.data());
    }

    CudaTimeCalculator timeCalculator;

    timeCalculator.startClock();
    matrixS.openTensorCoreModeForSampled(tensorCoreConfig);
    timeCalculator.endClock();
    const float openTensorCoreModeForSampled_time = timeCalculator.getTime();
    printf("openTensorCoreModeForSampled matrixS : row = %d, col = %d\n", matrixS.row(), matrixS.col());

    SparseMatrix<float> matrixP_cpu_res(matrixS.row(), matrixS.col(), matrixS.nnz(),
                                        matrixS.rowIndices(), matrixS.colIndices());

    timeCalculator.startClock();
    // comp by cpu
    sddmm_cpu(matrixA, matrixB, matrixS, matrixP_cpu_res);
    timeCalculator.endClock();
    std::cout << "Func sddmm_cpu time : " << timeCalculator.getTime() << " ms" << std::endl;

    dev::vector<UIN> matrixS_rowIndex_coo(matrixS.rowIndices());
    dev::vector<UIN> matrixS_colIndex_coo(matrixS.colIndices());
    dev::vector<UIN> matrixS_matrixTileMappedToWarpIndex_coo(matrixS.matrixTileMappedToWarpIndex());
    dev::vector<float> matrixS_value_coo(matrixS.values());
    dev::vector<float> matrixP_value_coo3(matrixS.values());
    timeCalculator.startClock();
    sddmm_gpu_coo_3(tensorCoreConfig,
                    matrixS.row(),
                    matrixS.col(),
                    matrixA.col(),
                    matrixA_values_convertedType.data(),
                    matrixA.storageOrder(),
                    matrixB_values_convertedType.data(),
                    matrixB.storageOrder(),
                    matrixS_rowIndex_coo.data(),
                    matrixS_colIndex_coo.data(),
                    matrixS_value_coo.data(),
                    matrixS_matrixTileMappedToWarpIndex_coo.data(),
                    matrixP_value_coo3.data());
    timeCalculator.endClock();
    const float time_sddmm_gpu_coo3 = timeCalculator.getTime();
    std::cout << "Func time_sddmm_gpu_coo3 time : " << time_sddmm_gpu_coo3 << " ms" << std::endl;

    std::cout << "check matrixP_cpu_res and sddmm_gpu_coo_3 : " << std::endl;

    size_t numError_3 = 0;
    if (!checkData(matrixP_cpu_res.values(), matrixP_value_coo3, numError_3)) {
        printf("[checkData : NO PASS Error rate : %2.2f%%]\n",
               static_cast<float>(numError_3) / static_cast<float>(matrixP_cpu_res.values().size()) * 100);
    }

    std::cout << "closeTensorCoreMode" << std::endl;
    matrixA.closeTensorCoreMode();
    matrixB.closeTensorCoreMode();
    matrixS.closeTensorCoreMode();

    const float time_sddmm_zcx = openTensorCoreModeForSampled_time + time_sddmm_gpu_coo3;
    std::cout << "sddmm_zcx time : " << time_sddmm_zcx << " ms" << std::endl;

    printf("[zcx_sddmm : %.2f]\n", time_sddmm_gpu_coo3);
    printf("[zcx_other : %.2f]\n", openTensorCoreModeForSampled_time);
    printf("[zcx : %.2f]\n", time_sddmm_zcx);
}

// Reordering method
void sddmm(const Matrix<float> &matrixA,
           const Matrix<float> &matrixB,
           const sparseDataType::CSR<float> &matrixS,
           sparseDataType::CSR<float> &matrixP) {

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();

    // Reordering
    ReBELL rebell(matrixS);

    timeCalculator.endClock();
    float rebell_time = timeCalculator.getTime();
    printf("rebell time : %.2f\n", rebell_time);

    // Error check
    bool rowReorderingIsCorrect = check_rowReordering(matrixS, rebell);
    if (!rowReorderingIsCorrect) {
        std::cerr << "Error! The row reordering is incorrect!" << std::endl;
    }

    // Error check
    bool colReorderingIsCorrect = check_colReordering(matrixS, rebell);
    if (!colReorderingIsCorrect) {
        std::cerr << "Error! The col reordering is incorrect!" << std::endl;
    }

    // sddmm comp by cpu
    sddmm_gpu_rebell(matrixA, matrixB, matrixS, rebell, matrixP);

    // sddmm comp by cpu
    sparseDataType::CSR<float> matrixP_cpu_res(matrixS);
    sddmm_cpu(matrixA, matrixB, matrixS, matrixP_cpu_res);

    // Error check
    size_t numError = 0;
    if (!checkData(matrixP_cpu_res.values_, matrixP.values_, numError)) {
        printf("[checkData : NO PASS Error rate : %2.2f%%]\n",
               static_cast<float>(numError) / static_cast<float>(matrixP.values_.size()) * 100);
    }
}