#include <iostream>
#include <string>
#include <typeinfo>

//#include "sddmm_isratnisa.h"
//#include "util_isratnisa.h"
#include "Matrix.hpp"
#include "devMatrix.cuh"
#include "kernel.cuh"
#include "TensorCoreConfig.cuh"
#include "cudaErrorCheck.cuh"
#include "CudaTimeCalculator.cuh"
#include "host.hpp"
#include "checkData.hpp"
#include "devVector.cuh"
#include "util.hpp"

const std::string folderPath("../dataset/test/matrix_10000_15000_/");
//const std::string folderPath("./");
//const std::string fileName = ("nips");
//const std::string fileName = ("test");
const std::string fileName("matrix_10000_15000_7500000");
const std::string fileFormat(".mtx");
const std::string filePath = folderPath + fileName + fileFormat;

// TODO :
//      测试矩阵的尺寸按照论文中的尺寸
//      1: 将 comp_sddmm_gpu 全部使用 Tensor core 执行          OK
//      2: 测试不同尺寸的 wmma 的速度表现                         OK
//      3: 测试使用稀疏度比较器的速度表现
//              稀疏度大于50%使用 isratnisa 的方法
//              稀疏度小于50%使用 Tensor core 方法
//      4: 全部数据放在device内存                                OK
//      5: 优化openTensorCoreModeForSampled()                  OK
//      6: 测试更大的K(<5k)的结果                                OK
//      7: 优化positionCalculator(),                           OK
//                  支持 WMMA 维度 : 16×16×16                   OK
//                  支持 WMMA 维度 : 32×8×16                    OK
//                  支持 WMMA 维度 : 8×32×16                    OK
//      8: sddmm函数中支持各种矩阵储存维度
//                    matrixA: row_major matrixB: row_major    OK
//                    matrixA: row_major matrixB: col_major    OK
//                    matrixA: col_major matrixB: row_major
//                    matrixA: col_major matrixB: col_major
//      9: TensorCoreConfig 支持 WarpOrder
//      10: 在sddmm函数中使用共享内存

//#define MAKE_MATRIX_DATA

int main(int argc, char *argv[]) {

#ifdef MAKE_MATRIX_DATA
    // make sparse matrix data
    {
        SparseMatrix<int> matrixTmp;
        const size_t thousand = 1000;
        const size_t million = 1000000;
//    const size_t makeDataRow = 3 * thousand;
//    const size_t makeDataCol = 7 * thousand;
        const size_t makeDataRow = 10000;
        const size_t makeDataCol = 15000;
//    const float density = 4.006f;
//    const size_t makeDataNNZ = static_cast<int> (makeDataRow * makeDataCol * density / 100);
//    const float sparsity = 0.80;
//    const size_t makeDataNNZ = makeDataRow * makeDataCol * (1 - sparsity);
//    const size_t makeDataNNZ = 1 * million;
        const size_t makeDataNNZ = 7500000;
        matrixTmp.makeData(makeDataRow, makeDataCol, makeDataNNZ);
        matrixTmp.outputToMarketMatrixFile();
        std::cout << "makeData : M : " << makeDataRow
                  << ", N : " << makeDataCol
                  << ", K : " << 256
                  << ", nnz : " << makeDataNNZ
                  << ", sparsity : "
                  << (float) (makeDataRow * makeDataCol - makeDataNNZ) / (makeDataRow * makeDataCol) * 100 << "%"
                  << std::endl;
        exit(0);
    }
#endif // MAKE_MATRIX_DATA

    size_t K = 256;
    SparseMatrix<float> matrixS;

    if (argc > 2) {
        if (!matrixS.initializeFromMatrixMarketFile(argv[1])) {
            exit(1);
        }
        K = std::stol(argv[2]);
    } else if (argc == 2) {
        if (!matrixS.initializeFromMatrixMarketFile(argv[1])) {
            exit(1);
        }
    } else {
        if (!matrixS.initializeFromMatrixMarketFile(util::getParentFolderPath(argv[0]) + filePath)) {
            exit(1);
        }
    }

#ifdef _DEBUG
    printf("@Build type : Debug @\n");
#endif

#ifdef NDEBUG
    printf("@Build type : Release @\n");
#endif

    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("@Device : %s @\n", deviceProp.name);

    printf("@M : %d @, ", matrixS.row());
    printf("@N : %d @, ", matrixS.col());
    printf("@K : %ld @, ", K);
    printf("@NNZ : %d @, ", matrixS.nnz());
    printf("@sparsity : %.2f%% @\n", matrixS.getSparsity() * 100);

    TensorCoreConfig tensorCoreConfig(matrixS.row(), matrixS.col());
    printf("Kernel gridDim : [%d,%d,%d], blockDim : [%d,%d,%d]\n",
           tensorCoreConfig.gridDim().x,tensorCoreConfig.gridDim().y,tensorCoreConfig.gridDim().z,
           tensorCoreConfig.blockDim().x,tensorCoreConfig.blockDim().y,tensorCoreConfig.blockDim().z);
    printf("@WMMA_M : %d @, @WMMA_N : %d @, @WMMA_K : %d @\n", WMMA_M, WMMA_N, WMMA_K);

    printf("@matrixA type : %s @\n", typeid(MATRIX_A_TYPE).name());
    printf("@matrixB type : %s @\n", typeid(MATRIX_B_TYPE).name());
    printf("@matrixC type : %s @\n", typeid(MATRIX_C_TYPE).name());

    Matrix<float> matrixA(matrixS.row(), K, MatrixStorageOrder::row_major);
    matrixA.makeData(matrixA.row(), K);
//    std::cout << "matrixA.size() : " << matrixA.values().size() << " matrixA : ";
//    matrixA.print();

    Matrix<float> matrixB(K, matrixS.col(), MatrixStorageOrder::row_major);
    matrixB.makeData(K, matrixS.col());
//    std::cout << "matrixB.size() : " << matrixB.values().size() << " matrixB : ";
//    matrixB.print();

//    matrixA.changeStorageOrder();
//    matrixB.changeStorageOrder();

    if (matrixA.storageOrder() == MatrixStorageOrder::row_major) { printf("@matrixA storageOrder : row_major @\n"); }
    else { printf("@matrixA storageOrder : col_major @\n"); }
    if (matrixB.storageOrder() == MatrixStorageOrder::row_major) { printf("@matrixB storageOrder : row_major @\n"); }
    else { printf("@matrixB storageOrder : col_major @\n"); }

    matrixA.openTensorCoreMode(tensorCoreConfig, MatrixMultiplicationOrder::left_multiplication);
    printf("openTensorCoreMode matrixA : row = %d, col = %d\n", matrixA.row(), matrixA.col());
    matrixB.openTensorCoreMode(tensorCoreConfig, MatrixMultiplicationOrder::right_multiplication);
    printf("openTensorCoreMode matrixB : row = %d, col = %d\n", matrixB.row(), matrixB.col());

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
                                        matrixS.rowIndex(), matrixS.colIndex());

    timeCalculator.startClock();
    // comp by cpu
    sddmm_cpu(matrixA, matrixB, matrixS, matrixP_cpu_res);
    timeCalculator.endClock();
    std::cout << "Func sddmm_cpu time : " << timeCalculator.getTime() << " ms" << std::endl;


//    dev::SparseMatrix<float> matrixS_dev(matrixS);
//    timeCalculator.startClock();
//    matrixS_dev.openTensorCoreModeForSampled(tensorCoreConfig);
//    timeCalculator.endClock();
//    const float openTensorCoreModeForSampled_dev_time = timeCalculator.getTime();
//    std::cout << "Func openTensorCoreModeForSampled_dev_time : " << openTensorCoreModeForSampled_dev_time << " ms"
//              << std::endl;
//    std::cout << "openTensorCoreModeForSampled matrixS_dev : row = "
//              << matrixS_dev.row() << ", col = " << matrixS_dev.col() << std::endl;
//
//    printf("@zcx_other : %.2f @\n", openTensorCoreModeForSampled_dev_time);
//
//
////    matrixS.openTensorCoreModeForSampled(tensorCoreConfig);
////    std::cout << "openTensorCoreModeForSampled matrixS : row = " << matrixS.row() << ", col = " << matrixS.col()
////              << std::endl;
//
//    SparseMatrix<float> matrixS_cpu(matrixS.row(), matrixS_dev.col(), matrixS_dev.nnz());
//    d2h(matrixS_cpu.setRowIndex(), matrixS_dev.rowIndex());
//    d2h(matrixS_cpu.setColIndex(), matrixS_dev.colIndex());
//    d2h(matrixS_cpu.setValues(), matrixS_dev.values());
//
//
////    Matrix<float> matrixC(matrixS.row(), matrixS.col(), MatrixStorageOrder::row_major);
////    dmm_cpu(matrixA, matrixB, matrixC);
////    matrixC.printToMarkdownTable();
//
//    dev::vector<float> matrixP_value_coo2(matrixS_dev.nnz());
//    timeCalculator.startClock();
//    sddmm_gpu_coo_2<<<tensorCoreConfig.gridDim(), tensorCoreConfig.blockDim()>>>(tensorCoreConfig,
//                                                                           matrixS_dev.row(),
//                                                                           matrixS_dev.col(),
//                                                                           matrixA.col(),
//                                                                           matrixS_dev.nnz(),
//                                                                           matrixA_values_convertedType.data(),
//                                                                           matrixB_values_convertedType.data(),
//                                                                           matrixS_dev.rowIndex().data(),
//                                                                           matrixS_dev.colIndex().data(),
//                                                                           matrixS_dev.values().data(),
//                                                                           matrixS_dev.matrixTileMappedToWarpIndex().data(),
//                                                                           matrixS_dev.matrixTileMappedToWarpIndexData().data(),
//                                                                           matrixP_value_coo2.data());
//    timeCalculator.endClock();
//    const float time_sddmm_gpu_coo2 = timeCalculator.getTime();
//    std::cout << "Func sddmm_gpu_coo_2 time : " << time_sddmm_gpu_coo2 << " ms" << std::endl;
//
//    std::cout << "check matrixP_cpu_res and sddmm_gpu_coo_2 : " << std::endl;
//
//    size_t numError = 0;
//    if (!checkData(matrixP_cpu_res.values(), matrixP_value_coo2, numError)) {
//        printf("@checkData : NO PASS numError = %leadingDimension @\n", numError);
//    }

    dev::vector<UIN> matrixS_rowIndex_coo(matrixS.rowIndex());
    dev::vector<UIN> matrixS_colIndex_coo(matrixS.colIndex());
    dev::vector<UIN> matrixS_matrixTileMappedToWarpIndex_coo(matrixS.matrixTileMappedToWarpIndex());
    dev::vector<float> matrixS_value_coo(matrixS.values());
    dev::vector<float> matrixP_value_coo3(matrixS.nnz());
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
        printf("@checkData : NO PASS Error rate : %2.2f%% @\n",
               static_cast<float>(numError_3) / static_cast<float>(matrixP_cpu_res.values().size()) * 100);
    }

    std::cout << "closeTensorCoreMode" << std::endl;
    matrixA.closeTensorCoreMode();
    matrixB.closeTensorCoreMode();
    matrixS.closeTensorCoreMode();

    const float time_sddmm_zcx = openTensorCoreModeForSampled_time + time_sddmm_gpu_coo3;
    std::cout << "sddmm_zcx time : " << time_sddmm_zcx << " ms" << std::endl;

    printf("@zcx_sddmm : %.2f @\n", time_sddmm_gpu_coo3);
    printf("@zcx_other : %.2f @\n", openTensorCoreModeForSampled_time);
    printf("@zcx : %.2f @\n", time_sddmm_zcx);

//    dmm_cpu(matrixA,matrixB,matrixS2D);

//    std::cout << "matrixP_gpu_res : " << std::endl;
//    matrixP_gpu_res.print();

//    isratnisa::Matrix isratnisaMatrixS;
//    isratnisaMatrixS.copyFromMatrix(matrixS);
//
//    float *valuesP_isratnisa = nullptr;
//    preprocessing(isratnisaMatrixS, matrixA.values(), matrixB.values(), valuesP_isratnisa);

    return 0;
}