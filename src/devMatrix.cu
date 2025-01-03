#include <cuda_runtime.h>

#include <fstream>

#include "devMatrix.cuh"
#include "devMatrixKernel.cuh"
#include "parallelAlgorithm.cuh"
#include "devVector.cuh"
#include "CudaTimeCalculator.cuh"
#include "checkData.hpp"
#include "cudaUtil.cuh"

//#include <set>

template<typename T>
UIN dev::Matrix<T>::rowOfValueIndex(UIN idx) const {
    if (idx == 0) {
        return 0;
    }
    if (storageOrder_ == MatrixStorageOrder::row_major) {
        return idx / leadingDimension_;
    } else {
        return idx % leadingDimension_;
    }
}

template<typename T>
UIN dev::Matrix<T>::colOfValueIndex(UIN idx) const {
    if (idx == 0) {
        return 0;
    }
    if (storageOrder_ == MatrixStorageOrder::row_major) {
        return idx % leadingDimension_;
    } else {
        return idx / leadingDimension_;
    }
}

template<typename T>
void dev::Matrix<T>::openTensorCoreMode(MatrixMultiplicationOrder multiplicationOrder) {
    // TODO
}

template<typename T>
void dev::Matrix<T>::closeTensorCoreMode() {
    // TODO
}

//template<typename T>
//void dev::SparseMatrix<T>::initializeFromMatrixMarketFile(const std::string &filePath) {
//    std::ifstream inFile;
//    inFile.open(filePath, std::ios::in); // open file
//    if (!inFile.is_open()) {
//        std::cerr << "Error, MatrixMarket file cannot be opened : " << filePath << std::endl;
//        return;
//    }
//
//    std::cout << "SparseMatrix initialize From MatrixMarket file : " << filePath << std::endl;
//
//    std::string line; // Store the data for each line
//    getline(inFile, line); // First line does not operate
//
//    getline(inFile, line);
//    int wordIter = 0;
//    row_ = std::stoi(::util::iterateOneWordFromLine(line, wordIter));
//    col_ = std::stoi(::util::iterateOneWordFromLine(line, wordIter));
//    nnz_ = std::stoi(::util::iterateOneWordFromLine(line, wordIter));
//
//    if (wordIter < line.size()) {
//        std::cerr << "Error, Matrix Market file " << line << " line format is incorrect!" << std::endl;
//    }
//
//    std::vector<UIN> rowIndices(nnz_);
//    std::vector<UIN> colIndices(nnz_);
//    std::vector<T> values(nnz_);
//
//    UIN idx = 0;
//    while (getline(inFile, line)) {
//        wordIter = 0;
//        const UIN row = std::stoi(util::iterateOneWordFromLine(line, wordIter)) - 1;
//        const UIN col = std::stoi(util::iterateOneWordFromLine(line, wordIter)) - 1;
//        const T val = static_cast<T>(std::stod(util::iterateOneWordFromLine(line, wordIter)));
//
//        if (wordIter < line.size()) {
//            std::cerr << "Error, Matrix Market file " << line << " line format is incorrect!" << std::endl;
//        }
//
//        rowIndices[idx] = row;
//        colIndices[idx] = col;
//        values[idx] = val;
//
//        ++idx;
//    }
//
//    inFile.close();
//
//    h2d(rowIndices_,rowIndices);
//    h2d(colIndices_,colIndices);
//    h2d(values_,values);
//
//    rowBeforeChange_ = row_;
//    colBeforeChange_ = col_;
//}

//#define TEST
//#define PRINT_TIME

template<typename T>
void dev::SparseMatrix<T>::openTensorCoreModeForSampled(TensorCoreConfig tensorCoreConfig) {
    if (tensorCoreMode_) {
        return;
    }

    tensorCoreMode_ = true;
    rowBeforeChange_ = row_;
    colBeforeChange_ = col_;

    row_ = tensorCoreConfig.MForTensorCore(rowBeforeChange_);
    col_ = tensorCoreConfig.NForTensorCore(colBeforeChange_);

    const UIN numWarpX = tensorCoreConfig.numWarpX();
    const UIN numWarpY = tensorCoreConfig.numWarpY();
    const UIN numWarps = numWarpX * numWarpY;

    const UIN numThreadsPerBlock = NUMBER_OF_THREADS_PER_BLOCK;
    const UIN numBlocks = (numWarps + numThreadsPerBlock - 1) / numThreadsPerBlock;

#ifdef PRINT_TIME
    CudaTimeCalculator timeCalculator;
#endif // PRINT_TIME

    //////////////////////////////// 1 right
#ifdef TEST
    std::vector<UIN> rightNum;
    {
        const UIN numTileM = row_ / WMMA_M;
        const UIN numTileN = col_ / WMMA_N;

        dev::vector<UIN> numOfIndexPerWarp_1(numWarps);
        dev::fill_n(numOfIndexPerWarp_1.data(), numOfIndexPerWarp_1.size(), 0);

        timeCalculator.startClock();
        getIndexPerWarp_1<<<numBlocks, NUMBER_OF_THREADS_PER_BLOCK>>>(numWarps,
                                                                      numWarpX,
                                                                      numTileM,
                                                                      numTileN,
                                                                      nnz_,
                                                                      rowIndices_.data(),
                                                                      colIndices_.data(),
                                                                      updateNumOfIndexOperator_1(numOfIndexPerWarp_1.data()));
        timeCalculator.endClock();
        float getNumIndexPerWarp_1_time = timeCalculator.getTime();
        std::cout << "    getNumIndexPerWarp_1_time : " << getNumIndexPerWarp_1_time << " ms" << std::endl;

        d2h(rightNum, numOfIndexPerWarp_1);

        timeCalculator.startClock();

        matrixTileMappedToWarpIndex_.resize(numWarps + 1);
        dev::fill_n(matrixTileMappedToWarpIndex_.data(), 1, 0);
        dev::inclusive_scan(numOfIndexPerWarp_1.data(),
                            numOfIndexPerWarp_1.data() + numOfIndexPerWarp_1.size(),
                            matrixTileMappedToWarpIndex_.data() + 1);
        const UIN numIndexData = matrixTileMappedToWarpIndex_.back_data();
        timeCalculator.endClock();
        float inclusive_scan_time = timeCalculator.getTime();
        std::cout << "    inclusive_scan_time : " << inclusive_scan_time << " ms" << std::endl;

        matrixTileMappedToWarpIndexData_.resize(numIndexData);

        timeCalculator.startClock();
        getIndexPerWarp_1<<<numBlocks, numThreadsPerBlock>>>(numWarps,
                                                             numWarpX,
                                                             numTileM,
                                                             numTileN,
                                                             nnz_,
                                                             rowIndices_.data(),
                                                             colIndices_.data(),
                                                             updateIndexDataPerWarpOperator_1(
                                                                 matrixTileMappedToWarpIndex_.data(),
                                                                 matrixTileMappedToWarpIndexData_.data()));
        timeCalculator.endClock();
        float getTileIndexDataPerWarp_time = timeCalculator.getTime();
        std::cout << "    getTileIndexDataPerWarp_time : " << getTileIndexDataPerWarp_time << " ms" << std::endl;

        printf(" @@@Method 1 time : %f\n",
               getNumIndexPerWarp_1_time + inclusive_scan_time + getTileIndexDataPerWarp_time);
    }
#endif // TEST
    //////////////////////////////// 2 OK
    {

#ifdef PRINT_TIME
        timeCalculator.startClock();
#endif // PRINT_TIME

        dev::vector<UIN> numIndexOfPerWarp_2(numWarps);
        dev::fill_n(numIndexOfPerWarp_2.data(), numIndexOfPerWarp_2.size(), 0);
        getIndexPerWarp_2<<<numBlocks, numThreadsPerBlock>>>(numWarpX,
                                                             nnz_,
                                                             rowIndex_.data(),
                                                             colIndex_.data(),
                                                             updateNumOfIndexOperator_2(numIndexOfPerWarp_2.data()));

#ifdef PRINT_TIME
        timeCalculator.endClock();
        float getIndexPerWarp_2_time = timeCalculator.getTime();
        std::cout << "    getIndexPerWarp_2_time : " << getIndexPerWarp_2_time << " ms" << std::endl;
#endif // PRINT_TIME

#ifdef TEST
        printf("  check rightNum and numIndexOfPerWarp_2 : \n");
        if (!checkData(rightNum, numIndexOfPerWarp_2)) {
            exit(1);
        }
#endif // TEST

#ifdef PRINT_TIME
        timeCalculator.startClock();
#endif // PRINT_TIME

        matrixTileMappedToWarpIndex_.resize(numWarps + 1);
        dev::fill_n(matrixTileMappedToWarpIndex_.data(), 1, 0);
        dev::inclusive_scan(numIndexOfPerWarp_2.data(),
                            numIndexOfPerWarp_2.data() + numIndexOfPerWarp_2.size(),
                            matrixTileMappedToWarpIndex_.data() + 1);
        const UIN numIndexData_2 = matrixTileMappedToWarpIndex_.back_data();

#ifdef PRINT_TIME
        timeCalculator.endClock();
        float inclusive_scan_2_time = timeCalculator.getTime();
        std::cout << "    inclusive_scan_2_time : " << inclusive_scan_2_time << " ms" << std::endl;
#endif // PRINT_TIME

#ifdef PRINT_TIME
        timeCalculator.startClock();
#endif // PRINT_TIME

        matrixTileMappedToWarpIndexData_.resize(numIndexData_2);
        getIndexPerWarp_2<<<numBlocks, numThreadsPerBlock>>>(numWarpX,
                                                             nnz_,
                                                             rowIndex_.data(),
                                                             colIndex_.data(),
                                                             updateIndexDataPerWarpOperator_2(
                                                                 matrixTileMappedToWarpIndex_.data(),
                                                                 matrixTileMappedToWarpIndexData_.data()));

#ifdef PRINT_TIME
        timeCalculator.endClock();
        float getIndexPerWarp_2_2_time = timeCalculator.getTime();
        std::cout << "    getIndexPerWarp_2_2_time : " << getIndexPerWarp_2_2_time << " ms" << std::endl;

        printf(" @@@Method 2 time : %f\n",
               getIndexPerWarp_2_time + inclusive_scan_2_time + getIndexPerWarp_2_2_time);
#endif // PRINT_TIME
    }

    //////////////////////////////// 3 OK 但是内存分配太多!
#ifdef TEST
    {
        dim3 gridForGetIndex;
        gridForGetIndex.x = (numWarps + NUMBER_OF_THREADS_PER_BLOCK - 1) / NUMBER_OF_THREADS_PER_BLOCK;
        gridForGetIndex.y = (nnz_ + SHARED_MEMORY_SIZE - 1) / SHARED_MEMORY_SIZE;
        dev::vector<UIN> scatteredNumOfIndexPerWarp_3(nnz_ * gridForGetIndex.y * NUMBER_OF_THREADS_PER_BLOCK);
        dev::fill_n(scatteredNumOfIndexPerWarp_3.data(), scatteredNumOfIndexPerWarp_3.size(), 0);

#ifdef PRINT_TIME
        timeCalculator.startClock();
#endif // PRINT_TIME

        getIndexPerWarp_3<<<gridForGetIndex, NUMBER_OF_THREADS_PER_BLOCK>>>(numWarpX,
                                                                            nnz_,
                                                                            rowIndices_.data(),
                                                                            colIndices_.data(),
                                                                            updateScatteredNumOfIndexOperator_3(
                                                                                scatteredNumOfIndexPerWarp_3.data()));

#ifdef PRINT_TIME
        timeCalculator.endClock();
        float updateScatteredNumOfIndexOperator_3_time = timeCalculator.getTime();
        std::cout << "    updateScatteredNumOfIndexOperator_3_time : " << updateScatteredNumOfIndexOperator_3_time
                  << " ms"
                  << std::endl;
#endif // PRINT_TIME

#ifdef PRINT_TIME
        timeCalculator.startClock();
#endif // PRINT_TIME

        dev::vector<UIN> numIndexPerWarp_3(numWarps);
        dev::fill_n(numIndexPerWarp_3.data(), numIndexPerWarp_3.size(), 0);
        mergeScatteredNumOfIndex_3<<<gridForGetIndex.x, NUMBER_OF_THREADS_PER_BLOCK>>>(numWarps, gridForGetIndex.y,
                                                                                       scatteredNumOfIndexPerWarp_3.data(),
                                                                                       numIndexPerWarp_3.data());
#ifdef PRINT_TIME
        timeCalculator.endClock();
        float mergeNumOfIndexPerWarp_3_time = timeCalculator.getTime();
        std::cout << "    mergeNumOfIndexPerWarp_3_time : " << mergeNumOfIndexPerWarp_3_time << " ms" << std::endl;

#endif // PRINT_TIME

#ifdef TEST
        printf("check rightNum and numIndexPerWarp_3 : \n");
        if (!checkData(rightNum, numIndexPerWarp_3)) {
            exit(1);
        }
#endif // TEST

#ifdef PRINT_TIME
        timeCalculator.startClock();
#endif // PRINT_TIME

        dev::vector<UIN> indexForScatteredNumOfIndex(scatteredNumOfIndexPerWarp_3.size() + 1);
        dev::fill_n(indexForScatteredNumOfIndex.data(), 1, 0);
        dev::inclusive_scan(scatteredNumOfIndexPerWarp_3.data(),
                            scatteredNumOfIndexPerWarp_3.data() + scatteredNumOfIndexPerWarp_3.size(),
                            indexForScatteredNumOfIndex.data() + 1);
        const UIN scatteredNumIndexData = indexForScatteredNumOfIndex.back_data();

#ifdef PRINT_TIME
        timeCalculator.endClock();
        float inclusive_scan_scattered_3_time = timeCalculator.getTime();
        printf("    inclusive_scan_scattered_3_time : %f\n", inclusive_scan_scattered_3_time);
#endif // PRINT_TIME

#ifdef PRINT_TIME
        timeCalculator.startClock();
#endif // PRINT_TIME

        dev::vector<UIN> scatteredIndexData(scatteredNumIndexData);
        getIndexPerWarp_3<<<gridForGetIndex, NUMBER_OF_THREADS_PER_BLOCK>>>(numWarpX,
                                                                            nnz_,
                                                                            rowIndices_.data(),
                                                                            colIndices_.data(),
                                                                            updateScatteredIndexDataPerWarpOperator_3(
                                                                                indexForScatteredNumOfIndex.data(),
                                                                                scatteredIndexData.data()));

#ifdef PRINT_TIME
        timeCalculator.endClock();
        float updateScatteredIndexDataPerWarpOperator_3_time = timeCalculator.getTime();
        printf("    updateScatteredIndexDataPerWarpOperator_3_time : %f ms\n",
               updateScatteredIndexDataPerWarpOperator_3_time);
#endif // PRINT_TIME

#ifdef PRINT_TIME
        timeCalculator.startClock();
#endif // PRINT_TIME

        matrixTileMappedToWarpIndex_.resize(numWarps + 1);
        dev::fill_n(matrixTileMappedToWarpIndex_.data(), 1, 0);
        dev::inclusive_scan(numIndexPerWarp_3.data(),
                            numIndexPerWarp_3.data() + numIndexPerWarp_3.size(),
                            matrixTileMappedToWarpIndex_.data() + 1);
        const UIN numIndexData_3 = matrixTileMappedToWarpIndex_.back_data();

        matrixTileMappedToWarpIndexData_.resize(numIndexData_3);

#ifdef PRINT_TIME
        timeCalculator.endClock();
        float inclusive_scan_3_time = timeCalculator.getTime();
        printf("    inclusive_scan_3_time : %f\n", inclusive_scan_3_time);
#endif // PRINT_TIME

#ifdef PRINT_TIME
        timeCalculator.startClock();
#endif // PRINT_TIME

        sortScatteredIndexData_3<<<gridForGetIndex.x, NUMBER_OF_THREADS_PER_BLOCK>>>(numWarps,
                                                                                     gridForGetIndex.y,
                                                                                     matrixTileMappedToWarpIndex_.data(),
                                                                                     indexForScatteredNumOfIndex.data(),
                                                                                     scatteredIndexData.data(),
                                                                                     matrixTileMappedToWarpIndexData_.data());
#ifdef PRINT_TIME
        timeCalculator.endClock();
        float sortScatteredIndexData_time = timeCalculator.getTime();
        std::cout << "    sortScatteredIndexData_time : " << sortScatteredIndexData_time << " ms" << std::endl;

        printf(" @@@Method 3 time : %f\n",
               updateScatteredNumOfIndexOperator_3_time + inclusive_scan_scattered_3_time
                   + updateScatteredIndexDataPerWarpOperator_3_time + mergeNumOfIndexPerWarp_3_time
                   + inclusive_scan_3_time
                   + sortScatteredIndexData_time);

//        d2d(matrixTileMappedToWarpIndex_, matrixTileMappedToWarpIndex_3);
//        d2d(matrixTileMappedToWarpIndexData_, matrixTileMappedToWarpIndexData_3);

#endif // PRINT_TIME
    }
#endif // TEST
    //////////////////////////////// 4 OK
//    {
//        dim3 gridForGetIndex;
//        gridForGetIndex.x = (numWarps + NUMBER_OF_CALCULATED_BY_ONE_BLOCK - 1) / NUMBER_OF_CALCULATED_BY_ONE_BLOCK;
//        gridForGetIndex.y = (nnz_ + SHARED_MEMORY_SIZE - 1) / SHARED_MEMORY_SIZE;
//        printf("    grid.x = %d, grid.y = %d\n", gridForGetIndex.x, gridForGetIndex.y);
//
//        dev::vector<UIN>
//            scatteredNumOfIndexPerWarp_4(NUMBER_OF_THREADS_PER_BLOCK * gridForGetIndex.y * gridForGetIndex.x);
//        dev::fill_n(scatteredNumOfIndexPerWarp_4.data(), scatteredNumOfIndexPerWarp_4.size(), 0);
//        timeCalculator.startClock();
//        getIndexPerWarp_4<<<gridForGetIndex, NUMBER_OF_THREADS_PER_BLOCK>>>(tensorCoreConfig,
//                                                                            numWarpX,
//                                                                            nnz_,
//                                                                            rowIndices_.data(),
//                                                                            colIndices_.data(),
//                                                                            updateScatteredNumOfIndexOperator_4(
//                                                                                scatteredNumOfIndexPerWarp_4.data()));
//        timeCalculator.endClock();
//        float updateScatteredNumOfIndexOperator_4_time = timeCalculator.getTime();
//        std::cout << "    updateScatteredNumOfIndexOperator_4_time : " << updateScatteredNumOfIndexOperator_4_time
//                  << " ms" << std::endl;
//
//        dev::vector<UIN> numIndexPerWarp_4(numWarps);
//        dev::fill_n(numIndexPerWarp_4.data(), numIndexPerWarp_4.size(), 0);
//        const int numBlockForMerge = (numWarps + NUMBER_OF_THREADS_PER_BLOCK - 1) / NUMBER_OF_THREADS_PER_BLOCK;
//        timeCalculator.startClock();
//        mergeScatteredNumOfIndex_4<<<numBlockForMerge, NUMBER_OF_THREADS_PER_BLOCK>>>(numWarps, gridForGetIndex.y,
//                                                                                      scatteredNumOfIndexPerWarp_4.data(),
//                                                                                      numIndexPerWarp_4.data());
//        timeCalculator.endClock();
//        float mergeNumOfIndexPerWarp_4_time = timeCalculator.getTime();
//        std::cout << "    mergeNumOfIndexPerWarp_4_time : " << mergeNumOfIndexPerWarp_4_time << " ms" << std::endl;
//
//        printf("check rightNum and numIndexPerWarp_4\n");
//        if (!checkData(rightNum, numIndexPerWarp_4)) {
//            exit(1);
//        }
//
//        printf(" @@@Method 4 time : %f\n", updateScatteredNumOfIndexOperator_4_time + mergeNumOfIndexPerWarp_4_time);
//    }

    ////////////////////////////////

//    std::cout << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;

//    // check
//    std::vector<UIN> rowIndices;
//    d2h(rowIndices, rowIndices_);
//    std::vector<UIN> colIndices;
//    d2h(colIndices, colIndices_);
//
//    std::set<std::pair<size_t, size_t>> rowColSet;
//    for (int idx = 0; idx < nnz_; ++idx) { // 检查是否有相同行列值
//        std::pair<size_t, size_t> rowColPair(rowIndices[idx], colIndices[idx]);
//        if (rowColSet.find(rowColPair) != rowColSet.end()) {
//            std::cout << " 有相同行列值1111???!!!!???!!! "
//                      << "idx = " << idx << ", "
//                      << rowIndices[idx] << " "
//                      << colIndices[idx]
//                      << std::endl;
//            exit(1);
//        }
//        rowColSet.insert(rowColPair);
//    }
//
//    std::vector<UIN> matrixTileMappedToWarpIndexData;
//    d2h(matrixTileMappedToWarpIndexData, matrixTileIndexData_);
//    for (int idx = 0; idx < matrixTileIndexData_.size(); ++idx) { // 检查是否出现不一样的值
//        std::pair<size_t, size_t> rowColPair(rowIndices[matrixTileIndexData[idx]], colIndices[matrixTileMappedToWarpIndexData[idx]]);
//        if (rowColSet.find(rowColPair) == rowColSet.end()) {
//            std::cout << " 出现不一样的值333???!!!!???!!! " << rowIndices[matrixTileMappedToWarpIndexData[idx]]
//                      << " " << colIndices[matrixTileMappedToWarpIndexData[idx]]
//                      << std::endl;
//            exit(1);
//        }
//    }

}

template<typename T>
void dev::SparseMatrix<T>::closeTensorCoreMode() {
    if (!tensorCoreMode_) {
        return;
    }
    tensorCoreMode_ = false;
    row_ = rowBeforeChange_;
    col_ = colBeforeChange_;

    matrixTileMappedToWarpIndex_.clear();
    matrixTileMappedToWarpIndexData_.clear();
}

template
class dev::Matrix<int>;
template
class dev::Matrix<float>;
template
class dev::Matrix<double>;
template
class dev::SparseMatrix<int>;
template
class dev::SparseMatrix<float>;
template
class dev::SparseMatrix<double>;