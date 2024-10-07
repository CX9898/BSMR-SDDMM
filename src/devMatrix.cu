#include <cuda_runtime.h>

#include <fstream>

#include "devMatrix.cuh"
#include "devMatrixKernel.cuh"
#include "parallelAlgorithm.cuh"
//#include "util.hpp"
#include "devVector.cuh"
#include "CudaTimeCalculator.cuh"
#include "checkData.hpp"

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
    if (tensorCoreMode_) {
        return;
    }
    tensorCoreMode_ = true;
    rowBeforeChange_ = row_;
    colBeforeChange_ = col_;

    size_t rowComplement = 0;
    size_t colComplement = 0;
    if (multiplicationOrder == MatrixMultiplicationOrder::left_multiplication) {
        rowComplement = rowBeforeChange_ % WMMA_M == 0 ? 0 : WMMA_M - rowBeforeChange_ % WMMA_M;
        colComplement = colBeforeChange_ % WMMA_K == 0 ? 0 : WMMA_K - colBeforeChange_ % WMMA_K;
    } else {
        rowComplement = rowBeforeChange_ % WMMA_K == 0 ? 0 : WMMA_K - rowBeforeChange_ % WMMA_K;
        colComplement = colBeforeChange_ % WMMA_N == 0 ? 0 : WMMA_N - colBeforeChange_ % WMMA_N;
    }

    row_ = rowBeforeChange_ + rowComplement;
    col_ = colBeforeChange_ + colComplement;
    if (storageOrder_ == MatrixStorageOrder::row_major) {
        leadingDimension_ = col_;
    } else {
        leadingDimension_ = row_;
    }

    if (storageOrder_ == MatrixStorageOrder::row_major) {
        for (size_t rowIter = 0; rowIter < rowBeforeChange_; ++rowIter) {
//            values_.insert(values_.begin() + rowIter * leadingDimension_ + colBeforeChange_, colComplement, 0);
        }
//        values_.insert(values_.end(), rowComplement * col_, 0);
    } else {
        for (size_t colIter = 0; colIter < colBeforeChange_; ++colIter) {
//            values_.insert(values_.begin() + colIter * leadingDimension_ + rowBeforeChange_, rowComplement, 0);
        }
//        values_.insert(values_.end(), colComplement * row_, 0);
    }
}

template<typename T>
void dev::Matrix<T>::closeTensorCoreMode() {
    if (!tensorCoreMode_) {
        return;
    }
    tensorCoreMode_ = false;

    const size_t rowComplement = row_ < rowBeforeChange_ ? rowBeforeChange_ - row_ : row_ - rowBeforeChange_;
    const size_t colComplement = col_ < colBeforeChange_ ? colBeforeChange_ - col_ : col_ - colBeforeChange_;

    row_ = rowBeforeChange_;
    col_ = colBeforeChange_;
    if (storageOrder_ == MatrixStorageOrder::row_major) {
        leadingDimension_ = col_;
    } else {
        leadingDimension_ = row_;
    }
    if (storageOrder_ == MatrixStorageOrder::row_major) {
        for (size_t rowIter = 0; rowIter < rowBeforeChange_; ++rowIter) {
            const auto curRowBeginIter = values_.begin() + rowIter * leadingDimension_ + colBeforeChange_;
//            values_.erase(curRowBeginIter, curRowBeginIter + colComplement);
        }
    } else {
        for (size_t colIter = 0; colIter < colBeforeChange_; ++colIter) {
            const auto curColBeginIter = values_.begin() + colIter * leadingDimension_ + rowBeforeChange_;
//            values_.erase(curColBeginIter, curColBeginIter + rowComplement);
        }
    }
    values_.resize(row_ * col_);
}

template<typename T>
void dev::SparseMatrix<T>::setValuesFromDenseData(UIN row, UIN col, UIN ld, const dev::vector<T> &denseData) {
    if (row < row_ || col < col_) {
        std::cerr << "row < row_ || col < col_" << std::endl;
    }
    values_.resize(nnz_);

    const int numThreads = 1024;
    const int numBlocks = (nnz_ + numThreads - 1) / numThreads;
    getValuesFromDenseData<<<numBlocks, numThreads>>>(row,
                                                      col,
                                                      nnz_,
                                                      ld,
                                                      rowIndex_.data(),
                                                      colIndex_.data(),
                                                      denseData.data(),
                                                      values_.data());
    cudaDeviceSynchronize();
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
//    std::vector<UIN> rowIndex(nnz_);
//    std::vector<UIN> colIndex(nnz_);
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
//        rowIndex[idx] = row;
//        colIndex[idx] = col;
//        values[idx] = val;
//
//        ++idx;
//    }
//
//    inFile.close();
//
//    h2d(rowIndex_,rowIndex);
//    h2d(colIndex_,colIndex);
//    h2d(values_,values);
//
//    rowBeforeChange_ = row_;
//    colBeforeChange_ = col_;
//}

template<typename T>
void dev::SparseMatrix<T>::openTensorCoreModeForSampled(TensorCoreConfig tensorCoreConfig) {
    if (tensorCoreMode_) {
        return;
    }
    tensorCoreMode_ = true;
    rowBeforeChange_ = row_;
    colBeforeChange_ = col_;

    const int rowComplement = rowBeforeChange_ % WMMA_M == 0 ? 0 : WMMA_M - rowBeforeChange_ % WMMA_M;
    const int colComplement = colBeforeChange_ % WMMA_N == 0 ? 0 : WMMA_N - colBeforeChange_ % WMMA_N;
    row_ = rowBeforeChange_ + rowComplement;
    col_ = colBeforeChange_ + colComplement;

    const UIN numTileM = row_ / WMMA_M;
    const UIN numTileN = col_ / WMMA_N;

    const UIN numWarpX = tensorCoreConfig.numWarpX();
    const UIN numWarpY = tensorCoreConfig.numWarpY();
    const UIN numWarps = numWarpX * numWarpY;

    const UIN numThreadsPerBlock = NUMBER_OF_THREADS_PER_BLOCK;
    const UIN numBlocks = (numWarps + numThreadsPerBlock - 1) / numThreadsPerBlock;
    CudaTimeCalculator timeCalculator;


    //////////////////////////////// 1
    dev::vector<UIN> numOfIndexPerWarp(numWarps);
    dev::fill_n(numOfIndexPerWarp.data(), numOfIndexPerWarp.size(), 0);

    timeCalculator.startClock();
    getIndexPerWarp_1<<<numBlocks, NUMBER_OF_THREADS_PER_BLOCK>>>(numWarps,
                                                                  numWarpX,
                                                                  numTileM,
                                                                  numTileN,
                                                                  nnz_,
                                                                  rowIndex_.data(),
                                                                  colIndex_.data(),
                                                                  updateNumOfIndexOperator_1(numOfIndexPerWarp.data()));
    timeCalculator.endClock();
    float getNumIndexPerWarp_1_time = timeCalculator.getTime();
    std::cout << "  getNumIndexPerWarp_1_time : " << getNumIndexPerWarp_1_time << " ms" << std::endl;
    std::vector<UIN> rightNum;
    d2h(rightNum, numOfIndexPerWarp);

    timeCalculator.startClock();

    matrixTileMappedToWarpIndex_.resize(numWarps + 1);
    dev::fill_n(matrixTileMappedToWarpIndex_.data(), 1, 0);
    dev::inclusive_scan(numOfIndexPerWarp.data(),
                        numOfIndexPerWarp.data() + numOfIndexPerWarp.size(),
                        matrixTileMappedToWarpIndex_.data() + 1);
    const UIN numIndexData = matrixTileMappedToWarpIndex_.back_data();
    timeCalculator.endClock();
    float inclusive_scan_time = timeCalculator.getTime();
    std::cout << "  inclusive_scan_time : " << inclusive_scan_time << " ms" << std::endl;

    matrixTileMappedToWarpIndexData_.resize(numIndexData);

    timeCalculator.startClock();
    getIndexPerWarp_1<<<numBlocks, numThreadsPerBlock>>>(numWarps,
                                                         numWarpX,
                                                         numTileM,
                                                         numTileN,
                                                         nnz_,
                                                         rowIndex_.data(),
                                                         colIndex_.data(),
                                                         updateIndexDataPerWarpOperator_1(matrixTileMappedToWarpIndex_.data(),
                                                                                          matrixTileMappedToWarpIndexData_.data()));
    timeCalculator.endClock();
    float getTileIndexDataPerWarp_time = timeCalculator.getTime();
    std::cout << "  getTileIndexDataPerWarp_time : " << getTileIndexDataPerWarp_time << " ms" << std::endl;

    printf(" @@@Method 1 time : %f\n",
           getNumIndexPerWarp_1_time + inclusive_scan_time + getTileIndexDataPerWarp_time);


    //////////////////////////////// 2 error
    {
        dev::vector<UIN> numIndexOfPerWarp_2(numWarps);
        dev::fill_n(numIndexOfPerWarp_2.data(), numIndexOfPerWarp_2.size(), 0);
        timeCalculator.startClock();
        getIndexPerWarp_2<<<numBlocks, numThreadsPerBlock>>>(numWarps,
                                                             numWarpX,
                                                             numTileM,
                                                             numTileN,
                                                             nnz_,
                                                             rowIndex_.data(),
                                                             colIndex_.data(),
                                                             updateNumOfIndexOperator_2(numIndexOfPerWarp_2.data()));

        timeCalculator.endClock();
        float getIndexPerWarp_2_time = timeCalculator.getTime();
        std::cout << "  getIndexPerWarp_2_time : " << getIndexPerWarp_2_time << " ms" << std::endl;

        printf("  check rightNum and numIndexOfPerWarp_2\n");
        const int indexNum = 0;
//    printf("    rightNum[%d] = %d, num2[%d] = %d\n", indexNum, rightNum[indexNum], indexNum, num2[indexNum]);
//        if (!checkData(rightNum, numIndexOfPerWarp_2)) {
//        exit(1);
//        }

        timeCalculator.startClock();

        dev::vector<UIN> matrixTileMappedToWarpIndex_2(numWarps + 1);
        dev::fill_n(matrixTileMappedToWarpIndex_2.data(), 1, 0);
        dev::inclusive_scan(numIndexOfPerWarp_2.data(),
                            numIndexOfPerWarp_2.data() + numIndexOfPerWarp_2.size(),
                            matrixTileMappedToWarpIndex_2.data() + 1);
        const UIN numIndexData_2 = matrixTileMappedToWarpIndex_2.back_data();
        timeCalculator.endClock();
        float inclusive_scan_2_time = timeCalculator.getTime();
        std::cout << "  inclusive_scan_2_time : " << inclusive_scan_2_time << " ms" << std::endl;

        dev::vector<UIN> matrixTileMappedToWarpIndexData_2(numIndexData_2);
        timeCalculator.startClock();
        getIndexPerWarp_2<<<numBlocks, numThreadsPerBlock>>>(numWarps,
                                                             numWarpX,
                                                             numTileM,
                                                             numTileN,
                                                             nnz_,
                                                             rowIndex_.data(),
                                                             colIndex_.data(),
                                                             updateIndexDataPerWarpOperator_2(
                                                                 matrixTileMappedToWarpIndex_2.data(),
                                                                 matrixTileMappedToWarpIndexData_2.data()));

        timeCalculator.endClock();
        float getIndexPerWarp_2_2_time = timeCalculator.getTime();
        std::cout << "  getIndexPerWarp_2_2_time : " << getIndexPerWarp_2_2_time << " ms" << std::endl;

        printf(" @@@Method 2 time : %f\n",
               getIndexPerWarp_2_time + inclusive_scan_2_time + getIndexPerWarp_2_2_time);

//        d2d(matrixTileMappedToWarpIndex_, matrixTileMappedToWarpIndex_2);
//        d2d(matrixTileMappedToWarpIndexData_, matrixTileMappedToWarpIndexData_2);
    }

    //////////////////////////////// 3 OK
    {
        dim3 gridForGetIndex;
        gridForGetIndex.x = (numWarps + NUMBER_OF_THREADS_PER_BLOCK - 1) / NUMBER_OF_THREADS_PER_BLOCK;
        gridForGetIndex.y = (nnz_ + SHARED_MEMORY_SIZE - 1) / SHARED_MEMORY_SIZE;
        dev::vector<UIN> scatteredNumOfIndexPerWarp_3(nnz_ * gridForGetIndex.y);
        dev::fill_n(scatteredNumOfIndexPerWarp_3.data(), scatteredNumOfIndexPerWarp_3.size(), 0);
        timeCalculator.startClock();
        getIndexPerWarp_3<<<gridForGetIndex, NUMBER_OF_THREADS_PER_BLOCK>>>(numWarpX,
                                                                            nnz_,
                                                                            rowIndex_.data(),
                                                                            colIndex_.data(),
                                                                            updateScatteredNumOfIndexOperator_3(
                                                                                scatteredNumOfIndexPerWarp_3.data()));
        timeCalculator.endClock();
        float updateScatteredNumOfIndexOperator_time = timeCalculator.getTime();
        std::cout << "    updateScatteredNumOfIndexOperator_time : " << updateScatteredNumOfIndexOperator_time << " ms"
                  << std::endl;

        timeCalculator.startClock();

        dev::vector<UIN> indexForScatteredNumOfIndex(scatteredNumOfIndexPerWarp_3.size() + 1);
        dev::fill_n(indexForScatteredNumOfIndex.data(), 1, 0);
        dev::inclusive_scan(scatteredNumOfIndexPerWarp_3.data(),
                            scatteredNumOfIndexPerWarp_3.data() + scatteredNumOfIndexPerWarp_3.size(),
                            indexForScatteredNumOfIndex.data() + 1);
        const UIN scatteredNumIndexData = indexForScatteredNumOfIndex.back_data();

        timeCalculator.endClock();
        float inclusive_scan_scattered_time = timeCalculator.getTime();
        std::cout << "    inclusive_scan_scattered_time : " << inclusive_scan_scattered_time
                  << " ms" << std::endl;

        dev::vector<UIN> scatteredMatrixTileMappedToWarpIndexData(scatteredNumIndexData);
        timeCalculator.startClock();

        getIndexPerWarp_3<<<gridForGetIndex, NUMBER_OF_THREADS_PER_BLOCK>>>(numWarpX,
                                                                            nnz_,
                                                                            rowIndex_.data(),
                                                                            colIndex_.data(),
                                                                            updateScatteredIndexDataPerWarpOperator_3(
                                                                                indexForScatteredNumOfIndex.data(),
                                                                                scatteredMatrixTileMappedToWarpIndexData.data()));
        timeCalculator.endClock();
        float updateScatteredIndexDataPerWarpOperator_time = timeCalculator.getTime();
        std::cout << "    updateScatteredIndexDataPerWarpOperator_time : "
                  << updateScatteredIndexDataPerWarpOperator_time
                  << " ms" << std::endl;

        dev::vector<UIN> numIndexPerWarp_3_sum(numWarps);
        dev::fill_n(numIndexPerWarp_3_sum.data(), numIndexPerWarp_3_sum.size(), 0);
        timeCalculator.startClock();
        mergeScatteredNumOfIndex<<<gridForGetIndex.x, NUMBER_OF_THREADS_PER_BLOCK>>>(numWarps, gridForGetIndex.y,
                                                                                     scatteredNumOfIndexPerWarp_3.data(),
                                                                                     numIndexPerWarp_3_sum.data());
        timeCalculator.endClock();
        float mergeNumOfIndexPerWarp_time = timeCalculator.getTime();
        std::cout << "    mergeNumOfIndexPerWarp_time : " << mergeNumOfIndexPerWarp_time << " ms" << std::endl;

        printf("check rightNum and numIndexPerWarp_3_sum\n");
        if (!checkData(rightNum, numIndexPerWarp_3_sum)) {
            exit(1);
        }
        timeCalculator.startClock();

        dev::vector<UIN> matrixTileMappedToWarpIndex_3(numWarps + 1);
        dev::fill_n(matrixTileMappedToWarpIndex_3.data(), 1, 0);
        dev::inclusive_scan(numIndexPerWarp_3_sum.data(),
                            numIndexPerWarp_3_sum.data() + numIndexPerWarp_3_sum.size(),
                            matrixTileMappedToWarpIndex_3.data() + 1);
        const UIN numIndexData_3 = matrixTileMappedToWarpIndex_3.back_data();

        dev::vector<UIN> matrixTileMappedToWarpIndexData_3(numIndexData_3);
        timeCalculator.endClock();
        float inclusive_scan_3_time = timeCalculator.getTime();
        std::cout << "    inclusive_scan_3_time : " << inclusive_scan_3_time
                  << " ms" << std::endl;

        timeCalculator.startClock();
        sortScatteredIndexData<<<gridForGetIndex.x, NUMBER_OF_THREADS_PER_BLOCK>>>(numWarps,
                                                                                   gridForGetIndex.y,
                                                                                   matrixTileMappedToWarpIndex_3.data(),
                                                                                   indexForScatteredNumOfIndex.data(),
                                                                                   scatteredMatrixTileMappedToWarpIndexData.data(),
                                                                                   matrixTileMappedToWarpIndexData_3.data());
        timeCalculator.endClock();
        float sortScatteredIndexData_time = timeCalculator.getTime();
        std::cout << "    sortScatteredIndexData_time : " << sortScatteredIndexData_time << " ms" << std::endl;

        printf(" @@@Method 3 time : %f\n",
               updateScatteredNumOfIndexOperator_time + inclusive_scan_scattered_time
                   + updateScatteredIndexDataPerWarpOperator_time + mergeNumOfIndexPerWarp_time + inclusive_scan_3_time
                   + sortScatteredIndexData_time);

//        d2d(matrixTileMappedToWarpIndex_, matrixTileMappedToWarpIndex_3);
//        d2d(matrixTileMappedToWarpIndexData_, matrixTileMappedToWarpIndexData_3);
    }
    ////////////////////////////////

//    std::cout << cudaGetErrorString(cudaDeviceSynchronize()) << std::endl;

//    // check
//    std::vector<UIN> rowIndex;
//    d2h(rowIndex, rowIndex_);
//    std::vector<UIN> colIndex;
//    d2h(colIndex, colIndex_);
//
//    std::set<std::pair<size_t, size_t>> rowColSet;
//    for (int idx = 0; idx < nnz_; ++idx) { // 检查是否有相同行列值
//        std::pair<size_t, size_t> rowColPair(rowIndex[idx], colIndex[idx]);
//        if (rowColSet.find(rowColPair) != rowColSet.end()) {
//            std::cout << " 有相同行列值1111???!!!!???!!! "
//                      << "idx = " << idx << ", "
//                      << rowIndex[idx] << " "
//                      << colIndex[idx]
//                      << std::endl;
//            exit(1);
//        }
//        rowColSet.insert(rowColPair);
//    }
//
//    std::vector<UIN> matrixTileIndexData;
//    d2h(matrixTileIndexData, matrixTileIndexData_);
//    for (int idx = 0; idx < matrixTileIndexData_.size(); ++idx) { // 检查是否出现不一样的值
//        std::pair<size_t, size_t> rowColPair(rowIndex[matrixTileIndexData[idx]], colIndex[matrixTileIndexData[idx]]);
//        if (rowColSet.find(rowColPair) == rowColSet.end()) {
//            std::cout << " 出现不一样的值333???!!!!???!!! " << rowIndex[matrixTileIndexData[idx]]
//                      << " " << colIndex[matrixTileIndexData[idx]]
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