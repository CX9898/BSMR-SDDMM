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

    dev::vector<UIN> numIndexPerWarp(numWarps);
    dev::fill_n(numIndexPerWarp.data(), numWarps, 0);
    const UIN numThreadsPerBlock = NumberOfThreadsPerBlock;
    const UIN numBlocks = (numWarps + numThreadsPerBlock - 1) / numThreadsPerBlock;
    CudaTimeCalculator timeCalculator;


    //////////////////////////////// 1
    timeCalculator.startClock();
    getNumIndexPerWarp_1<<<numBlocks, NumberOfThreadsPerBlock>>>(numWarps,
                                                                 numWarpX,
                                                                 numTileM,
                                                                 numTileN,
                                                                 nnz_,
                                                                 rowIndex_.data(),
                                                                 colIndex_.data(),
                                                                 numIndexPerWarp.data());
    timeCalculator.endClock();
    float getNumIndexPerWarp_1_time = timeCalculator.getTime();
    std::cout << "  getNumIndexPerWarp_1_time : " << getNumIndexPerWarp_1_time << " ms" << std::endl;
    std::vector<UIN> num1;
    d2h(num1, numIndexPerWarp);

    //////////////////////////////// 2
    timeCalculator.startClock();
    getNumIndexPerWarp_2<<<numBlocks, numThreadsPerBlock>>>(numWarps,
                                                            numWarpX,
                                                            numTileM,
                                                            numTileN,
                                                            nnz_,
                                                            rowIndex_.data(),
                                                            colIndex_.data(),
                                                            numIndexPerWarp.data());

    timeCalculator.endClock();
    float getNumIndexPerWarp_2_time = timeCalculator.getTime();
    std::cout << "  getNumIndexPerWarp_time_2 : " << getNumIndexPerWarp_2_time << " ms" << std::endl;

    std::vector<UIN> num2;
    d2h(num2, numIndexPerWarp);

    std::cout << "  check num1, num2" << std::endl;
    const int indexNum = 0;
    printf("num1[%d] = %d, num2[%d] = %d\n", indexNum, num1[indexNum], indexNum, num2[indexNum]);
    if (!checkData(num1, num2)) {
        exit(1);
    }

    //////////////////////////////// 3

    dim3 grid;
    grid.x = (numWarps + numThreadsPerBlock - 1) / numThreadsPerBlock;
    grid.y = (nnz_ + SharedMemorySize - 1) / SharedMemorySize;
    timeCalculator.startClock();
    // TODO : getNumIndexPerWarp_3()
    dev::vector<UIN> numIndexPerWarp_3(nnz_ * 1111111111111);
    getNumIndexPerWarp_3<<<grid, numThreadsPerBlock>>>(numWarpX,
                                                       nnz_,
                                                       rowIndex_.data(),
                                                       colIndex_.data(),
                                                       numIndexPerWarp_3.data());
    timeCalculator.endClock();
    float getNumIndexPerWarp_3_time = timeCalculator.getTime();
    std::cout << "  getNumIndexPerWarp_3_time : " << getNumIndexPerWarp_3_time << " ms" << std::endl;

    matrixTileMappedToWarpIndex_.resize(numWarps + 1);

    timeCalculator.startClock();

    dev::fill_n(matrixTileMappedToWarpIndex_.data(), 1, 0);
    dev::inclusive_scan(numIndexPerWarp.data(),
                        numIndexPerWarp.data() + numIndexPerWarp.size(),
                        matrixTileMappedToWarpIndex_.data() + 1);
    const UIN numIndexData = matrixTileMappedToWarpIndex_.back_data();
    timeCalculator.endClock();
    float inclusive_scan_time = timeCalculator.getTime();
    std::cout << "  inclusive_scan_time : " << inclusive_scan_time << " ms" << std::endl;

    matrixTileMappedToWarpIndexData_.resize(numIndexData);

    timeCalculator.startClock();
    getTileIndexDataPerWarp_2<<<numBlocks, numThreadsPerBlock>>>(numWarps,
                                                                 numWarpX,
                                                                 numTileM,
                                                                 numTileN,
                                                                 nnz_,
                                                                 rowIndex_.data(),
                                                                 colIndex_.data(),
                                                                 matrixTileMappedToWarpIndex_.data(),
                                                                 matrixTileMappedToWarpIndexData_.data());
    timeCalculator.endClock();
    float getTileIndexDataPerWarp_time = timeCalculator.getTime();
    std::cout << "  getTileIndexDataPerWarp_time : " << getTileIndexDataPerWarp_time << " ms" << std::endl;
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