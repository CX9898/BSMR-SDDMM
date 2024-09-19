#include <cuda_runtime.h>

#include <fstream>

#include "devMatrix.cuh"
#include "devMatrixKernel.cuh"
#include "parallelAlgorithm.cuh"
//#include "util.hpp"
#include "devVector.cuh"

//#include <set>

//namespace dev {
template<typename T>
size_t dev::Matrix<T>::rowOfValueIndex(size_t idx) const {
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
size_t dev::Matrix<T>::colOfValueIndex(size_t idx) const {
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
void Matrix<T>::openTensorCoreMode(MatrixMultiplicationOrder multiplicationOrder) {
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
void Matrix<T>::closeTensorCoreMode() {
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
    const UIN numThreadsPerBlock = 1024;
    const UIN numBlocks = (numWarps + numThreadsPerBlock - 1) / numThreadsPerBlock;
    getNumIndexPerWarp<<<numBlocks, numThreadsPerBlock>>>(numWarps,
                                                          numWarpX,
                                                          numTileM,
                                                          numTileN,
                                                          nnz_,
                                                          rowIndex_.data(),
                                                          colIndex_.data(),
                                                          numIndexPerWarp.data());
    matrixTileIndex_.resize(numWarps + 1);
    dev::fill_n(matrixTileIndex_.data(), 1, 0);
    dev::inclusive_scan(numIndexPerWarp.data(),
                        numIndexPerWarp.data() + numIndexPerWarp.size(),
                        matrixTileIndex_.data() + 1);
    const UIN numIndexData = matrixTileIndex_.back_data();
    matrixTileIndexData_.resize(numIndexData);
    getTileIndexDataPerWarp<<<numBlocks, numThreadsPerBlock>>>(numWarps,
                                                               numWarpX,
                                                               numTileM,
                                                               numTileN,
                                                               nnz_,
                                                               rowIndex_.data(),
                                                               colIndex_.data(),
                                                               matrixTileIndex_.data(),
                                                               matrixTileIndexData_.data());

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

    matrixTileIndex_.clear();
    matrixTileIndexData_.clear();
}

template class dev::Matrix<int>;
template class dev::Matrix<float>;
template class dev::Matrix<double>;
template class dev::SparseMatrix<int>;
template class dev::SparseMatrix<float>;
template class dev::SparseMatrix<double>;
//} // namespace dev