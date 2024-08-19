#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <set>

#include "Matrix.hpp"
#include "util.hpp"
#include "cudaUtil.cuh"
#include "wmmaSetting.hpp"

template<typename T>
Matrix<T>::Matrix(const SparseMatrix<T> &matrixS) {
    row_ = matrixS.row();
    col_ = matrixS.col();
    const int size = matrixS.row() * matrixS.col();
    size_ = size;
    storageOrder_ = MatrixStorageOrder::row_major;
    const int ld = matrixS.col();
    leadingDimension_ = ld;

    const auto &rowIndexS = matrixS.rowIndex();
    const auto &colIndexS = matrixS.colIndex();
    const auto &valuesS = matrixS.values();

    values_.clear();
    values_.resize(size);
    for (int idx = 0; idx < matrixS.nnz(); ++idx) {
        const int row = rowIndexS[idx];
        const int col = colIndexS[idx];
        const auto val = valuesS[idx];

        values_[row * ld + col] = val;
    }
}

template<typename T>
UIN Matrix<T>::rowOfValueIndex(UIN idx) const {
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
UIN Matrix<T>::colOfValueIndex(UIN idx) const {
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
bool Matrix<T>::initializeValue(const std::vector<T> &src) {
    if (src.size() != size_) {
        std::cerr << "Error! Matrix value size mismatch" << std::endl;
        return false;
    }
    values_ = src;
    return true;
}

template<typename T>
void Matrix<T>::changeStorageOrder() {
    const auto oldMajorOrder = storageOrder_;
    const auto oldLd = leadingDimension_;
    const auto &oldValues = values_;

    MatrixStorageOrder newMatrixOrder;
    UIN newLd;
    std::vector<T> newValues(size_);
    if (oldMajorOrder == MatrixStorageOrder::row_major) {
        newMatrixOrder = MatrixStorageOrder::col_major;
        newLd = row_;

        for (int idx = 0; idx < oldValues.size(); ++idx) {
            const int row = idx / oldLd;
            const int col = idx % oldLd;
            const auto val = oldValues[idx];

            newValues[col * newLd + row] = val;
        }
    } else {
        newMatrixOrder = MatrixStorageOrder::row_major;
        newLd = col_;

        for (int idx = 0; idx < values_.size(); ++idx) {
            const int col = idx / oldLd;
            const int row = idx % oldLd;
            const auto val = values_[idx];

            newValues[row * newLd + col] = val;
        }
    }

    storageOrder_ = newMatrixOrder;
    leadingDimension_ = newLd;
    values_ = newValues;
}

template<typename T>
void Matrix<T>::makeData(UIN numRow, UIN numCol, MatrixStorageOrder storageOrder) {
    row_ = numRow;
    col_ = numCol;
    size_ = numRow * numCol;
    storageOrder_ = storageOrder;
    if (storageOrder == MatrixStorageOrder::row_major) {
        leadingDimension_ = numCol;
    } else {
        leadingDimension_ = numRow;
    }
    values_.resize(size_);

    for (int idx = 0; idx < values_.size(); ++idx) {
        values_[idx] = idx + 1;
    }
//    std::mt19937 generator;
//    auto distribution = util::createRandomUniformDistribution(static_cast<T>(0), static_cast<T>(10));
//
//    for (int idx = 0; idx < values_.size(); ++idx) {
//        values_[idx] = distribution(generator);
//    }
}

template<typename T>
void Matrix<T>::print() const {
    for (auto iter : values_) {
        std::cout << iter << " ";
    }
    std::cout << std::endl;
}

template<typename T>
T Matrix<T>::getOneValueForMultiplication(MatrixMultiplicationOrder multiplicationOrder,
                                          UIN row,
                                          UIN col,
                                          UIN k) const {
    if (multiplicationOrder == MatrixMultiplicationOrder::left_multiplication) {
        if (storageOrder_ == MatrixStorageOrder::row_major) {
            return values_[row * leadingDimension_ + k];
        } else {
            return values_[k * leadingDimension_ + row];
        }
    } else {
        if (storageOrder_ == MatrixStorageOrder::row_major) {
            return values_[k * leadingDimension_ + col];
        } else {
            return values_[col * leadingDimension_ + k];
        }
    }
}

template<typename T>
T Matrix<T>::getOneValue(int row, int col) const {
    if (storageOrder_ == MatrixStorageOrder::row_major) {
        return values_[row * leadingDimension_ + col];
    } else {
        return values_[col * leadingDimension_ + row];
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

    UIN rowComplement;
    UIN colComplement;
    if (multiplicationOrder == MatrixMultiplicationOrder::left_multiplication) {
        rowComplement = WMMA_M - rowBeforeChange_ % WMMA_M;
        colComplement = WMMA_K - colBeforeChange_ % WMMA_K;
    } else {
        rowComplement = WMMA_K - rowBeforeChange_ % WMMA_K;
        colComplement = WMMA_N - colBeforeChange_ % WMMA_N;
    }

    row_ = rowBeforeChange_ + rowComplement;
    col_ = colBeforeChange_ + colComplement;
    size_ = row_ * col_;

    if (storageOrder_ == MatrixStorageOrder::row_major) {
        for (int rowIter = 0; rowIter < rowBeforeChange_; ++rowIter) {
            values_.insert(values_.begin() + rowIter * row_ + colBeforeChange_, colComplement, 0);
        }
        values_.insert(values_.end() - 1, rowComplement * col_, 0);
    } else {
        for (int colIter = 0; colIter < colBeforeChange_; ++colIter) {
            values_.insert(values_.begin() + colIter * col_ + rowBeforeChange_, rowComplement, 0);
        }
        values_.insert(values_.end() - 1, colComplement * row_, 0);
    }
}

template<typename T>
void Matrix<T>::closeTensorCoreMode() {
    if (!tensorCoreMode_) {
        return;
    }
    tensorCoreMode_ = false;

    const UIN rowComplement = abs(row_ - rowBeforeChange_);
    const UIN colComplement = abs(col_ - colBeforeChange_);

    if (storageOrder_ == MatrixStorageOrder::row_major) {
        for (int rowIter = 0; rowIter < rowBeforeChange_; ++rowIter) {
            const auto curRowBeginIter = values_.begin() + rowIter * colBeforeChange_ + colBeforeChange_;
            values_.erase(curRowBeginIter, curRowBeginIter + colComplement);
        }
    } else {
        for (int colIter = 0; colIter < colBeforeChange_; ++colIter) {
            const auto curColBeginIter = values_.begin() + colIter * rowBeforeChange_ + rowBeforeChange_;
            values_.erase(curColBeginIter, curColBeginIter + rowComplement);
        }
    }
    row_ = rowBeforeChange_;
    col_ = colBeforeChange_;
    size_ = row_ * col_;
    values_.resize(size_);
}

template<typename T>
void SparseMatrix<T>::print() const {
    std::cout << "SparseMatrix : [row,col,value]" << std::endl;
    for (int idx = 0; idx < nnz_; ++idx) {
        std::cout << "[" << rowIndex_[idx] << ","
                  << colIndex_[idx] << ","
                  << values_[idx] << "] ";
    }
    std::cout << std::endl;
}

template<typename T>
bool SparseMatrix<T>::setValuesFromMatrix(const Matrix<T> &inputMatrix) {
    values_.clear();
    values_.resize(nnz_);

    for (int idx = 0; idx < nnz_; ++idx) {
        const int row = rowIndex_[idx];
        const int col = colIndex_[idx];

        values_[idx] = inputMatrix.getOneValue(row, col);
    }

    return true;
}

template<typename T>
SparseMatrix<T>::SparseMatrix(const std::string &filePath) {
    std::ifstream inFile;
    inFile.open(filePath, std::ios::in); // open file
    if (!inFile.is_open()) {
        std::cerr << "Error, MatrixMarket file cannot be opened : " << filePath << std::endl;
        return;
    }

    std::cout << "SparseMatrix initialize From MatrixMarket file : " << filePath << std::endl;

    std::string line; // Store the data for each line
    getline(inFile, line); // First line does not operate

    getline(inFile, line);
    int wordIter = 0;
    row_ = std::stoi(util::iterateOneWordFromLine(line, wordIter));
    col_ = std::stoi(util::iterateOneWordFromLine(line, wordIter));
    nnz_ = std::stoi(util::iterateOneWordFromLine(line, wordIter));

    if (wordIter < line.size()) {
        std::cerr << "Error, Matrix Market file " << line << " line format is incorrect!" << std::endl;
    }

    rowIndex_.resize(nnz_);
    colIndex_.resize(nnz_);
    values_.resize(nnz_);

    int idx = 0;
    while (getline(inFile, line)) {
        wordIter = 0;
        const int row = std::stoi(util::iterateOneWordFromLine(line, wordIter)) - 1;
        const int col = std::stoi(util::iterateOneWordFromLine(line, wordIter)) - 1;
        const T val = static_cast<T>(std::stod(util::iterateOneWordFromLine(line, wordIter)));

        if (wordIter < line.size()) {
            std::cerr << "Error, Matrix Market file " << line << " line format is incorrect!" << std::endl;
        }

        rowIndex_[idx] = row;
        colIndex_[idx] = col;
        values_[idx] = val;

        ++idx;
    }

    inFile.close();
}

template<typename T>
void SparseMatrix<T>::getSpareMatrixOneDataByCOO(const int idx, UIN &row, UIN &col, T &value) const {
    row = rowIndex_[idx];
    col = colIndex_[idx];
    value = values_[idx];
}

template<typename T>
bool SparseMatrix<T>::outputToMarketMatrixFile(const std::string &fileName) {
    const std::string fileFormat(".mtx");

    std::string fileString(fileName + fileFormat);

    // check fileExists
    if (util::io::fileExists(fileString)) {
        std::cout << fileName + fileFormat << " file already exists" << std::endl;
        int fileId = 1;
        while (util::io::fileExists(fileName + "_" + std::to_string(fileId) + fileFormat)) {
            ++fileId;
        }
        fileString = fileName + "_" + std::to_string(fileId) + fileFormat;

        std::cout << "Change file name form \"" << fileName + fileFormat
                  << "\" to \""
                  << fileString << "\"" << std::endl;
    }

    // creat file
    std::ofstream outfile(fileString);
    if (outfile.is_open()) {
        std::cout << "File created successfully: " << fileString << std::endl;
    } else {
        std::cerr << "Unable to create file: " << fileString << std::endl;
        return false;
    }

    std::string firstLine("%%MatrixMarket matrix coordinate real general\n");
    outfile << firstLine;

    std::string secondLine(std::to_string(row_) + " " + std::to_string(col_) + " " + std::to_string(nnz_) + "\n");
    outfile << secondLine;

    for (int idx = 0; idx < nnz_; ++idx) {
        outfile << std::to_string(rowIndex_[idx] + 1) << " ";
        outfile << std::to_string(colIndex_[idx] + 1) << " ";
        outfile << std::to_string(values_[idx]);

        if (idx < nnz_ - 1) {
            outfile << "\n";
        }
    }

    outfile.close();
    return true;
}

template<typename T>
void SparseMatrix<T>::makeData(const UIN numRow, const UIN numCol, const UIN nnz) {
    row_ = numRow;
    col_ = numCol;
    nnz_ = nnz;

    rowIndex_.resize(nnz);
    colIndex_.resize(nnz);
    values_.resize(nnz);

    // make data
    std::mt19937 generator;
    auto distributionRow = util::createRandomUniformDistribution(static_cast<UIN>(0), static_cast<UIN>(10));
    auto distributionCol = util::createRandomUniformDistribution(static_cast<UIN>(0), static_cast<UIN>(10));
    auto distributionValue = util::createRandomUniformDistribution(static_cast<T>(0), static_cast<T>(10));
    std::set<std::pair<UIN, UIN>> rowColSet;
    for (UIN idx = 0; idx < nnz; ++idx) {
        UIN row = distributionRow(generator);
        UIN col = distributionCol(generator);
        std::pair<UIN, UIN> rowColPair(row, col);
        auto findSet = rowColSet.find(rowColPair);
        while (findSet != rowColSet.end()) {
            row = distributionRow(generator);
            col = distributionCol(generator);
            rowColPair.first = row;
            rowColPair.second = col;
            findSet = rowColSet.find(rowColPair);
        }

        rowColSet.insert(rowColPair);

        rowIndex_[idx] = row;
        colIndex_[idx] = col;
        values_[idx] = distributionValue(generator);
    }

    // sort rowIndex and colIndex
    cuUtil::host::sort_by_key(rowIndex_.data(), rowIndex_.data() + rowIndex_.size(), colIndex_.data());
    UIN lastRowNumber = rowIndex_[0];
    UIN lastBegin = 0;
    for (UIN idx = 0; idx < nnz_; ++idx) {
        const UIN curRowNumber = rowIndex_[idx];
        if (curRowNumber != lastRowNumber) { // new row
            cuUtil::host::sort(colIndex_.data() + lastBegin, colIndex_.data() + idx);

            lastBegin = idx + 1;
            lastRowNumber = curRowNumber;
        }

        if (idx == nnz_ - 1) {
            cuUtil::host::sort(colIndex_.data() + lastBegin, colIndex_.data() + colIndex_.size());
        }
    }
}

template<typename T>
void SparseMatrix<T>::openTensorCoreMode(MatrixMultiplicationOrder multiplicationOrder) {
    if (tensorCoreMode_) {
        return;
    }
    tensorCoreMode_ = true;
    rowBeforeChange_ = row_;
    colBeforeChange_ = col_;

    UIN rowComplement;
    UIN colComplement;
    if (multiplicationOrder == MatrixMultiplicationOrder::left_multiplication) {
        rowComplement = WMMA_M - rowBeforeChange_ % WMMA_M;
        colComplement = WMMA_K - colBeforeChange_ % WMMA_K;
    } else {
        rowComplement = WMMA_K - rowBeforeChange_ % WMMA_K;
        colComplement = WMMA_N - colBeforeChange_ % WMMA_N;
    }
    row_ = rowBeforeChange_ + rowComplement;
    col_ = colBeforeChange_ + colComplement;
}

template<typename T>
void SparseMatrix<T>::openTensorCoreModeForSampled() {
    if (tensorCoreMode_) {
        return;
    }
    tensorCoreMode_ = true;
    rowBeforeChange_ = row_;
    colBeforeChange_ = col_;

    const UIN rowComplement = WMMA_M - rowBeforeChange_ % WMMA_M;
    const UIN colComplement = WMMA_N - colBeforeChange_ % WMMA_N;
    row_ = rowBeforeChange_ + rowComplement;
    col_ = colBeforeChange_ + colComplement;
}

template<typename T>
void SparseMatrix<T>::closeTensorCoreMode() {
    if (!tensorCoreMode_) {
        return;
    }
    tensorCoreMode_ = false;
    row_ = rowBeforeChange_;
    col_ = colBeforeChange_;
}

template
class Matrix<int>;
template
class Matrix<float>;
template
class Matrix<double>;

template
class SparseMatrix<int>;
template
class SparseMatrix<float>;
template
class SparseMatrix<double>;