#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <set>

#include "Matrix.hpp"
#include "util.hpp"
#include "cudaUtil.cuh"

template<typename T>
bool SparseMatrix<T>::initializeFromMatrixMarketFile(const std::string &filePath) {
    std::ifstream inFile;
    inFile.open(filePath, std::ios::in); // open file
    if (!inFile.is_open()) {
        std::cerr << "Error, Matrix Market file cannot be opened : " << filePath << std::endl;
        return false;
    }

    std::cout << "SparseMatrix initialize From MatrixMarketFile : " << filePath << std::endl;

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

    return true;
}

template<typename T>
bool Matrix<T>::initializeFromSparseMatrix(const SparseMatrix<T> &matrixS) {
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

    return true;
}

template<typename T>
void Matrix<T>::changeStorageOrder() {
    const auto oldMajorOrder = storageOrder_;
    const auto oldLd = leadingDimension_;
    const auto &oldValues = values_;

    MatrixStorageOrder newMatrixOrder;
    size_t newLd;
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
    } else if (oldMajorOrder == MatrixStorageOrder::col_major) {
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
void SparseMatrix<T>::print() {
    std::cout << "SparseMatrix : [row,col,value]" << std::endl;
    for (int idx = 0; idx < nnz_; ++idx) {
        std::cout << "[" << rowIndex_[idx] << ","
                  << colIndex_[idx] << ","
                  << values_[idx] << "] ";
    }
    std::cout << std::endl;
}

template<typename T>
void Matrix<T>::print() {
    for (auto iter : values_) {
        std::cout << iter << " ";
    }
    std::cout << std::endl;
}

template<typename T>
T Matrix<T>::getOneValueForMultiplication(MatrixMultiplicationOrder multiplicationOrder,
                                          size_t row,
                                          size_t col,
                                          size_t k) const {
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
void SparseMatrix<T>::getSpareMatrixOneDataByCOO(const int idx, size_t &row, size_t &col, T &value) const {
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
void Matrix<T>::makeData(size_t numRow, size_t numCol, MatrixStorageOrder storageOrder) {
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

    std::mt19937 generator;
    auto distribution = util::createRandomUniformDistribution(static_cast<T>(0), static_cast<T>(10));

    for (int idx = 0; idx < values_.size(); ++idx) {
        values_[idx] = distribution(generator);
    }
}

template<typename T>
void SparseMatrix<T>::makeData(const size_t numRow, const size_t numCol, const size_t nnz) {
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
    host::sort_by_key(rowIndex_.data(), rowIndex_.data() + rowIndex_.size(), colIndex_.data());
    UIN lastRowNumber = rowIndex_[0];
    UIN lastBegin = 0;
    for (UIN idx = 0; idx < nnz_; ++idx) {
        const UIN curRowNumber = rowIndex_[idx];
        if (curRowNumber != lastRowNumber) { // new row
            host::sort(colIndex_.data() + lastBegin, colIndex_.data() + idx);

            lastBegin = idx + 1;
            lastRowNumber = curRowNumber;
        }

        if (idx == nnz_ - 1) {
            host::sort(colIndex_.data() + lastBegin, colIndex_.data() + colIndex_.size());
        }
    }
}

template
class Matrix<int>;
template
class SparseMatrix<int>;

template
class Matrix<float>;
template
class SparseMatrix<float>;