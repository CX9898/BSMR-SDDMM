#include <iostream>
#include <fstream>
#include <string>

#include "Matrix.hpp"
#include "util.hpp"

template<typename T>
bool SparseMatrix<T>::initializeFromMatrixMarketFile(const std::string &filePath) {
    std::ifstream inFile;
    inFile.open(filePath, std::ios::in); // open file
    if (!inFile.is_open()) {
        std::cout << "Error, Matrix Market file cannot be opened." << std::endl;
        return false;
    }

    std::string line; // Store the data for each line
    getline(inFile, line); // First line does not operate

    getline(inFile, line);
    int wordIter = 0;
    _row = std::stoi(iterateOneWordFromLine(line, wordIter));
    _col = std::stoi(iterateOneWordFromLine(line, wordIter));
    _nnz = std::stoi(iterateOneWordFromLine(line, wordIter));

    if (wordIter < line.size()) {
        std::cerr << "Error, Matrix Market file " << line << " line format is incorrect!" << std::endl;
    }

    _rowIndex.resize(_nnz);
    _colIndex.resize(_nnz);
    _values.resize(_nnz);

    int idx = 0;
    while (getline(inFile, line)) {
        wordIter = 0;
        const int row = std::stoi(iterateOneWordFromLine(line, wordIter)) - 1;
        const int col = std::stoi(iterateOneWordFromLine(line, wordIter)) - 1;
        const T val = (T) std::stod(iterateOneWordFromLine(line, wordIter));

        if (wordIter < line.size()) {
            std::cerr << "Error, Matrix Market file " << line << " line format is incorrect!" << std::endl;
        }

        _rowIndex[idx] = row;
        _colIndex[idx] = col;
        _values[idx] = val;

        ++idx;
    }

    inFile.close();

    return true;
}

template<typename T>
bool Matrix<T>::initializeFromSparseMatrix(const SparseMatrix<T> &matrixS) {
    _row = matrixS.row();
    _col = matrixS.col();
    const int size = matrixS.row() * matrixS.col();
    _size = size;
    _storageOrder = MatrixStorageOrder::row_major;
    const int ld = matrixS.col();
    _leadingDimension = ld;

    const auto &rowIndexS = matrixS.rowIndex();
    const auto &colIndexS = matrixS.colIndex();
    const auto &valuesS = matrixS.values();

    _values.clear();
    _values.resize(size);
    for (int idx = 0; idx < matrixS.nnz(); ++idx) {
        const int row = rowIndexS[idx];
        const int col = colIndexS[idx];
        const auto val = valuesS[idx];

        _values[row * ld + col] = val;
    }

    return true;
}

template<typename T>
void Matrix<T>::changeStorageOrder() {
    const auto oldMajorOrder = _storageOrder;
    const auto oldLd = _leadingDimension;
    const auto &oldValues = _values;

    MatrixStorageOrder newMatrixOrder;
    size_t newLd;
    std::vector<float> newValues(_size);
    if (oldMajorOrder == MatrixStorageOrder::row_major) {
        newMatrixOrder = MatrixStorageOrder::col_major;
        newLd = _row;

        for (int idx = 0; idx < oldValues.size(); ++idx) {
            const int row = idx / oldLd;
            const int col = idx % oldLd;
            const auto val = oldValues[idx];

            newValues[col * newLd + row] = val;
        }
    } else if (oldMajorOrder == MatrixStorageOrder::col_major) {
        newMatrixOrder = MatrixStorageOrder::row_major;
        newLd = _col;

        for (int idx = 0; idx < _values.size(); ++idx) {
            const int col = idx / oldLd;
            const int row = idx % oldLd;
            const auto val = _values[idx];

            newValues[row * newLd + col] = val;
        }
    }

    _storageOrder = newMatrixOrder;
    _leadingDimension = newLd;
    _values = newValues;
}

template<typename T>
void SparseMatrix<T>::printfValue() {
    std::cout << "row :\t";
    for (auto iter : _rowIndex) {
        std::cout << iter << "\t";
    }
    std::cout << std::endl;
    std::cout << "col :\t";
    for (auto iter : _colIndex) {
        std::cout << iter << "\t";
    }
    std::cout << std::endl;
    std::cout << "value :\t";
    for (auto iter : _values) {
        std::cout << iter << "\t";
    }
    std::cout << std::endl;
}

template<typename T>
void Matrix<T>::printfValue() {
    for (auto iter : _values) {
        std::cout << iter << " ";
    }
    std::cout << std::endl;
}

template<typename T>
T Matrix<T>::getOneValueForMultiplication(MatrixMultiplicationOrder multiplicationOrder,
                                          size_t row,
                                          size_t col,
                                          size_t k) const {
    if (multiplicationOrder == MatrixMultiplicationOrder::Left_multiplication) {
        if (_storageOrder == MatrixStorageOrder::row_major) {
            return _values[row * _leadingDimension + k];
        } else {
            return _values[k * _leadingDimension + row];
        }
    } else {
        if (_storageOrder == MatrixStorageOrder::row_major) {
            return _values[k * _leadingDimension + col];
        } else {
            return _values[col * _leadingDimension + k];
        }
    }
}

template class Matrix<float>;
template class SparseMatrix<float>;