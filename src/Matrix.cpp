#include <iostream>
#include <fstream>
#include <string>
#include <random>

#include "Matrix.hpp"
#include "util.hpp"
#include "cudaUtil.cuh"

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
        const T val = static_cast<T>(std::stod(iterateOneWordFromLine(line, wordIter)));

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
void SparseMatrix<T>::print() {
    std::cout << "SparseMatrix : [row,col,value]" << std::endl;
    for (int idx = 0; idx < _nnz; ++idx) {
        std::cout << "[" << _rowIndex[idx] << ","
                  << _colIndex[idx] << ","
                  << _values[idx] << "] ";
    }
    std::cout << std::endl;
}

template<typename T>
void Matrix<T>::print() {
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
    if (multiplicationOrder == MatrixMultiplicationOrder::left_multiplication) {
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

template<typename T>
T Matrix<T>::getOneValue(int row, int col) const {
    if (_storageOrder == MatrixStorageOrder::row_major) {
        return _values[row * _leadingDimension + col];
    } else {
        return _values[col * _leadingDimension + row];
    }
}

template<typename T>
bool SparseMatrix<T>::setValuesFromMatrix(const Matrix<T> &inputMatrix) {
    _values.clear();
    _values.resize(_nnz);

    for (int idx = 0; idx < _nnz; ++idx) {
        const int row = _rowIndex[idx];
        const int col = _colIndex[idx];

        _values[idx] = inputMatrix.getOneValue(row, col);
    }

    return true;
}

template<typename T>
void SparseMatrix<T>::getSpareMatrixOneDataByCOO(const int idx, size_t &row, size_t &col, T &value) const {
    row = _rowIndex[idx];
    col = _colIndex[idx];
    value = _values[idx];
}

template<typename T>
bool SparseMatrix<T>::outputToMarketMatrixFile(const std::string &fileName) {
    std::string fileFormat(".mtx");

    std::string fileString(fileName + fileFormat);

    // check fileExists
    if (io::fileExists(fileString)) {
        std::cout << fileName + fileFormat << " file already exists" << std::endl;
        int fileId = 1;
        while (io::fileExists(fileName + "_" + std::to_string(fileId) + fileFormat)) {
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

    std::string secondLine(std::to_string(_row) + " " + std::to_string(_col) + " " + std::to_string(_nnz) + "\n");
    outfile << secondLine;

    for (int idx = 0; idx < _nnz; ++idx) {
        outfile << std::to_string(_rowIndex[idx] + 1) << " ";
        outfile << std::to_string(_colIndex[idx] + 1) << " ";
        outfile << std::to_string(_values[idx]);

        if (idx < _nnz - 1) {
            outfile << "\n";
        }
    }

    outfile.close();
    return true;
}

template<typename T>
void Matrix<T>::makeData(int row, size_t col, MatrixStorageOrder storageOrder = MatrixStorageOrder::row_major) {
    _row = row;
    _col = col;
    _size = row * col;
    _storageOrder = storageOrder;
    if (storageOrder == MatrixStorageOrder::row_major) {
        _leadingDimension = col;
    } else {
        _leadingDimension = row;
    }
    _values.resize(_size);

    std::mt19937 generator;
    std::uniform_real_distribution<T> distributionValue(0, 10);
    for (int idx = 0; idx < _values.size(); ++idx) {
        _values[idx] = distributionValue(generator);
    }
}

template<typename T>
void SparseMatrix<T>::makeData(const int row, const int col, const int nnz) {
    _row = row;
    _col = col;
    _nnz = nnz;

    _rowIndex.resize(nnz);
    _colIndex.resize(nnz);
    _values.resize(nnz);

    std::mt19937 generator;
    std::uniform_int_distribution<int> distributionRow(0, row - 1);
    std::uniform_int_distribution<int> distributionCol(0, col - 1);
    std::uniform_real_distribution<T> distributionValue(0, 1);

    for (int idx = 0; idx < nnz; ++idx) {
        _rowIndex[idx] = distributionRow(generator);
        _colIndex[idx] = distributionCol(generator);
        _values[idx] = distributionValue(generator);
    }

    // TODO : change data order
//    std::vector<uint64_t> rowIndexTmp(_rowIndex);
//    host::sort_by_key(_rowIndex.data(), _rowIndex.data() + _rowIndex.size(), _colIndex.data());
//    host::sort_by_key(rowIndexTmp.data(), rowIndexTmp.data() + rowIndexTmp.size(), _values.data());
//
//    std::vector<uint64_t> colIndexTmp(_colIndex);
//    host::sort_by_key(_colIndex.data(), _colIndex.data() + _colIndex.size(), _rowIndex.data());
//    host::sort_by_key(colIndexTmp.data(), colIndexTmp.data() + colIndexTmp.size(), _values.data());
}

template
class Matrix<float>;
template
class SparseMatrix<float>;