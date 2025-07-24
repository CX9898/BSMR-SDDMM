#include <iostream>
#include <fstream>
#include <string>
#include <random>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <limits>

#include <omp.h>

#include "util.hpp"
#include "Matrix.hpp"
#include "parallelAlgorithm.cuh"

using UIN = uint32_t;
constexpr UIN NULL_VALUE = std::numeric_limits<UIN>::max();

template<typename T>
Matrix<T>::Matrix(const sparseMatrix::COO<T> &matrixS) {
    row_ = matrixS.row();
    col_ = matrixS.col();
    const UIN size = matrixS.row() * matrixS.col();
    storageOrder_ = MatrixStorageOrder::row_major;
    const UIN ld = matrixS.col();
    leadingDimension_ = ld;

    values_.clear();
    values_.resize(size);
#pragma omp parallel for
    for (int idx = 0; idx < matrixS.nnz(); ++idx) {
        const UIN curRow = matrixS.rowIndices()[idx];
        const UIN curCol = matrixS.colIndices()[idx];
        const auto curVal = matrixS.values()[idx];

        values_[curRow * ld + curCol] = curVal;
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
    if (src.size() != row_ * col_) {
        std::cout << "Warning! Matrix value size mismatch" << std::endl;
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
    std::vector<T> newValues(values_.size());
    if (oldMajorOrder == MatrixStorageOrder::row_major) {
        newMatrixOrder = MatrixStorageOrder::col_major;
        newLd = row_;

#pragma omp parallel for
        for (int idx = 0; idx < oldValues.size(); ++idx) {
            const UIN row = idx / oldLd;
            const UIN col = idx % oldLd;
            const auto val = oldValues[idx];

            newValues[col * newLd + row] = val;
        }
    } else {
        newMatrixOrder = MatrixStorageOrder::row_major;
        newLd = col_;

#pragma omp parallel for
        for (int idx = 0; idx < values_.size(); ++idx) {
            const UIN col = idx / oldLd;
            const UIN row = idx % oldLd;
            const auto val = values_[idx];

            newValues[row * newLd + col] = val;
        }
    }

    storageOrder_ = newMatrixOrder;
    leadingDimension_ = newLd;
    values_ = newValues;
}

template<typename T>
void Matrix<T>::makeData() {
    makeData(row_, col_);
}

template<typename T>
void Matrix<T>::makeData(UIN numRow, UIN numCol) {
    row_ = numRow;
    col_ = numCol;
    if (storageOrder_ == MatrixStorageOrder::row_major) {
        leadingDimension_ = numCol;
    } else {
        leadingDimension_ = numRow;
    }
    values_.resize(numRow * numCol);

//    for (UIN idx = 0; idx < values_.size(); ++idx) {
//        values_[idx] = static_cast<T>(idx);
//    }
    std::mt19937 generator;
    auto distribution = util::createRandomUniformDistribution(static_cast<T>(0), static_cast<T>(2));

#pragma omp parallel for
    for (int idx = 0; idx < values_.size(); ++idx) {
        values_[idx] = distribution(generator);
    }
}

template<typename T>
void Matrix<T>::print() const {
//    for (size_t idx = 0; idx < size(); ++idx) {
//        printf("matrix[%d] = %f\n", idx, static_cast<float>(values_[idx]));
//    }
    for (auto iter : values_) {
        std::cout << iter << " ";
    }
    std::cout << std::endl;
}

template<typename T>
void Matrix<T>::printToMarkdownTable() const {
    std::cout << "| |";
    for (int colIdx = 0; colIdx < col_; ++colIdx) {
        std::cout << colIdx << "|";
    }
    std::cout << std::endl;

    std::cout << "|-|";
    for (int colIdx = 0; colIdx < col_; ++colIdx) {
        std::cout << "-|";
    }
    std::cout << std::endl;

    for (int rowIdx = 0; rowIdx < row_; ++rowIdx) {
        std::cout << "|" << rowIdx << "|";
        for (int colIdx = 0; colIdx < col_; ++colIdx) {
            std::cout << getOneValue(rowIdx, colIdx) << "|";
        }
        std::cout << std::endl;
    }
}

template<typename T>
std::vector<T> Matrix<T>::getRowVector(UIN row) const {
    std::vector<T> rowVector(col());

#pragma omp parallel for
    for (int col = 0; col < col_; ++col) {
        rowVector[col] = getOneValue(row, col);
    }

    return rowVector;
}

template<typename T>
std::vector<T> Matrix<T>::getColVector(UIN col) const {
    std::vector<T> colVector(row());

#pragma omp parallel for
    for (int row = 0; row < row_; ++row) {
        colVector[row] = getOneValue(row, col);
    }

    return colVector;
}

template<typename T>
T Matrix<T>::getOneValueForMultiplication(MatrixMultiplicationOrder multiplicationOrder,
                                          UIN rowMtxC,
                                          UIN colMtxC,
                                          UIN positionOfKIter) const {
    if (multiplicationOrder == MatrixMultiplicationOrder::left_multiplication) {
        if (rowMtxC > row_) {
            std::cout << "Warning! The input rows exceed the matrix" << std::endl;
        }
        if (storageOrder_ == MatrixStorageOrder::row_major) {
            return values_[rowMtxC * leadingDimension_ + positionOfKIter];
        } else {
            return values_[positionOfKIter * leadingDimension_ + rowMtxC];
        }
    } else {
        if (colMtxC > col_) {
            std::cout << "Warning! The input columns exceed the matrix" << std::endl;
        }
        if (storageOrder_ == MatrixStorageOrder::row_major) {
            return values_[positionOfKIter * leadingDimension_ + colMtxC];
        } else {
            return values_[colMtxC * leadingDimension_ + positionOfKIter];
        }
    }
}

template<typename T>
T Matrix<T>::getOneValue(UIN row, UIN col) const {
    if (row > row_ || col > col_) {
        std::cout << "Warning! The input rows or columns exceed the matrix" << std::endl;
    }
    if (storageOrder_ == MatrixStorageOrder::row_major) {
        return values_[row * leadingDimension_ + col];
    } else {
        return values_[col * leadingDimension_ + row];
    }
}

void getCsrRowOffsets(const UIN row, const std::vector<UIN> &rowIndices, std::vector<UIN> &rowOffsets) {
    rowOffsets.resize(row + 1);
    rowOffsets[0] = 0;
    UIN rowPtrIdx = 0;
    for (int idx = 0; idx < rowIndices.size(); ++idx) {
        while (rowPtrIdx < rowOffsets.size() - 1 && rowPtrIdx != rowIndices[idx]) {
            rowOffsets[rowPtrIdx + 1] = idx;
            ++rowPtrIdx;
        }
    }
    while (rowPtrIdx < rowOffsets.size() - 1) {
        rowOffsets[rowPtrIdx + 1] = rowIndices.size();
        ++rowPtrIdx;
    }
}

template
class Matrix<int>;

template
class Matrix<float>;

template
class Matrix<double>;

template
class sparseMatrix::CSR<int>;

template
class sparseMatrix::CSR<float>;

template
class sparseMatrix::CSR<double>;

template
class sparseMatrix::COO<int>;

template
class sparseMatrix::COO<float>;

template
class sparseMatrix::COO<double>;

template<typename T>
bool sparseMatrix::CSR<T>::initializeFromMatrixFile(const std::string &file) {

    const std::string fileSuffix = util::getFileSuffix(file);
    if (fileSuffix == ".mtx" || fileSuffix == ".mmio") {
        return initializeFromMtxFile(file);
    } else if (fileSuffix == ".smtx") {
        return initializeFromSmtxFile(file);
    } else if (fileSuffix == ".txt") {
        return initializeFromGraphDataset(file);
    } else {
        std::cerr << "Error, file format is not supported : " << file << std::endl;
    }

    return false;
}

template<typename T>
bool sparseMatrix::CSR<T>::initializeFromSmtxFile(const std::string &file) {
    std::ifstream inFile;
    inFile.open(file, std::ios::in); // open file
    if (!inFile.is_open()) {
        std::cerr << "Error, file cannot be opened : " << file << std::endl;
        return false;
    }

    std::cout << "sparseMatrix::CSR initialize From file : " << file << std::endl;

    std::string line; // Store the data for each line
    while (getline(inFile, line) && line[0] == '%') {} // Skip comments

    int wordIter = 0;
    row_ = std::stoi(util::iterateOneWordFromLine(line, wordIter));
    col_ = std::stoi(util::iterateOneWordFromLine(line, wordIter));
    nnz_ = std::stoi(util::iterateOneWordFromLine(line, wordIter));

    if (nnz_ == 0) {
        std::cerr << "Error, file " << file << " nnz is 0!" << std::endl;
        return false;
    }

    rowOffsets_.resize(row_ + 1);
    colIndices_.resize(nnz_);
    values_.resize(nnz_);

    // initialize rowOffsets
    {
        getline(inFile, line);
        wordIter = 0;
        int idx = 0;
        for (; idx < rowOffsets_.size(); ++idx) {
            const UIN rowOffset = std::stoi(util::iterateOneWordFromLine(line, wordIter));
            rowOffsets_[idx] = rowOffset;
        }
        if (idx < rowOffsets_.size()) {
            std::cerr << "Error, file " << file << " rowOffsets is not enough!" << std::endl;
            return false;
        }
    }

    // initialize colIndices and values
    {
        getline(inFile, line);
        wordIter = 0;
        int idx = 0;
        for (; idx < colIndices_.size(); ++idx) {
            const UIN col = std::stoi(util::iterateOneWordFromLine(line, wordIter));
            colIndices_[idx] = col;
            values_[idx] = static_cast<T>(1);
        }
        if (idx < nnz_) {
            std::cerr << "Error, file " << file << " nnz is not enough!" << std::endl;
            return false;
        }
    }

    // Check data
    for (int row = 0; row < row_; ++row) {
        std::unordered_set<UIN> colSet;
        for (int idx = rowOffsets_[row]; idx < rowOffsets_[row + 1]; ++idx) {
            const UIN col = colIndices_[idx];
            if (colSet.find(col) != colSet.end()) {
                std::cerr << "Error, matrix has duplicate data!" << std::endl;
                return false;
            }
            colSet.insert(col);
        }
    }

    inFile.close();

    return true;
}

template<typename T>
bool getOneLineThreeData(const std::string &line, UIN &first, UIN &second, T &third) {
    if (line.empty()) {
        return false;
    }

    int wordIter = 0;
    first = std::stoi(util::iterateOneWordFromLine(line, wordIter));
    second = std::stoi(util::iterateOneWordFromLine(line, wordIter));
    const std::string valueStr = util::iterateOneWordFromLine(line, wordIter);
    try {
        third = valueStr.empty() ? static_cast<T>(0) : static_cast<T>(std::stod(valueStr));
    } catch (const std::out_of_range&) {
        std::cout << "Warning: valueStr out of range: " << valueStr << std::endl;
        third = static_cast<T>(0);
    }

//    if (wordIter < line.size()) {
//        std::cerr << "Error, file \"" << line << "\" line format is incorrect!" << std::endl;
//        return false;
//    }

    return true;
}

template<typename T>
bool sparseMatrix::CSR<T>::initializeFromMtxFile(const std::string &file) {
    std::ifstream inFile;
    inFile.open(file, std::ios::in); // open file
    if (!inFile.is_open()) {
        std::cerr << "Error, file cannot be opened : " << file << std::endl;
        return false;
    }

    std::cout << "sparseMatrix::CSR initialize from file : " << file << std::endl;

    std::string line; // Store the data for each line
    while (getline(inFile, line) && line[0] == '%') {} // Skip comments

    getOneLineThreeData(line, row_, col_, nnz_);
    if (row_ == NULL_VALUE || col_ == NULL_VALUE || nnz_ == NULL_VALUE) {
        std::cerr << "Error, file " << file << " format is incorrect!" << std::endl;
        return false;
    }

    std::vector<UIN> rowIndices(nnz_);
    std::vector<UIN> colIndices(nnz_);
    std::vector<T> values(nnz_);

    UIN idx = 0;
    while (getline(inFile, line)) {
        UIN row = NULL_VALUE, col = NULL_VALUE;
        T val;
        if (!getOneLineThreeData(line, row, col, val)) {
            continue;
        }

        if (idx >= nnz_) {
            std::cerr << "Error, file " << file << " too many elements, exceeding the number nnz!" << std::endl;
            return false;
        }

        rowIndices[idx] = row - 1;
        colIndices[idx] = col - 1;
        values[idx] = val;

        ++idx;
    }

    // Check data
    if (idx < nnz_) {
        std::cerr << "Error, file " << file << " elements is not enough!" << std::endl;
        return false;
    }
    std::set<std::pair<UIN, UIN>> rowColSet;
    for (int idx = 0; idx < nnz_; ++idx) {
        const UIN row = rowIndices[idx];
        const UIN col = colIndices[idx];
        if (row >= row_ || col >= col_) {
            std::cerr << "Error, file " << file << " row or col is too big!" << std::endl;
            return false;
        }
        std::pair<UIN, UIN> rowColPair(row, col);
        if (rowColSet.find(rowColPair) != rowColSet.end()) {
            std::cerr << "Error, matrix has duplicate data!" << std::endl;
            return false;
        }
        rowColSet.insert(rowColPair);
    }
    if (nnz_ <= 1) {
        std::cerr << "Warning, file " << file << " nnz is 1, this is not a valid matrix!" << std::endl;
        return false;
    }

    host::sort_by_key_for_multiple_vectors(rowIndices.data(),
                                           rowIndices.data() + rowIndices.size(),
                                           colIndices.data(),
                                           values.data());

    std::vector<UIN> rowOffsets;
    getCsrRowOffsets(row_, rowIndices, rowOffsets_);
    colIndices_ = colIndices;
    values_ = values;

    inFile.close();

    return true;
}

template<typename T>
bool sparseMatrix::CSR<T>::initializeFromGraphDataset(const std::string &file) {
    std::ifstream inFile;
    inFile.open(file, std::ios::in); // open file
    if (!inFile.is_open()) {
        std::cerr << "Error, file cannot be opened : " << file << std::endl;
        return false;
    }

    std::cout << "sparseMatrix::CSR initialize From file : " << file << std::endl;

    std::string line; // Store the data for each line
    while (getline(inFile, line) && line[0] == '#') {
        int wordIter = 0;
        const std::string nodesStr("Nodes: ");
        const std::string edgesStr("Edges: ");
        if (line.find(nodesStr) != std::string::npos) {
            wordIter = line.find(nodesStr) + 7;
            const int nodes = std::stoi(util::iterateOneWordFromLine(line, wordIter));
            row_ = nodes;
            col_ = nodes;
        }
        if (line.find(edgesStr) != std::string::npos) {
            wordIter = line.find(edgesStr) + 7;
            const int edges = std::stoi(util::iterateOneWordFromLine(line, wordIter));
            nnz_ = edges;
        }
    }

    if (!row_ || !col_ || !nnz_) {
        std::cerr << "Error, file " << file << " row or col or nnz not initialized!" << std::endl;
        return false;
    }

    std::vector<UIN> rowIndices(nnz_);
    std::vector<UIN> colIndices(nnz_);
    std::vector<T> values(nnz_, 0);

    UIN idx = 0;
    std::unordered_map<UIN, UIN> nodeToIdMap;
    UIN nodeCount = 0;
    do {
        UIN node, node2;
        T third;
        if (!getOneLineThreeData(line, node, node2, third)) {
            continue;
        }
        if (nodeToIdMap.find(node) == nodeToIdMap.end()) {
            nodeToIdMap[node] = nodeCount;
            ++nodeCount;
        }
        if (nodeToIdMap.find(node2) == nodeToIdMap.end()) {
            nodeToIdMap[node2] = nodeCount;
            ++nodeCount;
        }

        if (idx >= nnz_) {
            std::cerr << "Error, file " << file << " too many elements, exceeding the number nnz!" << std::endl;
            return false;
        }

        rowIndices[idx] = nodeToIdMap[node];
        colIndices[idx] = nodeToIdMap[node2];
        values[idx] = third;

        ++idx;
    } while (getline(inFile, line));

    // Check data
    if (idx < nnz_) {
        std::cerr << "Error, file " << file << " elements is not enough!" << std::endl;
        return false;
    }
    std::set<std::pair<UIN, UIN>> rowColSet;
    for (int idx = 0; idx < nnz_; ++idx) {
        const UIN row = rowIndices[idx];
        const UIN col = colIndices[idx];
        if (row >= row_ || col >= col_) {
            std::cerr << "Error, file " << file << " row or col is too big!" << std::endl;
            return false;
        }
        std::pair<UIN, UIN> rowColPair(row, col);
        if (rowColSet.find(rowColPair) != rowColSet.end()) {
            fprintf(stderr, "Error, matrix has duplicate data! row:%d, col:%d\n",
                    row, col);
            return false;
        }
        rowColSet.insert(rowColPair);
    }

    host::sort_by_key_for_multiple_vectors(rowIndices.data(),
                                           rowIndices.data() + rowIndices.size(),
                                           colIndices.data(),
                                           values.data());

    std::vector<UIN> rowOffsets;
    getCsrRowOffsets(row_, rowIndices, rowOffsets_);
    colIndices_ = colIndices;
    values_ = values;

    inFile.close();

    return true;
}

template<typename T>
sparseMatrix::COO<T>::COO(const CSR<T> &csr) {
    row_ = csr.row();
    col_ = csr.col();
    nnz_ = csr.nnz();

    rowIndices_.resize(nnz_);
    colIndices_.resize(nnz_);
    values_.resize(nnz_);

#pragma omp parallel for
    for (int row = 0; row < row_; ++row) {
        for (int idx = csr.rowOffsets()[row]; idx < csr.rowOffsets()[row + 1]; ++idx) {
            const UIN col = csr.colIndices()[idx];
            const T val = csr.values()[idx];

            rowIndices_[idx] = row;
            colIndices_[idx] = col;
            values_[idx] = val;
        }
    }
}

template<typename T>
bool sparseMatrix::COO<T>::initializeFromMatrixMarketFile(const std::string &file) {
    std::ifstream inFile;
    inFile.open(file, std::ios::in); // open file
    if (!inFile.is_open()) {
        std::cerr << "Error, file cannot be opened : " << file << std::endl;
        return false;
    }

    std::cout << "sparseMatrix::COO initialize from file : " << file << std::endl;

    std::string line; // Store the data for each line
    while (getline(inFile, line) && line[0] == '%') {} // Skip comments

    getOneLineThreeData(line, row_, col_, nnz_);
    if (row_ == NULL_VALUE || col_ == NULL_VALUE || nnz_ == NULL_VALUE) {
        std::cerr << "Error, file " << file << " format is incorrect!" << std::endl;
        return false;
    }

    rowIndices_.resize(nnz_);
    colIndices_.resize(nnz_);
    values_.resize(nnz_);

    UIN idx = 0;
    while (getline(inFile, line)) {
        UIN row = NULL_VALUE, col = NULL_VALUE;
        T val;
        if (!getOneLineThreeData(line, row, col, val)) {
            continue;
        }

        rowIndices_[idx] = row - 1;
        colIndices_[idx] = col - 1;
        values_[idx] = val;

        ++idx;
    }

    // Check data
    if (idx < nnz_) {
        std::cerr << "Error, file " << file << " nnz is not enough!" << std::endl;
        return false;
    }
    std::set<std::pair<UIN, UIN>> rowColSet;
    for (int idx = 0; idx < nnz_; ++idx) {
        const UIN row = rowIndices_[idx];
        const UIN col = colIndices_[idx];
        if (row >= row_ || col >= col_) {
            std::cerr << "Error, file " << file << " row or col is too big!" << std::endl;
            return false;
        }
        std::pair<UIN, UIN> rowColPair(row, col);
        if (rowColSet.find(rowColPair) != rowColSet.end()) {
            std::cerr << "Error, matrix has duplicate data!" << std::endl;
            return false;
        }
        rowColSet.insert(rowColPair);
    }

    inFile.close();

    return true;
}

template<typename T>
bool sparseMatrix::COO<T>::setValuesFromMatrix(const Matrix<T> &inputMatrix) {
    if (inputMatrix.row() < row_ || inputMatrix.col() < col_) {
        std::cout << "Warning! The input matrix size is too small." << std::endl;
    }
    values_.clear();
    values_.resize(nnz_);

#pragma omp parallel for
    for (int idx = 0; idx < nnz_; ++idx) {
        const UIN row = rowIndices_[idx];
        const UIN col = colIndices_[idx];

        values_[idx] = inputMatrix.getOneValue(row, col);
    }

    return true;
}

template<typename T>
std::tuple<UIN, UIN, T> sparseMatrix::COO<T>::getSpareMatrixOneData(const UIN idx) const {
    return std::make_tuple(rowIndices_[idx], colIndices_[idx], values_[idx]);
}

template<typename T>
sparseMatrix::CSR<T> sparseMatrix::COO<T>::getCsrData() const {
    std::vector<UIN> rowIndicesTmp = rowIndices_;
    std::vector<UIN> colIndicesTmp = colIndices_;
    std::vector<T> valuesTmp = values_;

    host::sort_by_key_for_multiple_vectors(rowIndicesTmp.data(),
                                           rowIndicesTmp.data() + rowIndicesTmp.size(),
                                           colIndicesTmp.data(),
                                           valuesTmp.data());

    std::vector<UIN> rowOffsets;
    getCsrRowOffsets(row_, rowIndicesTmp, rowOffsets);
    sparseMatrix::CSR<T> csrData(row_, col_, nnz_, rowOffsets, colIndicesTmp, valuesTmp);
    return csrData;
}

template<typename T>
void sparseMatrix::COO<T>::print() const {
    std::cout << "SparseMatrix : [row,col,value]" << std::endl;
    for (UIN idx = 0; idx < nnz_; ++idx) {
        std::cout << "[" << rowIndices_[idx] << ","
                  << colIndices_[idx] << ","
                  << values_[idx] << "] ";
    }
    std::cout << std::endl;
}

template<typename T>
bool checkMatrixData(const sparseMatrix::CSR<T> &csr) {
    if (csr.nnz() == 0) {
        if (!csr.colIndices().empty()) {
            fprintf(stderr, "Error, CSR nnz is 0, but colIndices is not empty!\n");
            return false;
        } else {
            return true;
        }
    }

    int numNonZeroRows = 0;
    for (int row = 0; row < csr.row(); ++row) {
        if (csr.rowOffsets()[row] > csr.rowOffsets()[row + 1]) {
            fprintf(stderr, "Error, CSR rowOffsets[%d] > rowOffsets[%d]\n", row, row + 1);
            return false;
        }
        if (csr.rowOffsets()[row + 1] - csr.rowOffsets()[row] > 0) {
            ++numNonZeroRows;
        }
    }
    if (numNonZeroRows == 0 && csr.nnz() > 0) {
        fprintf(stderr, "Error, CSR nnz is %d, but no non-zero rows!\n", csr.nnz());
        return false;
    }

    for (int row = 0; row < csr.row(); ++row) {
        for (int idx = csr.rowOffsets()[row]; idx < csr.rowOffsets()[row + 1]; ++idx) {
            const UIN col = csr.colIndices()[idx];
            if (col >= csr.col()) {
                return false;
            }
        }
    }

    return true;
}

template bool checkMatrixData<float>(const sparseMatrix::CSR<float> &csr);