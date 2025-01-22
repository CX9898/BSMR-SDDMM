#include <cmath>
#include <omp.h>

#include "bsa.hpp"
#include "Matrix.hpp"
#include "parallelAlgorithm.cuh"

#define COL_BLOCK_SIZE 32

void encoding(const sparseDataType::CSR &matrix, std::vector<std::vector<UIN>> &encodings) {
    const int colBlock = std::ceil(matrix.col_ / COL_BLOCK_SIZE);
    encodings.resize(matrix.row_);
#pragma omp parallel for dynamic
    for (int row = 0; row < matrix.row_; ++row) {
        encodings[row].resize(colBlock);
        for (int idx = matrix.rowPtr_[row]; idx < matrix.rowPtr_[idx + 1]; ++idx) {
            const int col = matrix.colIndices_[idx];
            ++encodings[row][col / COL_BLOCK_SIZE];
        }
    }
}

void calculateDispersion(const UIN col,
                         const std::vector<std::vector<UIN>> &encodins,
                         std::vector<float> &dispersions) {
#pragma omp parallel for dynamic
    for (int row = 0; row < encodins.size(); ++row) {
        UIN numOfNonZeroColBlocks = 0;
        UIN zeroFillings = 0;
        for (int colBlockIdx = 0; colBlockIdx < encodins[row].size(); ++colBlockIdx) {
            const UIN numOfNonZeroCols = encodins[row][colBlockIdx];
            if (numOfNonZeroCols == 0) {
                ++numOfNonZeroColBlocks;
            } else {
                zeroFillings += col - numOfNonZeroCols;
            }
        }
        dispersions[row] = COL_BLOCK_SIZE * numOfNonZeroColBlocks + zeroFillings;
    }
}

void reordering(sparseDataType::CSR &matrix) {
    std::vector<std::vector<UIN>> encodings;
    encoding(matrix, encodings);
    std::vector<float> dispersions(matrix.row_);
    calculateDispersion(matrix.col_, encodings, dispersions);
}