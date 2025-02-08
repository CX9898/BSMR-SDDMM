#include <cmath>
#include <omp.h>
#include <numeric>
#include <algorithm>

#include "reordering.hpp"
#include "parallelAlgorithm.cuh"

#define COL_BLOCK_SIZE 32

void encoding(const sparseDataType::CSR<float> &matrix, std::vector<std::vector<UIN>> &encodings) {
    const int colBlock = std::ceil(static_cast<float>(matrix.col_) / COL_BLOCK_SIZE);
    encodings.resize(matrix.row_);
#pragma omp parallel for dynamic
    for (int row = 0; row < matrix.row_; ++row) {
        encodings[row].resize(colBlock);
        for (int idx = matrix.rowOffsets_[row]; idx < matrix.rowOffsets_[row + 1]; ++idx) {
            const int col = matrix.colIndices_[idx];
            ++encodings[row][col / COL_BLOCK_SIZE];
        }
    }
}

void calculateDispersion(const UIN col,
                         const std::vector<std::vector<UIN>> &encodins,
                         std::vector<UIN> &dispersions) {
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

// return similarity between two encodings
float clusterComparison(const std::vector<UIN> &encoding_rep, const std::vector<UIN> &encoding_cmp) {
    UIN sum_of_squares_rep = 0;
    UIN sum_of_squares_cmp = 0;
    for (int idx = 0; idx < encoding_rep.size(); ++idx) {
        sum_of_squares_rep += encoding_rep[idx] * encoding_rep[idx];
        sum_of_squares_cmp += encoding_cmp[idx] * encoding_cmp[idx];
    }
    if (sum_of_squares_rep == 0 && sum_of_squares_cmp == 0) {
        return 1.0f;
    } else if ((sum_of_squares_rep == 0 || sum_of_squares_cmp == 0)) {
        return 0.0f;
    }
    float norm_rep = sqrt((float) sum_of_squares_rep);
    float norm_cmp = sqrt((float) sum_of_squares_cmp);
    float min_sum = 0.0f;
    float max_sum = 0.0f;
    for (int idx = 0; idx < encoding_rep.size(); ++idx) {
        float sim_rep = (float) encoding_rep[idx] / norm_rep;
        float sim_cmp = (float) encoding_cmp[idx] / norm_cmp;
        min_sum += fminf(sim_rep, sim_cmp);
        max_sum += fmaxf(sim_rep, sim_cmp);
    }
    return min_sum / max_sum;
}

void clustering(const UIN row, const std::vector<std::vector<UIN>> &encodings,
                const std::vector<UIN> &ascending, const UIN startIndexOfNonZeroRow, std::vector<int> &clusterIds) {

    for (int idx = startIndexOfNonZeroRow; idx < row - 1; ++idx) {
        if (idx > startIndexOfNonZeroRow && clusterIds[idx] != -1) {
            continue;
        }
        clusterIds[idx] = idx;
#pragma omp parallel for dynamic
        for (int cmpIdx = idx + 1; cmpIdx < row; ++cmpIdx) {
            if (clusterIds[cmpIdx] != -1) {
                continue;
            }
            const float similarity =
                clusterComparison(encodings[ascending[startIndexOfNonZeroRow]], encodings[ascending[cmpIdx]]);
            if (similarity > row_similarity_threshold_alpha) {
                clusterIds[ascending[cmpIdx]] = clusterIds[ascending[idx]];
            }
        }
    }
}

void row_reordering(const sparseDataType::CSR<float> &matrix, struct ReorderedMatrix &reorderedMatrix) {
    std::vector<std::vector<UIN>> encodings;
    encoding(matrix, encodings);

    std::vector<UIN> dispersions(matrix.row_);
    calculateDispersion(matrix.col_, encodings, dispersions);

    std::vector<UIN> ascending(matrix.row_);
    std::iota(ascending.begin(), ascending.end(), 0); // ascending = {0, 1, 2, 3, ... rows-1}
    std::stable_sort(ascending.begin(),
                     ascending.end(),
                     [&dispersions](size_t i, size_t j) { return dispersions[i] < dispersions[j]; });

    std::vector<int> clusterIds(matrix.row_, -1);
    UIN startIndexOfNonZeroRow = 0;
    while (startIndexOfNonZeroRow < matrix.row_ && dispersions[ascending[startIndexOfNonZeroRow]] == 0) {
        clusterIds[ascending[startIndexOfNonZeroRow]] = 0;
        ++startIndexOfNonZeroRow;
    }
    clustering(matrix.row_, encodings, ascending, startIndexOfNonZeroRow, clusterIds);

    reorderedMatrix.reorderedRowIndices_.resize(matrix.row_);
    std::iota(reorderedMatrix.reorderedRowIndices_.begin(),
              reorderedMatrix.reorderedRowIndices_.end(),
              0); // rowIndices = {0, 1, 2, 3, ... rows-1}
    std::stable_sort(reorderedMatrix.reorderedRowIndices_.begin(),
                     reorderedMatrix.reorderedRowIndices_.end(),
                     [&clusterIds](int i, int j) { return clusterIds[i] < clusterIds[j]; });

    // Remove zero rows
    {
        UIN startIndexOfNonZeroRow = 0;
        while (startIndexOfNonZeroRow < matrix.row_
            && matrix.rowOffsets_[reorderedMatrix.reorderedRowIndices_[startIndexOfNonZeroRow] + 1]
                - matrix.rowOffsets_[reorderedMatrix.reorderedRowIndices_[startIndexOfNonZeroRow]] == 0) {
            ++startIndexOfNonZeroRow;
        }
        reorderedMatrix.reorderedRowIndices_.erase(reorderedMatrix.reorderedRowIndices_.begin(),
                                                   reorderedMatrix.reorderedRowIndices_.begin() + startIndexOfNonZeroRow);
    }
}