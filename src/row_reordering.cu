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
                         const std::vector<std::vector<UIN>> &encodings,
                         std::vector<UIN> &dispersions) {
#pragma omp parallel for dynamic
    for (int row = 0; row < encodings.size(); ++row) {
        UIN numOfNonZeroColBlocks = 0;
        UIN zeroFillings = 0;
        for (int colBlockIdx = 0; colBlockIdx < encodings[row].size(); ++colBlockIdx) {
            const UIN numOfNonZeroCols = encodings[row][colBlockIdx];
            if (numOfNonZeroCols != 0) {
                ++numOfNonZeroColBlocks;
                zeroFillings += block_col_size - numOfNonZeroCols;
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

void clustering(const std::vector<std::vector<UIN>> &encodings,
                const std::vector<UIN> &rows, const UIN startIndexOfNonZeroRow, std::vector<int> &clusterIds) {

//    UIN num = 0;
    for (int idxRow = startIndexOfNonZeroRow; idxRow < encodings.size() - 1; ++idxRow) {
        if (idxRow > startIndexOfNonZeroRow && clusterIds[idxRow] != -1) {
            continue;
        }
        clusterIds[idxRow] = idxRow;
//#pragma omp parallel for dynamic
        for (int cmpIdx = idxRow + 1; cmpIdx < encodings.size(); ++cmpIdx) {
            if (clusterIds[cmpIdx] != -1) {
                continue;
            }
            const float similarity =
                clusterComparison(encodings[rows[startIndexOfNonZeroRow]], encodings[rows[cmpIdx]]);
            if (similarity > row_similarity_threshold_alpha) {
                clusterIds[rows[cmpIdx]] = clusterIds[rows[idxRow]];
//                ++num;
            }
        }
    }
//    printf("!!! num = %d\n", num);
}

void row_reordering(const sparseDataType::CSR<float> &matrix, struct ReorderedMatrix &reorderedMatrix) {
    std::vector<std::vector<UIN>> encodings;
    encoding(matrix, encodings);

    std::vector<UIN> dispersions(matrix.row_);
    calculateDispersion(matrix.col_, encodings, dispersions);

    std::vector<UIN> ascendingRow(matrix.row_); // Store the original row id
    std::iota(ascendingRow.begin(), ascendingRow.end(), 0); // ascending = {0, 1, 2, 3, ... rows-1}
    std::stable_sort(ascendingRow.begin(),
                     ascendingRow.end(),
                     [&dispersions](size_t i, size_t j) { return dispersions[i] < dispersions[j]; });

    std::vector<int> clusterIds(matrix.row_, -1);
    UIN startIndexOfNonZeroRow = 0;
    while (startIndexOfNonZeroRow < matrix.row_ && dispersions[ascendingRow[startIndexOfNonZeroRow]] == 0) {
        clusterIds[ascendingRow[startIndexOfNonZeroRow]] = 0;
        ++startIndexOfNonZeroRow;
    }

//    printf("!!! 1103 numCols = %d\n", matrix.rowOffsets_[1104] - matrix.rowOffsets_[1103]);
//    printf("!!! 1178 numCols = %d\n", matrix.rowOffsets_[1179] - matrix.rowOffsets_[1178]);
//    for (int i = 0; i < encodings[1103].size(); ++i) {
//        if (encodings[1103][i] != 0) {
//            printf("!!! 1103 encodings[1103][%d] = %d\n", i, encodings[1103][i]);
//        }
//    }
//    for (int i = 0; i < encodings[1178].size(); ++i) {
//        if (encodings[1178][i] != 0) {
//            printf("!!! 1103 encodings[1103][%d] = %d\n", i, encodings[1178][i]);
//        }
//    }

    clustering(encodings, ascendingRow, startIndexOfNonZeroRow, clusterIds);

    reorderedMatrix.reorderedRowIndices_.resize(matrix.row_);
    std::iota(reorderedMatrix.reorderedRowIndices_.begin(),
              reorderedMatrix.reorderedRowIndices_.end(),
              0); // rowIndices = {0, 1, 2, 3, ... rows-1}
    std::stable_sort(reorderedMatrix.reorderedRowIndices_.begin(),
                     reorderedMatrix.reorderedRowIndices_.end(),
                     [&clusterIds](int i, int j) { return clusterIds[i] < clusterIds[j]; });

    // Remove zero rows
    {
        startIndexOfNonZeroRow = 0;
        while (startIndexOfNonZeroRow < matrix.row_
            && matrix.rowOffsets_[reorderedMatrix.reorderedRowIndices_[startIndexOfNonZeroRow] + 1]
                - matrix.rowOffsets_[reorderedMatrix.reorderedRowIndices_[startIndexOfNonZeroRow]] == 0) {
            ++startIndexOfNonZeroRow;
        }
        reorderedMatrix.reorderedRowIndices_.erase(reorderedMatrix.reorderedRowIndices_.begin(),
                                                   reorderedMatrix.reorderedRowIndices_.begin()
                                                       + startIndexOfNonZeroRow);
    }
}