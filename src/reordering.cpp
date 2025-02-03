#include <cstdio>
#include <unordered_map>

#include "reordering.hpp"
#include "CudaTimeCalculator.cuh"

ReorderedMatrix reordering(const sparseDataType::CSR<float> &matrix) {
    ReorderedMatrix reorderedMatrix;

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();
    row_reordering(matrix, reorderedMatrix);
    timeCalculator.endClock();
    printf("row_reordering time : %f ms\n", timeCalculator.getTime());

    timeCalculator.startClock();
    col_reordering(matrix, reorderedMatrix);
    timeCalculator.endClock();
    printf("col_reordering time : %f ms\n", timeCalculator.getTime());

    return reorderedMatrix;
}

bool testReorderedMatrixCorrectness(const sparseDataType::CSR<float> &matrix, const ReorderedMatrix &reorderedMatrix) {
    for(int reorderedRowIdx = 0; reorderedRowIdx < reorderedMatrix.rowIndices_.size(); ++reorderedRowIdx){
        const UIN row = reorderedMatrix.rowIndices_[reorderedRowIdx];

        std::unordered_map<UIN, float> colAndValuesMap;
        for(int matrixColIdx = matrix.rowOffsets_[row]; matrixColIdx < matrix.rowOffsets_[row + 1]; ++matrixColIdx){
            const UIN col = matrix.colIndices_[matrixColIdx];
            const float value = matrix.values_[matrixColIdx];
            colAndValuesMap[col] = value;
        }


    }
}