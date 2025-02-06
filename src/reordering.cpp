#include <cstdio>
#include <unordered_map>
#include <cmath>
#include <unordered_map>

#include "reordering.hpp"
#include "CudaTimeCalculator.cuh"
#include "parallelAlgorithm.cuh"

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

template<typename T>
ReBELL<T>::ReBELL(const sparseDataType::CSR<T> &csrMatrix) {
    this->row_ = csrMatrix.row_;
    this->col_ = csrMatrix.col_;
    this->nnz_ = csrMatrix.nnz_;

    ReorderedMatrix reorderedMatrix = reordering(csrMatrix);

    reorderedRowIndices_ = reorderedMatrix.reorderedRowIndices_;
    reorderedColIndices_ = reorderedMatrix.reorderedColIndices_;
    reorderedColIndicesOffset_ = reorderedMatrix.reorderedColIndicesOffset_;

    // initialize blockRowOffsets_
    const UIN numRowPanel = std::ceil(static_cast<float>(reorderedMatrix.reorderedRowIndices_.size()) / row_panel_size);
    std::vector<UIN> numBlockInEachRowPanel(numRowPanel);
#pragma omp parallel for
    for (int rowPanelIdx = 0; rowPanelIdx < numRowPanel; ++rowPanelIdx) {
        const UIN numColIndices = reorderedColIndicesOffset_[rowPanelIdx + 1] - reorderedColIndicesOffset_[rowPanelIdx];
        numBlockInEachRowPanel[rowPanelIdx] = std::ceil(static_cast<float>(numColIndices) / col_block_size);
    }
    this->blockRowOffsets_.resize(numRowPanel + 1);
    this->blockRowOffsets_[0] = 0;
    host::inclusive_scan(numBlockInEachRowPanel.data(),
                         numBlockInEachRowPanel.data() + numBlockInEachRowPanel.size(),
                         this->blockRowOffsets_.data() + 1);

    // initialize blockColIndices_

    // initialize blockValues_
    this->blockValues_.resize(this->blockRowOffsets_.back() * block_size);
#pragma omp parallel for
    for (int idxOfRowIndices = 0; idxOfRowIndices < reorderedRowIndices_.size(); ++idxOfRowIndices) {
        const UIN row = reorderedRowIndices_[idxOfRowIndices];

        std::unordered_map<UIN, T> colAndValueMap;
        for (int idxOfOriginalMatrix = csrMatrix.rowOffsets_[row]; idxOfOriginalMatrix < csrMatrix.rowOffsets_[row + 1];
             ++idxOfOriginalMatrix) {
            colAndValueMap[csrMatrix.colIndices_[idxOfOriginalMatrix]] = csrMatrix.values_[idxOfOriginalMatrix];
        }

        const UIN rowPanelIdx = idxOfRowIndices / row_panel_size;
        const UIN localRowIdx = idxOfRowIndices % row_panel_size;
        const UIN startIndexOfBlockValuesInThisRowPanel = this->blockRowOffsets_[rowPanelIdx] * block_size;
        for (int idxOfReorderedColIndices = reorderedColIndicesOffset_[rowPanelIdx];
             idxOfReorderedColIndices < reorderedColIndicesOffset_[rowPanelIdx + 1]; ++idxOfReorderedColIndices) {
            const UIN localColIdx = idxOfReorderedColIndices % col_block_size;
            const UIN colBlockId = idxOfReorderedColIndices / col_block_size;
            const UIN idxOfBlockValues = startIndexOfBlockValuesInThisRowPanel + colBlockId * block_size +
                localRowIdx * col_block_size + localColIdx;

            const UIN col = reorderedColIndices_[idxOfReorderedColIndices];
            const auto findIter = colAndValueMap.find(col);
            if (findIter != colAndValueMap.end()) {
                this->blockValues_[idxOfBlockValues] = findIter->second;
            } else {
                this->blockValues_[idxOfBlockValues] = static_cast<T>(0);
            }
        }
    }
}

template
class ReBELL<float>;

bool testReorderedMatrixCorrectness(const sparseDataType::CSR<float> &matrix, const ReorderedMatrix &reorderedMatrix) {
    for (int reorderedRowIdx = 0; reorderedRowIdx < reorderedMatrix.reorderedRowIndices_.size(); ++reorderedRowIdx) {
        const UIN row = reorderedMatrix.reorderedRowIndices_[reorderedRowIdx];

        std::unordered_map<UIN, float> colAndValueMap;
        for (int matrixColIdx = matrix.rowOffsets_[row]; matrixColIdx < matrix.rowOffsets_[row + 1]; ++matrixColIdx) {
            const UIN col = matrix.colIndices_[matrixColIdx];
            const float value = matrix.values_[matrixColIdx];
            colAndValueMap[col] = value;
        }

    }

    return true;
}