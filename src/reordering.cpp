#include <cstdio>
#include <unordered_map>
#include <cmath>
#include <unordered_map>
#include <limits>

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

    // Error check
    bool isCorrect = check_colReordering(matrix, reorderedMatrix);
    if (!isCorrect) {
        std::cerr << "Error! The col reordering is incorrect!" << std::endl;
    }

    return reorderedMatrix;
}

ReBELL::ReBELL(const sparseDataType::CSR<float> &csrMatrix) {

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
        numBlockInEachRowPanel[rowPanelIdx] = std::ceil(static_cast<float>(numColIndices) / block_col_size);
    }
    blockRowOffsets_.resize(numRowPanel + 1);
    blockRowOffsets_[0] = 0;
    host::inclusive_scan(numBlockInEachRowPanel.data(),
                         numBlockInEachRowPanel.data() + numBlockInEachRowPanel.size(),
                         blockRowOffsets_.data() + 1);

    // initialize blockValues_
    blockValues_.resize(blockRowOffsets_.back() * block_size);
    host::fill_n(blockValues_.data(), blockValues_.size(), MAX_UIN);
#pragma omp parallel for
    for (int idxOfRowIndices = 0; idxOfRowIndices < reorderedRowIndices_.size(); ++idxOfRowIndices) {
        const UIN row = reorderedRowIndices_[idxOfRowIndices];

        std::unordered_map<UIN, UIN> colAndIndexMap;
        for (int idxOfOriginalMatrix = csrMatrix.rowOffsets_[row]; idxOfOriginalMatrix < csrMatrix.rowOffsets_[row + 1];
             ++idxOfOriginalMatrix) {
            colAndIndexMap[csrMatrix.colIndices_[idxOfOriginalMatrix]] = idxOfOriginalMatrix;
        }

        const UIN rowPanelIdx = idxOfRowIndices / row_panel_size;
        const UIN localRowIdx = idxOfRowIndices % row_panel_size;
        const UIN startIndexOfBlockValuesInThisRowPanel = blockRowOffsets_[rowPanelIdx] * block_size;
        // Iterate over the columns in the row panel
        for (int iter = 0, idxOfReorderedColIndices = reorderedColIndicesOffset_[rowPanelIdx];
             idxOfReorderedColIndices < reorderedColIndicesOffset_[rowPanelIdx + 1];
             ++iter, ++idxOfReorderedColIndices) {
            const UIN localColIdx = iter % block_col_size;
            const UIN colBlockId = iter / block_col_size;
            const UIN idxOfBlockValues = startIndexOfBlockValuesInThisRowPanel + colBlockId * block_size +
                                         localRowIdx * block_col_size + localColIdx;

            const UIN col = reorderedColIndices_[idxOfReorderedColIndices];
            const auto findIter = colAndIndexMap.find(col);
            if (findIter != colAndIndexMap.end()) {
                blockValues_[idxOfBlockValues] = findIter->second;
            }
        }
    }
}

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