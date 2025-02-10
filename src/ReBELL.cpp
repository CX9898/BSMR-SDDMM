#include <cstdio>
#include <unordered_map>
#include <cmath>
#include <unordered_map>
#include <limits>

#include "ReBELL.hpp"
#include "CudaTimeCalculator.cuh"
#include "parallelAlgorithm.cuh"

ReBELL::ReBELL(const sparseDataType::CSR<float> &matrix) {

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();
    rowReordering(matrix);
    timeCalculator.endClock();
    printf("row_reordering time : %f ms\n", timeCalculator.getTime());

    timeCalculator.startClock();
    colReordering(matrix);
    timeCalculator.endClock();
    printf("col_reordering time : %f ms\n", timeCalculator.getTime());

    // initialize blockRowOffsets_
    const UIN numRowPanel = std::ceil(static_cast<float>(reorderedRowIndices_.size()) / row_panel_size);
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
        for (int idxOfOriginalMatrix = matrix.rowOffsets_[row]; idxOfOriginalMatrix < matrix.rowOffsets_[row + 1];
             ++idxOfOriginalMatrix) {
            colAndIndexMap[matrix.colIndices_[idxOfOriginalMatrix]] = idxOfOriginalMatrix;
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