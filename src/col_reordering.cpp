#include <numeric>
#include <cmath>

#include "reordering.hpp"
#include "TensorCoreConfig.cuh"
#include "parallelAlgorithm.cuh"

void col_reordering(const sparseDataType::CSR<float> &matrix, struct ReorderedMatrix &reorderedMatrix) {
    const UIN numRowPanel = std::ceil(static_cast<float>(reorderedMatrix.reorderedRowIndices_.size()) / row_panel_size);

    std::vector<UIN> numOfNonZeroColSegmentInEachRowPanel(numRowPanel, 0);
    std::vector<std::vector<UIN>>
            colIndicesInEachRowPanel_sparse(numRowPanel, std::vector<UIN>(matrix.col_)); // Containing empty columns
#pragma omp parallel for
    for (int rowPanelIdx = 0; rowPanelIdx < numRowPanel; ++rowPanelIdx) {
        const UIN startIdxOfReorderedRowIndicesInThisRowPanel = rowPanelIdx * row_panel_size;
        const UIN endIdxOfReorderedRowIndicesInThisRowPanel = std::min(
                startIdxOfReorderedRowIndicesInThisRowPanel + row_panel_size, static_cast<UIN>(reorderedMatrix.reorderedRowIndices_.size()));

        // Count the number of non-zero elements for each column segment
        std::vector<UIN> numOfNonZeroInEachColSegment(matrix.col_, 0);
        for (int idxOfSortedRowIndices = startIdxOfReorderedRowIndicesInThisRowPanel;
             idxOfSortedRowIndices < endIdxOfReorderedRowIndicesInThisRowPanel;
             ++idxOfSortedRowIndices) { // Loop through the rows in this row panel
            const UIN row = reorderedMatrix.reorderedRowIndices_[idxOfSortedRowIndices];
            for (UIN idx = matrix.rowOffsets_[row]; idx < matrix.rowOffsets_[row + 1];
                 ++idx) { // Loop through the columns in this row
                const UIN col = matrix.colIndices_[idx];
                ++numOfNonZeroInEachColSegment[col];
            }
        }

        auto &colIndicesInThisRowPanel = colIndicesInEachRowPanel_sparse[rowPanelIdx];
        std::iota(colIndicesInThisRowPanel.begin(), colIndicesInThisRowPanel.end(), 0);
        host::sort_by_key_descending_order(numOfNonZeroInEachColSegment.data(),
                                           numOfNonZeroInEachColSegment.data() + numOfNonZeroInEachColSegment.size(),
                                           colIndicesInThisRowPanel.data());
        UIN numNonZeroColSegment = 0;
        while (numNonZeroColSegment < matrix.col_ && numOfNonZeroInEachColSegment[numNonZeroColSegment] != 0) {
            ++numNonZeroColSegment;
        }
        numOfNonZeroColSegmentInEachRowPanel[rowPanelIdx] = numNonZeroColSegment;
    }

    reorderedMatrix.reorderedColIndicesOffset_.resize(numRowPanel + 1);
    reorderedMatrix.reorderedColIndicesOffset_[0] = 0;
    host::inclusive_scan(numOfNonZeroColSegmentInEachRowPanel.data(),
                         numOfNonZeroColSegmentInEachRowPanel.data() + numOfNonZeroColSegmentInEachRowPanel.size(),
                         reorderedMatrix.reorderedColIndicesOffset_.data() + 1);

    reorderedMatrix.reorderedColIndices_.resize(reorderedMatrix.reorderedColIndicesOffset_[numRowPanel]);
#pragma omp parallel for
    for (int rowPanelIdx = 0; rowPanelIdx < numRowPanel; ++rowPanelIdx) {
        std::copy(colIndicesInEachRowPanel_sparse[rowPanelIdx].begin(),
                  colIndicesInEachRowPanel_sparse[rowPanelIdx].begin()
                  + numOfNonZeroColSegmentInEachRowPanel[rowPanelIdx],
                  reorderedMatrix.reorderedColIndices_.begin() +
                  reorderedMatrix.reorderedColIndicesOffset_[rowPanelIdx]);
    }
}