#include <numeric>

#include "reordering.hpp"
#include "TensorCoreConfig.cuh"
#include "parallelAlgorithm.cuh"

const int row_panel_size = WMMA_M;

void col_reordering(const sparseDataType::CSR &matrix, struct ReorderedMatrix &reorderedMatrix) {
    UIN numRowPanel = std::ceil(static_cast<float>(reorderedMatrix.rowIndices_.size()) / row_panel_size);

    std::vector<UIN> numOfNonZeroColSegmentInEachRowPanel(numRowPanel, 0);
    std::vector<std::vector<UIN>>
        colIndicesInEachRowPanel_sparse(numRowPanel, std::vector<UIN>(matrix.col_)); // Containing empty columns
#pragma omp parallel for
    for (int rowPanelIdx = 0; rowPanelIdx < numRowPanel; ++rowPanelIdx) {
        const UIN startIdxInThisRowPanel = rowPanelIdx * row_panel_size;
        const UIN endIdxInThisRowPanel = std::min(startIdxInThisRowPanel + row_panel_size, matrix.row_);

        // Count the number of non-zero elements for each column segment
        std::vector<UIN> numOfNonZeroInEachColSegment(matrix.col_, 0);
        for (int idxOfSortedRowIndices = startIdxInThisRowPanel; idxOfSortedRowIndices < endIdxInThisRowPanel;
             ++idxOfSortedRowIndices) { // Loop through the rows in this row panel
            const UIN row = reorderedMatrix.rowIndices_[idxOfSortedRowIndices];
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

    reorderedMatrix.colIndicesOffset_.resize(numRowPanel + 1);
    reorderedMatrix.colIndicesOffset_[0] = 0;
    host::inclusive_scan(numOfNonZeroColSegmentInEachRowPanel.data(),
                         numOfNonZeroColSegmentInEachRowPanel.data() + numOfNonZeroColSegmentInEachRowPanel.size(),
                         reorderedMatrix.colIndicesOffset_.data() + 1);

    reorderedMatrix.colIndicesInEachRowPanel_.resize(reorderedMatrix.colIndicesOffset_[numRowPanel]);
#pragma omp parallel for
    for (int rowPanelIdx = 0; rowPanelIdx < numRowPanel; ++rowPanelIdx) {
        std::copy(colIndicesInEachRowPanel_sparse[rowPanelIdx].begin(),
                  colIndicesInEachRowPanel_sparse[rowPanelIdx].begin()
                      + numOfNonZeroColSegmentInEachRowPanel[rowPanelIdx],
                  reorderedMatrix.colIndicesInEachRowPanel_.begin() + reorderedMatrix.colIndicesOffset_[rowPanelIdx]);
    }
}