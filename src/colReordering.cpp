#include <numeric>
#include <cmath>
#include <unordered_map>

#include "ReBELL.hpp"
#include "TensorCoreConfig.cuh"
#include "parallelAlgorithm.cuh"

void ReBELL::colReordering(const sparseDataType::CSR<float> &matrix) {
    numRowPanels_ = std::ceil(static_cast<float>(reorderedRows_.size()) / row_panel_size);
    std::vector<UIN> numOfNonZeroColSegmentInEachRowPanel(numRowPanels_, 0);
    std::vector<std::vector<UIN>>
        colsInEachRowPanel_sparse(numRowPanels_, std::vector<UIN>(matrix.col_)); // Containing empty columns
#pragma omp parallel for
    for (int rowPanelIdx = 0; rowPanelIdx < numRowPanels_; ++rowPanelIdx) {
        const UIN startIdxOfReorderedRowsCurrentRowPanel = rowPanelIdx * row_panel_size;
        const UIN endIdxOfReorderedRowsCurrentRowPanel = std::min(
            startIdxOfReorderedRowsCurrentRowPanel + row_panel_size,
            static_cast<UIN>(reorderedRows_.size()));

        // Count the number of non-zero elements for each column segment
        std::vector<UIN> numOfNonZeroInEachColSegment(matrix.col_, 0);
        for (int reorderedRowIndex = startIdxOfReorderedRowsCurrentRowPanel;
             reorderedRowIndex < endIdxOfReorderedRowsCurrentRowPanel;
             ++reorderedRowIndex) { // Loop through the rows in this row panel
            const UIN row = reorderedRows_[reorderedRowIndex];
            for (UIN idx = matrix.rowOffsets_[row]; idx < matrix.rowOffsets_[row + 1];
                 ++idx) { // Loop through the columns in this row
                const UIN col = matrix.colIndices_[idx];
                ++numOfNonZeroInEachColSegment[col];
            }
        }

        auto &colIndicesCurrentRowPanel = colsInEachRowPanel_sparse[rowPanelIdx];
        std::iota(colIndicesCurrentRowPanel.begin(), colIndicesCurrentRowPanel.end(), 0);
        host::sort_by_key_descending_order(numOfNonZeroInEachColSegment.data(),
                                           numOfNonZeroInEachColSegment.data() + numOfNonZeroInEachColSegment.size(),
                                           colIndicesCurrentRowPanel.data());
        UIN numNonZeroColSegment = 0;
        while (numNonZeroColSegment < matrix.col_ && numOfNonZeroInEachColSegment[numNonZeroColSegment] != 0) {
            ++numNonZeroColSegment;
        }
        numOfNonZeroColSegmentInEachRowPanel[rowPanelIdx] = numNonZeroColSegment;
    }

    reorderedColsOffset_.resize(numRowPanels_ + 1);
    reorderedColsOffset_[0] = 0;
    host::inclusive_scan(numOfNonZeroColSegmentInEachRowPanel.data(),
                         numOfNonZeroColSegmentInEachRowPanel.data() + numOfNonZeroColSegmentInEachRowPanel.size(),
                         reorderedColsOffset_.data() + 1);

    reorderedCols_.resize(reorderedColsOffset_[numRowPanels_]);
#pragma omp parallel for
    for (int rowPanelIdx = 0; rowPanelIdx < numRowPanels_; ++rowPanelIdx) {
        std::copy(colsInEachRowPanel_sparse[rowPanelIdx].begin(),
                  colsInEachRowPanel_sparse[rowPanelIdx].begin()
                      + numOfNonZeroColSegmentInEachRowPanel[rowPanelIdx],
                  reorderedCols_.begin() + reorderedColsOffset_[rowPanelIdx]);
    }
}