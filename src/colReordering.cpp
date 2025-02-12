#include <numeric>
#include <cmath>
#include <unordered_map>

#include "ReBELL.hpp"
#include "TensorCoreConfig.cuh"
#include "parallelAlgorithm.cuh"

void ReBELL::colReordering(const sparseMatrix::CSR<float> &matrix) {
    numRowPanels_ = std::ceil(static_cast<float>(reorderedRows_.size()) / ROW_PANEL_SIZE);
    std::vector<UIN> numOfNonZeroColSegmentInEachRowPanel(numRowPanels_, 0);
    std::vector<std::vector<UIN>>
        colsInEachRowPanel_sparse(numRowPanels_, std::vector<UIN>(matrix.col_)); // Containing empty columns
#pragma omp parallel for
    for (int rowPanelId = 0; rowPanelId < numRowPanels_; ++rowPanelId) {
        const UIN startIdxOfReorderedRowsCurrentRowPanel = rowPanelId * ROW_PANEL_SIZE;
        const UIN endIdxOfReorderedRowsCurrentRowPanel = std::min(
            startIdxOfReorderedRowsCurrentRowPanel + ROW_PANEL_SIZE,
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

        auto &colIndicesCurrentRowPanel = colsInEachRowPanel_sparse[rowPanelId];
        std::iota(colIndicesCurrentRowPanel.begin(), colIndicesCurrentRowPanel.end(), 0);
        host::sort_by_key_descending_order(numOfNonZeroInEachColSegment.data(),
                                           numOfNonZeroInEachColSegment.data() + numOfNonZeroInEachColSegment.size(),
                                           colIndicesCurrentRowPanel.data());
        UIN numNonZeroColSegment = 0;
        while (numNonZeroColSegment < matrix.col_ && numOfNonZeroInEachColSegment[numNonZeroColSegment] != 0) {
            ++numNonZeroColSegment;
        }
        numOfNonZeroColSegmentInEachRowPanel[rowPanelId] = numNonZeroColSegment;
    }

    reorderedColOffsets_.resize(numRowPanels_ + 1);
    reorderedColOffsets_[0] = 0;
    host::inclusive_scan(numOfNonZeroColSegmentInEachRowPanel.data(),
                         numOfNonZeroColSegmentInEachRowPanel.data() + numOfNonZeroColSegmentInEachRowPanel.size(),
                         reorderedColOffsets_.data() + 1);

    reorderedCols_.resize(reorderedColOffsets_[numRowPanels_]);
#pragma omp parallel for
    for (int rowPanelId = 0; rowPanelId < numRowPanels_; ++rowPanelId) {
        std::copy(colsInEachRowPanel_sparse[rowPanelId].begin(),
                  colsInEachRowPanel_sparse[rowPanelId].begin()
                      + numOfNonZeroColSegmentInEachRowPanel[rowPanelId],
                  reorderedCols_.begin() + reorderedColOffsets_[rowPanelId]);
    }
}