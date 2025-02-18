#include <numeric>
#include <cmath>
#include <unordered_map>
#include <omp.h>

#include "ReBELL.hpp"
#include "TensorCoreConfig.cuh"
#include "parallelAlgorithm.cuh"

void colReordering(const sparseMatrix::CSR<float> &matrix,
                   const UIN numRowPanels,
                   const std::vector<UIN> &reorderedRows,
                   std::vector<UIN> &reorderedColOffsets,
                   std::vector<UIN> &reorderedCols) {
    std::vector<UIN> numOfNonZeroColSegmentInEachRowPanel(numRowPanels, 0);
    std::vector<std::vector<UIN>>
        colsInEachRowPanel_sparse(numRowPanels, std::vector<UIN>(matrix.col())); // Containing empty columns
#pragma omp parallel for schedule(dynamic)
    for (int rowPanelId = 0; rowPanelId < numRowPanels; ++rowPanelId) {
        const UIN startIdxOfReorderedRowsCurrentRowPanel = rowPanelId * ROW_PANEL_SIZE;
        const UIN endIdxOfReorderedRowsCurrentRowPanel = std::min(
            startIdxOfReorderedRowsCurrentRowPanel + ROW_PANEL_SIZE,
            static_cast<UIN>(reorderedRows.size()));

        // Count the number of non-zero elements for each column segment
        std::vector<UIN> numOfNonZeroInEachColSegment(matrix.col(), 0);
        for (int reorderedRowIndex = startIdxOfReorderedRowsCurrentRowPanel;
             reorderedRowIndex < endIdxOfReorderedRowsCurrentRowPanel;
             ++reorderedRowIndex) { // Loop through the rows in this row panel
            const UIN row = reorderedRows[reorderedRowIndex];
            for (UIN idx = matrix.rowOffsets()[row]; idx < matrix.rowOffsets()[row + 1];
                 ++idx) { // Loop through the columns in this row
                const UIN col = matrix.colIndices()[idx];
                ++numOfNonZeroInEachColSegment[col];
            }
        }

        std::vector<UIN> &colIndicesCurrentRowPanel = colsInEachRowPanel_sparse[rowPanelId];
        std::iota(colIndicesCurrentRowPanel.begin(), colIndicesCurrentRowPanel.end(), 0);
        host::sort_by_key_descending_order(numOfNonZeroInEachColSegment.data(),
                                           numOfNonZeroInEachColSegment.data() + numOfNonZeroInEachColSegment.size(),
                                           colIndicesCurrentRowPanel.data());
        UIN numNonZeroColSegment = 0;
        while (numNonZeroColSegment < matrix.col() && numOfNonZeroInEachColSegment[numNonZeroColSegment] != 0) {
            ++numNonZeroColSegment;
        }
        numOfNonZeroColSegmentInEachRowPanel[rowPanelId] = numNonZeroColSegment;
    }

    reorderedColOffsets.resize(numRowPanels + 1);
    reorderedColOffsets[0] = 0;
    host::inclusive_scan(numOfNonZeroColSegmentInEachRowPanel.data(),
                         numOfNonZeroColSegmentInEachRowPanel.data() + numOfNonZeroColSegmentInEachRowPanel.size(),
                         reorderedColOffsets.data() + 1);

    reorderedCols.resize(reorderedColOffsets[numRowPanels]);
#pragma omp parallel for
    for (int rowPanelId = 0; rowPanelId < numRowPanels; ++rowPanelId) {
        std::copy(colsInEachRowPanel_sparse[rowPanelId].begin(),
                  colsInEachRowPanel_sparse[rowPanelId].begin()
                      + numOfNonZeroColSegmentInEachRowPanel[rowPanelId],
                  reorderedCols.begin() + reorderedColOffsets[rowPanelId]);
    }
}