#include <numeric>
#include <cmath>
#include <unordered_map>
#include <omp.h>

#include "ReBELL.hpp"
#include "TensorCoreConfig.cuh"
#include "parallelAlgorithm.cuh"
#include "sddmmKernel.cuh"

// return the number of dense column segments and the number of sparse column segments
std::pair<UIN, UIN> analysisDescendingOrderColSegment(const UIN dense_column_segment_threshold,
                                                      const std::vector<UIN> &numOfNonZeroInEachColSegment) {
    UIN numNonZeroColSegment = 0;
    UIN numDenseColSegment = 0;

    while (numNonZeroColSegment < numOfNonZeroInEachColSegment.size()
        && numOfNonZeroInEachColSegment[numNonZeroColSegment] > 0) {

        if (numOfNonZeroInEachColSegment[numNonZeroColSegment] >= dense_column_segment_threshold) {
            ++numDenseColSegment;
        }

        ++numNonZeroColSegment;
    }

    const UIN remainderNumber = numDenseColSegment % each_thread_block_counts_the_number_Of_cols;
    if (remainderNumber > each_thread_block_counts_the_number_Of_cols / 2) {
        numDenseColSegment = std::min(static_cast<UIN>(numOfNonZeroInEachColSegment.size()),
                                      numDenseColSegment - remainderNumber + BLOCK_COL_SIZE);
    } else {
        numDenseColSegment -= remainderNumber;
    }

    const UIN numSparseColSegment = numNonZeroColSegment - numDenseColSegment;
    return std::make_pair(numDenseColSegment, numSparseColSegment);
}

// Divide rows into row panels and columns reordered in each row panel. After the columns reordered, the columns are divided into dense and sparse residual columns.
void colReordering(const sparseMatrix::CSR<float> &matrix,
                   const UIN numRowPanels,
                   const std::vector<UIN> &reorderedRows,
                   const UIN dense_column_segment_threshold,
                   std::vector<UIN> &denseCols,
                   std::vector<UIN> &denseColOffsets,
                   std::vector<UIN> &sparseCols,
                   std::vector<UIN> &sparseColOffsets,
                   std::vector<UIN> &sparsePartDataOffsets) {
    std::vector<UIN> numOfDenseColSegmentInEachRowPanel(numRowPanels, 0);
    std::vector<UIN> numOfSparseColSegmentInEachRowPanel(numRowPanels, 0);
    std::vector<std::vector<UIN>> colsInEachRowPanel(
        numRowPanels, std::vector<UIN>(matrix.col())); // Containing empty columns
    std::vector<UIN> numOfSparsePartDataInEachRowPanel(numRowPanels, 0);
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

        std::vector<UIN> &colIndices = colsInEachRowPanel[rowPanelId];
        std::iota(colIndices.begin(), colIndices.end(), 0);
        host::sort_by_key_descending_order(numOfNonZeroInEachColSegment.data(),
                                           numOfNonZeroInEachColSegment.data() + numOfNonZeroInEachColSegment.size(),
                                           colIndices.data());

        const auto [numDenseColSegment, numSparseColSegment] =
            analysisDescendingOrderColSegment(dense_column_segment_threshold, numOfNonZeroInEachColSegment);

        UIN numSparsePartData = 0;
        for (int i = numDenseColSegment; i < numDenseColSegment + numSparseColSegment; ++i) {
            numSparsePartData += numOfNonZeroInEachColSegment[i];
        }
        numOfDenseColSegmentInEachRowPanel[rowPanelId] = numDenseColSegment;
        numOfSparseColSegmentInEachRowPanel[rowPanelId] = numSparseColSegment;
        numOfSparsePartDataInEachRowPanel[rowPanelId] = numSparsePartData;
    }

    // Initialize the sparsePartDataOffsets
    sparsePartDataOffsets.resize(numRowPanels + 1);
    sparsePartDataOffsets[0] = 0;
    host::inclusive_scan(numOfSparsePartDataInEachRowPanel.data(),
                         numOfSparsePartDataInEachRowPanel.data() + numOfSparsePartDataInEachRowPanel.size(),
                         sparsePartDataOffsets.data() + 1);

    // Initialize the denseColOffsets
    denseColOffsets.resize(numRowPanels + 1);
    denseColOffsets[0] = 0;
    host::inclusive_scan(numOfDenseColSegmentInEachRowPanel.data(),
                         numOfDenseColSegmentInEachRowPanel.data() + numOfDenseColSegmentInEachRowPanel.size(),
                         denseColOffsets.data() + 1);

    // Initialize the sparseColOffsets
    sparseColOffsets.resize(numRowPanels + 1);
    sparseColOffsets[0] = 0;
    host::inclusive_scan(numOfSparseColSegmentInEachRowPanel.data(),
                         numOfSparseColSegmentInEachRowPanel.data() + numOfSparseColSegmentInEachRowPanel.size(),
                         sparseColOffsets.data() + 1);

    // Initialize the denseCols,sparseCols
    denseCols.resize(denseColOffsets[numRowPanels]);
    sparseCols.resize(sparseColOffsets[numRowPanels]);
#pragma omp parallel for
    for (int rowPanelId = 0; rowPanelId < numRowPanels; ++rowPanelId) {
        std::copy(colsInEachRowPanel[rowPanelId].begin(),
                  colsInEachRowPanel[rowPanelId].begin() + numOfDenseColSegmentInEachRowPanel[rowPanelId],
                  denseCols.begin() + denseColOffsets[rowPanelId]);

        std::copy(colsInEachRowPanel[rowPanelId].begin() + numOfDenseColSegmentInEachRowPanel[rowPanelId],
                  colsInEachRowPanel[rowPanelId].begin()
                      + numOfDenseColSegmentInEachRowPanel[rowPanelId]
                      + numOfSparseColSegmentInEachRowPanel[rowPanelId],
                  sparseCols.begin() + sparseColOffsets[rowPanelId]);
    }
}

// Divide rows into row panels and columns reordered in each row panel.
void colReordering(const sparseMatrix::CSR<float> &matrix,
                   const UIN numRowPanels,
                   const std::vector<UIN> &reorderedRows,
                   std::vector<UIN> &reorderedCols,
                   std::vector<UIN> &reorderedColOffsets) {
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
                  colsInEachRowPanel_sparse[rowPanelId].begin() + numOfNonZeroColSegmentInEachRowPanel[rowPanelId],
                  reorderedCols.begin() + reorderedColOffsets[rowPanelId]);
    }
}