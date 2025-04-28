#include <cstdio>
#include <cmath>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include <limits>
#include <numeric>
#include <array>

#include <omp.h>

#include "ReBELL.hpp"
#include "CudaTimeCalculator.cuh"
#include "parallelAlgorithm.cuh"
#include "sddmmKernel.cuh"

ReBELL::ReBELL(const int K, const sparseMatrix::CSR<float> &matrix) {

//    // Calculate the dense column segment threshold
//    const float sparsityThreshold = (0.00219 * K + 79.81) / 100;
//    const UIN minNumNonZeroPerColSegment =
//        std::ceil(BLOCK_SIZE * (1 - sparsityThreshold) / static_cast<float>(BLOCK_COL_SIZE));
//    printf("sparsityThreshold = %f, minNumNonZeroCurrentSparsity : %d\n",
//           sparsityThreshold, minNumNonZeroPerColSegment);
//    dense_column_segment_threshold_ = minNumNonZeroPerColSegment > 0 ? minNumNonZeroPerColSegment : 1;
    dense_column_segment_threshold_ = 4;

    std::vector<UIN> reorderedRows;

    std::vector<UIN> denseColOffsets;
    std::vector<UIN> denseCols;
    std::vector<UIN> blockOffsets;
    std::vector<UIN> blockValues;

    std::vector<UIN> sparseValueOffsets;
    std::vector<UIN> sparseValues;
    std::vector<UIN> sparseRelativeRows;
    std::vector<UIN> sparseColIndices;

    // Row reordering
    float rowReordering_time;
    const UIN blockSize = calculateBlockSize(matrix);
//    noReorderRow(matrix, reorderedRows_, rowReordering_time);
    reorderedRows = bsa_rowReordering_gpu(matrix,
                                          row_similarity_threshold_alpha,
                                          blockSize,
                                          rowReordering_time);
//    std::vector<int> rows = bsa_rowReordering_cpu(matrix,
//                                                  row_similarity_threshold_alpha,
//                                                  blockSize,
//                                                  rowReordering_time);
//    reorderedRows_.resize(rows.size());
//    for(int i =0; i< rows.size();++i){
//        reorderedRows_[i] = rows[i];
//    }
//    rowReordering_cpu(matrix, reorderedRows_, rowReordering_time);
//    rowReordering_gpu(matrix, row_similarity_threshold_alpha, blockSize, reorderedRows_, rowReordering_time);

    printf("rowReordering time : %f ms\n", rowReordering_time);

    numRowPanels_ = std::ceil(static_cast<float>(reorderedRows.size()) / ROW_PANEL_SIZE);
    printf("numRowPanels : %d\n", numRowPanels_);

    // Column reordering
    std::vector<UIN> sparseCols;
    std::vector<UIN> sparseColOffsets;
    float colReordering_time;
    colReordering_cpu(matrix,
                      numRowPanels_,
                      reorderedRows,
                      dense_column_segment_threshold_,
                      denseCols,
                      denseColOffsets,
                      sparseCols,
                      sparseColOffsets,
                      sparseValueOffsets,
                      colReordering_time);
    printf("colReordering time : %f ms\n", colReordering_time);

    // Calculate the maximum number of dense column blocks in a row panel
    maxNumDenseColBlocks_ = 0;
#pragma omp parallel for reduction(max : maxNumDenseColBlocks_)
    for (int rowPanelId = 0; rowPanelId < numRowPanels_; ++rowPanelId) {
        const UIN numBlocksCurrentRowPanel = std::ceil(
            static_cast<float>(denseColOffsets[rowPanelId + 1] - denseColOffsets[rowPanelId])
                / BLOCK_COL_SIZE);
        maxNumDenseColBlocks_ = std::max(maxNumDenseColBlocks_, numBlocksCurrentRowPanel);
    }

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();

    // initialize blockRowOffsets_
    std::vector<UIN> numBlockInEachRowPanel(numRowPanels_);
#pragma omp parallel for
    for (int rowPanelId = 0; rowPanelId < numRowPanels_; ++rowPanelId) {
        const UIN numColIndices = denseColOffsets[rowPanelId + 1] - denseColOffsets[rowPanelId];
        numBlockInEachRowPanel[rowPanelId] = std::ceil(static_cast<float>(numColIndices) / BLOCK_COL_SIZE);
    }

    blockOffsets.resize(numRowPanels_ + 1);
    blockOffsets[0] = 0;
    host::inclusive_scan(numBlockInEachRowPanel.data(),
                         numBlockInEachRowPanel.data() + numBlockInEachRowPanel.size(),
                         blockOffsets.data() + 1);

    sparseRelativeRows.resize(sparseValueOffsets.back());
    sparseValues.resize(sparseValueOffsets.back());
    sparseColIndices.resize(sparseValueOffsets.back());

    // initialize blockValues_
    blockValues.resize(blockOffsets.back() * BLOCK_SIZE);
    host::fill_n(blockValues.data(), blockValues.size(), NULL_VALUE);
#pragma omp parallel for
    for (int indexOfReorderedRows = 0; indexOfReorderedRows < reorderedRows.size(); ++indexOfReorderedRows) {
        const UIN row = reorderedRows[indexOfReorderedRows];

        std::unordered_map<UIN, UIN> colToIndexOfOriginalMatrixMap;
        for (int idxOfOriginalMatrix = matrix.rowOffsets()[row]; idxOfOriginalMatrix < matrix.rowOffsets()[row + 1];
             ++idxOfOriginalMatrix) {
            colToIndexOfOriginalMatrixMap[matrix.colIndices()[idxOfOriginalMatrix]] = idxOfOriginalMatrix;
        }

        const UIN rowPanelId = indexOfReorderedRows / ROW_PANEL_SIZE;
        const UIN localRowId = indexOfReorderedRows % ROW_PANEL_SIZE;
        const UIN startIndexOfBlockValuesCurrentRowPanel = blockOffsets[rowPanelId] * BLOCK_SIZE;

        // Iterate over the dense columns in the row panel
        for (int count = 0, indexOfReorderedCols = denseColOffsets[rowPanelId];
             indexOfReorderedCols < denseColOffsets[rowPanelId + 1];
             ++count, ++indexOfReorderedCols) {
            const UIN localColId = count % BLOCK_COL_SIZE;
            const UIN colBlockId = count / BLOCK_COL_SIZE;
            const UIN idxOfBlockValues = startIndexOfBlockValuesCurrentRowPanel + colBlockId * BLOCK_SIZE +
                localRowId * BLOCK_COL_SIZE + localColId;

            const UIN col = denseCols[indexOfReorderedCols];
            const auto findIter = colToIndexOfOriginalMatrixMap.find(col);
            if (findIter != colToIndexOfOriginalMatrixMap.end()) {
                blockValues[idxOfBlockValues] = findIter->second;
            }
        }
    }

    // Initialize sparse part data
#pragma omp parallel for
    for (int rowPanelId = 0; rowPanelId < numRowPanels_; ++rowPanelId) {
        std::unordered_map<UIN, std::vector<std::array<UIN, 2>>> colToRelativeRowAndOriginIndexMap;

        const UIN startIndex = rowPanelId * ROW_PANEL_SIZE;
        const UIN endIndex = std::min(startIndex + ROW_PANEL_SIZE, static_cast<UIN>(reorderedRows.size()));

        for (int indexOfReorderedRows = startIndex; indexOfReorderedRows < endIndex; ++indexOfReorderedRows) {
            const UIN row = reorderedRows[indexOfReorderedRows];
            for (int idx = matrix.rowOffsets()[row]; idx < matrix.rowOffsets()[row + 1]; ++idx) {
                const UIN col = matrix.colIndices()[idx];
                std::array<UIN, 2> relativeRowAndOriginIndex =
                    {static_cast<UIN>(indexOfReorderedRows % ROW_PANEL_SIZE), static_cast<UIN>(idx)};

                auto findIter = colToRelativeRowAndOriginIndexMap.find(col);
                if (findIter == colToRelativeRowAndOriginIndexMap.end()) {
                    colToRelativeRowAndOriginIndexMap[col] = {relativeRowAndOriginIndex};
                } else {
                    findIter->second.push_back(relativeRowAndOriginIndex);
                }
            }
        }

        UIN count = 0;
        const UIN startSparsePartIndex = sparseValueOffsets[rowPanelId];
        // Iterate over the sparse columns in the row panel
        for (int indexOfReorderedCols = sparseColOffsets[rowPanelId];
             indexOfReorderedCols < sparseColOffsets[rowPanelId + 1]; ++indexOfReorderedCols) {
            const UIN col = sparseCols[indexOfReorderedCols];

            const auto findIter = colToRelativeRowAndOriginIndexMap.find(col);
            if (findIter != colToRelativeRowAndOriginIndexMap.end()) {
                for (const std::array<UIN, 2> &iter : findIter->second) {
                    sparseRelativeRows[startSparsePartIndex + count] = iter[0];
                    sparseValues[startSparsePartIndex + count] = iter[1];
                    sparseColIndices[startSparsePartIndex + count] = col;

                    ++count;
                }
            }
        }

    }

    // Calculate the maximum number of sparse column blocks in a row panel
    maxNumSparseColBlocks_ = 0;
#pragma omp parallel for reduction(max : maxNumSparseColBlocks_)
    for (int rowPanelId = 0; rowPanelId < numRowPanels_; ++rowPanelId) {
        const UIN numSparseData = sparseValueOffsets[rowPanelId + 1] - sparseValueOffsets[rowPanelId];
        const UIN numBlocksCurrentRowPanel = std::ceil(
            static_cast<float>(numSparseData) / sddmm_sparse_block_each_thread_block_counts_the_number_Of_data);
        maxNumSparseColBlocks_ = std::max(maxNumSparseColBlocks_, numBlocksCurrentRowPanel);
    }

//    // Try to optimize
//#pragma omp parallel for
//    for (int rowPanelId = 0; rowPanelId < numRowPanels_; ++rowPanelId) {
//        const UIN startIndex = sparseDataOffsets[rowPanelId];
//        const UIN endIndex = sparseDataOffsets[rowPanelId + 1];
//
//        host::sort_by_key_for_multiple_vectors(sparseColIndices_.data() + startIndex,
//                                               sparseColIndices_.data() + endIndex,
//                                               sparseRelativeRows_.data() + startIndex,
//                                               sparseData_.data() + startIndex);
//    }

    timeCalculator.endClock();
    float bell_time = timeCalculator.getTime();
    printf("bell time : %f ms\n", bell_time);
    time_ = rowReordering_time + colReordering_time + bell_time;

    // Copy data to device
    h2d(reorderedRows_, reorderedRows);
    h2d(denseColOffsets_, denseColOffsets);
    h2d(denseCols_, denseCols);
    h2d(blockOffsets_, blockOffsets);
    h2d(blockValues_, blockValues);
    h2d(sparseValueOffsets_, sparseValueOffsets);
    h2d(sparseValues_, sparseValues);
    h2d(sparseRelativeRows_, sparseRelativeRows);
    h2d(sparseColIndices_, sparseColIndices);
}

UIN ReBELL::getNumSparseBlocks() const {
    return sparseValueOffsets().back_data()
        / static_cast<float>(sddmm_sparse_block_each_thread_block_counts_the_number_Of_data);
}

UIN ReBELL::calculateRowPanelIdByBlockValuesIndex(UIN blockValueIndex) const {
    std::vector<UIN> blockOffsets;
    d2h(blockOffsets, blockOffsets_);

    UIN rowPanelId = 0;
    while (rowPanelId + 1 < blockOffsets.size()) {
        if (blockValueIndex < blockOffsets[rowPanelId + 1] * BLOCK_SIZE) {
            break;
        }
        ++rowPanelId;
    }
    return rowPanelId;
}

UIN ReBELL::calculateRowPanelIdByColIndex(UIN reorderedColIndex) const {
    std::vector<UIN> denseColOffsets;
    d2h(denseColOffsets, denseColOffsets_);

    UIN rowPanelId = 0;
    while (rowPanelId + 1 < denseColOffsets.size()) {
        if (reorderedColIndex < denseColOffsets[rowPanelId + 1]) {
            break;
        }
        ++rowPanelId;
    }
    return rowPanelId;
}

std::pair<UIN, UIN> ReBELL::calculateLocalRowColByBlockValueIndex(UIN blockValueIndex) const {
    const UIN rowPanelId = calculateRowPanelIdByBlockValuesIndex(blockValueIndex);
    const UIN startIndexOfBlockValuesCurrentRowPanel = blockOffsets()[rowPanelId] * BLOCK_SIZE;
    const UIN localIndex = (blockValueIndex - startIndexOfBlockValuesCurrentRowPanel) % BLOCK_SIZE;
    const UIN localRowId = localIndex / BLOCK_COL_SIZE;
    const UIN localColId = localIndex % BLOCK_COL_SIZE;
    return std::make_pair(localRowId, localColId);
}

std::pair<UIN, UIN> ReBELL::calculateRowColByBlockValueIndex(UIN blockValueIndex) const {
    std::vector<UIN> blockOffsets;
    std::vector<UIN> reorderedRows;
    std::vector<UIN> denseColOffsets;
    std::vector<UIN> denseCols;
    d2h(blockOffsets, blockOffsets_);
    d2h(reorderedRows, reorderedRows_);
    d2h(denseColOffsets, denseColOffsets_);
    d2h(denseCols, denseCols_);

    const UIN rowPanelId = calculateRowPanelIdByBlockValuesIndex(blockValueIndex);

    const UIN startIndexOfBlockValuesCurrentRowPanel = blockOffsets[rowPanelId] * BLOCK_SIZE;
    const UIN colBlockId = (blockValueIndex - startIndexOfBlockValuesCurrentRowPanel) / BLOCK_SIZE;

    const UIN localIndex = (blockValueIndex - startIndexOfBlockValuesCurrentRowPanel) % BLOCK_SIZE;
    const UIN localRowId = localIndex / BLOCK_COL_SIZE;
    const UIN localColId = localIndex % BLOCK_COL_SIZE;

    const UIN idxOfReorderedRows = rowPanelId * ROW_PANEL_SIZE + localRowId;
    const UIN row = idxOfReorderedRows < reorderedRows.size() ?
        reorderedRows[idxOfReorderedRows] : NULL_VALUE;

    const UIN idxOfReorderedCols = denseColOffsets[rowPanelId] +
        colBlockId * BLOCK_COL_SIZE + localColId;
    const UIN col = idxOfReorderedCols < denseColOffsets[rowPanelId + 1] ?
        denseCols[idxOfReorderedCols] : NULL_VALUE;

    return std::make_pair(row, col);
}

UIN ReBELL::calculateColBlockIdByBlockValueIndex(UIN blockValueIndex) const {
    std::vector<UIN> blockOffsets;
    d2h(blockOffsets, blockOffsets_);

    const UIN rowPanel = calculateRowPanelIdByBlockValuesIndex(blockValueIndex);
    const UIN startIndexOfBlockValueCurrentRowPanel = blockOffsets[rowPanel] * BLOCK_SIZE;

    return std::ceil((static_cast<float>(blockValueIndex - startIndexOfBlockValueCurrentRowPanel)) / BLOCK_SIZE);
}

float ReBELL::calculateAverageDensity() const {
    std::vector<UIN> blockOffsets;
    std::vector<UIN> blockValues;
    d2h(blockOffsets, blockOffsets_);
    d2h(blockValues, blockValues_);

    float totalDensity = 0.0f;
#pragma omp parallel for reduction(+ : totalDensity)
    for (int rowPanelId = 0; rowPanelId < numRowPanels_; ++rowPanelId) {
        const UIN startIndexOfBlockValuesCurrentRowPanel = blockOffsets[rowPanelId] * BLOCK_SIZE;
        const UIN endIndexOfBlockValuesCurrentRowPanel = blockOffsets[rowPanelId + 1] * BLOCK_SIZE;
        UIN numNonZero = 0;
        for (int idxOfBlockValues = startIndexOfBlockValuesCurrentRowPanel;
             idxOfBlockValues < endIndexOfBlockValuesCurrentRowPanel; ++idxOfBlockValues) {
            if (blockValues[idxOfBlockValues] != NULL_VALUE) {
                ++numNonZero;
            }
            if (idxOfBlockValues % BLOCK_SIZE == BLOCK_SIZE - 1) {
                totalDensity += static_cast<float>(numNonZero) / BLOCK_SIZE;
                numNonZero = 0;
            }
        }
    }

    return totalDensity / getNumDenseBlocks();
}

std::pair<float, float> ReBELL::calculateMaxMinDensity() const {
    std::vector<UIN> blockOffsets;
    std::vector<UIN> blockValues;
    d2h(blockOffsets, blockOffsets_);
    d2h(blockValues, blockValues_);

    float maxDensity = std::numeric_limits<float>::min();
    float minDensity = std::numeric_limits<float>::max();

#pragma omp parallel for reduction(max : maxDensity) reduction(min : minDensity)
    for (int rowPanelId = 0; rowPanelId < numRowPanels_; ++rowPanelId) {
        const UIN startIndexOfBlockValuesCurrentRowPanel = blockOffsets[rowPanelId] * BLOCK_SIZE;
        const UIN endIndexOfBlockValuesCurrentRowPanel = blockOffsets[rowPanelId + 1] * BLOCK_SIZE;
        UIN numNonZero = 0;
        for (int idxOfBlockValues = startIndexOfBlockValuesCurrentRowPanel;
             idxOfBlockValues < endIndexOfBlockValuesCurrentRowPanel; ++idxOfBlockValues) {
            if (blockValues[idxOfBlockValues] != NULL_VALUE) {
                ++numNonZero;
            }
            if (idxOfBlockValues % BLOCK_SIZE == BLOCK_SIZE - 1) {
                float curDensity = static_cast<float>(numNonZero) / BLOCK_SIZE;
                maxDensity = std::max(maxDensity, curDensity);
                minDensity = std::min(minDensity, curDensity);

                numNonZero = 0;
            }
        }
    }

    return std::make_pair(maxDensity, minDensity);
}

std::pair<float, UIN> ReBELL::calculateDensityMode() const {

    std::vector<UIN> blockOffsets;
    std::vector<UIN> blockValues;
    d2h(blockOffsets, blockOffsets_);
    d2h(blockValues, blockValues_);

    constexpr UIN numberOfDecimalPlacesToRetain = 3;
    const UIN divisor = static_cast<UIN>(std::pow(10, numberOfDecimalPlacesToRetain));

    std::unordered_map<UIN, UIN> densityToNumMap;
#pragma omp parallel for
    for (int rowPanelId = 0; rowPanelId < numRowPanels_; ++rowPanelId) {
        const UIN startIndexOfBlockValuesCurrentRowPanel = blockOffsets[rowPanelId] * BLOCK_SIZE;
        const UIN endIndexOfBlockValuesCurrentRowPanel = blockOffsets[rowPanelId + 1] * BLOCK_SIZE;
        UIN numNonZero = 0;
        for (int idxOfBlockValues = startIndexOfBlockValuesCurrentRowPanel;
             idxOfBlockValues < endIndexOfBlockValuesCurrentRowPanel; ++idxOfBlockValues) {
            if (blockValues[idxOfBlockValues] != NULL_VALUE) {
                ++numNonZero;
            }
            if (idxOfBlockValues % BLOCK_SIZE == BLOCK_SIZE - 1) {
                float curDensity = static_cast<float>(numNonZero) / BLOCK_SIZE;
                UIN density = static_cast<UIN>(curDensity * divisor);

#pragma omp critical
                {
                    if (densityToNumMap.find(density) == densityToNumMap.end()) {
                        densityToNumMap[density] = 1;
                    } else {
                        ++densityToNumMap[density];
                    }
                }

                numNonZero = 0;
            }
        }
    }

    UIN maxNum = std::numeric_limits<UIN>::min();
    float modeDensity = 0.0f;
    for (const auto &densityAndNum : densityToNumMap) {
        if (maxNum < densityAndNum.second) {
            maxNum = densityAndNum.second;
            modeDensity = static_cast<float>(densityAndNum.first) / divisor;
        }
    }

    return std::make_pair(modeDensity, maxNum);
}

bool check_rowReordering(const sparseMatrix::CSR<float> &matrix, const struct ReBELL &rebell) {

    std::vector<UIN> reorderedRows;
    d2h(reorderedRows, rebell.reorderedRows());

    std::unordered_map<UIN, UIN> rowToIndexOfReorderedRowsMap;
    for (int indexOfReorderedRows = 0; indexOfReorderedRows < reorderedRows.size();
         ++indexOfReorderedRows) {
        const UIN row = reorderedRows[indexOfReorderedRows];

        // Check if the row is duplicated
        if (rowToIndexOfReorderedRowsMap.find(row) != rowToIndexOfReorderedRowsMap.end()) {
            std::cerr << "Error! Row is duplicated! Duplicated row: " << row << std::endl;
            return false;
        }

        rowToIndexOfReorderedRowsMap[row] = indexOfReorderedRows;
    }

    for (int row = 0; row < matrix.row(); ++row) {
        const UIN numColIndices = matrix.rowOffsets()[row + 1] - matrix.rowOffsets()[row];

        if (numColIndices == 0) {

            // Check if empty rows are stored
            if (rowToIndexOfReorderedRowsMap.find(row) != rowToIndexOfReorderedRowsMap.end()) {
                std::cerr << "Error! Empty row is stored! Row: " << row << std::endl;
                return false;
            }
            continue;
        }

        // Check if there are any missing rows
        if (rowToIndexOfReorderedRowsMap.find(row) == rowToIndexOfReorderedRowsMap.end()) {
            std::cerr << "Error! Row is missing! Row: " << row << std::endl;
            return false;
        }
    }

    // TODO : Check if it is sorted correctly
    {

    }

    return true;
}

bool check_colReordering(const sparseMatrix::CSR<float> &matrix, const struct ReBELL &rebell) {
    bool isCorrect = true;

    std::vector<UIN> reorderedRows;
    std::vector<UIN> denseColOffsets;
    std::vector<UIN> denseCols;
    std::vector<UIN> sparseValueOffsets;
    std::vector<UIN> sparseRelativeRows;
    std::vector<UIN> sparseColIndices;

    d2h(reorderedRows, rebell.reorderedRows());
    d2h(denseColOffsets, rebell.denseColOffsets());
    d2h(denseCols, rebell.denseCols());
    d2h(sparseValueOffsets, rebell.sparseValueOffsets());
    d2h(sparseRelativeRows, rebell.sparseRelativeRows());
    d2h(sparseColIndices, rebell.sparseColIndices());

    for (int rowPanelId = 0; rowPanelId < rebell.numRowPanels(); ++rowPanelId) {

        const UIN startIdxOfReorderedRowIndicesCurrentRowPanel = rowPanelId * ROW_PANEL_SIZE;
        const UIN endIdxOfReorderedRowIndicesCurrentRowPanel =
            std::min(startIdxOfReorderedRowIndicesCurrentRowPanel + ROW_PANEL_SIZE,
                     static_cast<UIN>(reorderedRows.size()));

        // Count the number of non-zero elements for each column segment, and store the row and column indices in the current row panel
        std::unordered_map<UIN, UIN> colToNumOfNonZeroMap;
        std::unordered_set<UIN> rowIndicesCurrentRowPanelSet;
        for (int reorderedRowIndex = startIdxOfReorderedRowIndicesCurrentRowPanel;
             reorderedRowIndex < endIdxOfReorderedRowIndicesCurrentRowPanel;
             ++reorderedRowIndex) { // Loop through the rows in this row panel
            const UIN row = reorderedRows[reorderedRowIndex];
            rowIndicesCurrentRowPanelSet.insert(row);
            for (UIN idx = matrix.rowOffsets()[row]; idx < matrix.rowOffsets()[row + 1];
                 ++idx) { // Loop through the columns in this row
                const UIN col = matrix.colIndices()[idx];
                if (colToNumOfNonZeroMap.find(col) == colToNumOfNonZeroMap.end()) {
                    colToNumOfNonZeroMap[col] = 1;
                } else {
                    ++colToNumOfNonZeroMap[col];
                }
            }
        }

        // check dense column segment
        std::unordered_set<UIN> denseColIndicesRecordSet;
        for (int idxOfReorderedColIndices = denseColOffsets[rowPanelId];
             idxOfReorderedColIndices < denseColOffsets[rowPanelId + 1];
             ++idxOfReorderedColIndices) {
            const UIN col = denseCols[idxOfReorderedColIndices];

            // Check if column indexes are duplicated
            if (denseColIndicesRecordSet.find(col) != denseColIndicesRecordSet.end()) {
                std::cerr << "Error! Column indexes are duplicated\n" << std::endl;
                return false;
            }
            denseColIndicesRecordSet.insert(col);

            // Check if the column index in the row panel is correct
            if (colToNumOfNonZeroMap.find(col) == colToNumOfNonZeroMap.end()) {
                std::cerr << "Error! Column indexes in the row panel is incorrect!" << std::endl;
                return false;
            }

            // Check if the order of column indexes in the row panel is correct
            if (idxOfReorderedColIndices + 1 < denseColOffsets[rowPanelId + 1] &&
                colToNumOfNonZeroMap[col]
                    < colToNumOfNonZeroMap[denseCols[idxOfReorderedColIndices + 1]]) {
                std::cerr << "Error! The order of column indexes in the row panel is incorrect!" << std::endl;
                return false;
            }
        }

        // check sparse part data
        std::unordered_set<UIN> sparseColIndicesRecordSet;
        for (int idx = sparseValueOffsets[rowPanelId];
             idx < sparseValueOffsets[rowPanelId + 1];
             ++idx) {
            const UIN relativeRow = sparseRelativeRows[idx];
            const UIN row = reorderedRows[rowPanelId * ROW_PANEL_SIZE + relativeRow];
            const UIN col = sparseColIndices[idx];

            sparseColIndicesRecordSet.insert(col);

            // Check if the row is in the current row panel
            if (rowIndicesCurrentRowPanelSet.find(row) == rowIndicesCurrentRowPanelSet.end()) {
                fprintf(stderr,
                        "Error! Row not in current row panel! rowPanelId: %d, sparseValues[%d]\n",
                        rowPanelId, idx);
                return false;
            }

            // Check if the column index in the row panel is correct
            if (colToNumOfNonZeroMap.find(col) == colToNumOfNonZeroMap.end()) {
                fprintf(stderr,
                        "Error! Column not in current row panel! rowPanelId: %d, sparseValues[%d]\n",
                        rowPanelId, idx);
                return false;
            }

//            // Check if the column index is a dense column
//            if (colToNumOfNonZeroMap.find(col) != colToNumOfNonZeroMap.end()
//                && colToNumOfNonZeroMap.find(col)->second >= rebell.dense_column_segment_threshold()) {
//                fprintf(stderr,
//                        "Error! In sparse data, column index is not a sparse column! rowPanelId: %d, sparseData[%d]\n",
//                        rowPanelId,
//                        idx);
//                isCorrect = false;
//            }
        }

        // Check if the number of column indexes in the row panel is correct
        if ((denseColIndicesRecordSet.size() + sparseColIndicesRecordSet.size()) != colToNumOfNonZeroMap.size()) {
            fprintf(stderr,
                    "Error! The number of column indexes in the row panel is incorrect! Row panel : %d\n",
                    rowPanelId);
            return false;
        }

        // Check if the number of columns in the row panel is correct
        const UIN numColsCurrentRowPanel = denseColOffsets[rowPanelId + 1] -
            denseColOffsets[rowPanelId] + sparseColIndicesRecordSet.size();
        if (numColsCurrentRowPanel != colToNumOfNonZeroMap.size()) {
            fprintf(stderr, "Error! The number of columns in the row panel is incorrect! rowPanelId: %d\n", rowPanelId);
            return false;
        }
    }

    return true;
}

bool check_bell(const sparseMatrix::CSR<float> &matrix, const struct ReBELL &rebell) {

    std::vector<UIN> reorderedRows;

    // Dense block data
    std::vector<UIN> denseColOffsets;
    std::vector<UIN> denseCols;
    std::vector<UIN> blockOffsets;
    std::vector<UIN> blockValues;

    // Sparse block data
    std::vector<UIN> sparseValueOffsets;
    std::vector<UIN> sparseValues;
    std::vector<UIN> sparseRelativeRows;
    std::vector<UIN> sparseColIndices;

    // Copy data from device to host
    d2h(reorderedRows, rebell.reorderedRows());
    d2h(denseColOffsets, rebell.denseColOffsets());
    d2h(denseCols, rebell.denseCols());
    d2h(blockOffsets, rebell.blockOffsets());
    d2h(blockValues, rebell.blockValues());
    d2h(sparseValueOffsets, rebell.sparseValueOffsets());
    d2h(sparseValues, rebell.sparseValues());
    d2h(sparseRelativeRows, rebell.sparseRelativeRows());
    d2h(sparseColIndices, rebell.sparseColIndices());

    // Check if the blockRowOffsets is correct
    for (int idxOfBlockRowOffsets = 1; idxOfBlockRowOffsets < blockOffsets.size(); ++idxOfBlockRowOffsets) {
        const UIN rowPanelId = idxOfBlockRowOffsets - 1;
        const UIN numBlockCurrentRowPanel =
            blockOffsets[idxOfBlockRowOffsets] - blockOffsets[idxOfBlockRowOffsets - 1];
        const UIN numColsCurrentRowPanel =
            denseColOffsets[rowPanelId + 1] - denseColOffsets[rowPanelId];

        // Check if the number of blocks in the row panel is correct
        if (numBlockCurrentRowPanel !=
            static_cast<UIN>(std::ceil(static_cast<float>(numColsCurrentRowPanel) / BLOCK_COL_SIZE))) {
            fprintf(stderr, "Error! The number of blocks in the row panel is incorrect! rowPanelId: %d\n", rowPanelId);
            return false;
        }
    }

    std::unordered_set<UIN> blockValuesSet;
    for (UIN iter : blockValues) {
        // Check if the block value is duplicated
        if (blockValuesSet.find(iter) != blockValuesSet.end() && iter != NULL_VALUE) {
            fprintf(stderr, "Error! The block value is duplicated! val: %d\n", iter);
            return false;
        }
        blockValuesSet.insert(iter);
    }

    std::unordered_map<UIN, UIN> rowToIndexOfReorderedRowsMap;
    for (int indexOfReorderedRows = 0; indexOfReorderedRows < reorderedRows.size();
         ++indexOfReorderedRows) {
        const UIN row = reorderedRows[indexOfReorderedRows];
        rowToIndexOfReorderedRowsMap[row] = indexOfReorderedRows;
    }

    std::unordered_map<UIN, UIN> idxOfOriginalMatrixToSparsePartDataIndexMap;
    for (int idx = 0; idx < sparseValues.size(); ++idx) {
        const UIN originalMatrixIndex = sparseValues[idx];

        // Check if the original matrix index is duplicated in sparsePartData
        if (idxOfOriginalMatrixToSparsePartDataIndexMap.find(originalMatrixIndex)
            != idxOfOriginalMatrixToSparsePartDataIndexMap.end()) {
            fprintf(stderr,
                    "Error! The original matrix index is duplicated in sparseValues!"
                    " originalMatrixIndex: %u, sparsePartData[%d] and sparseValues[%d]\n",
                    originalMatrixIndex,
                    idx,
                    idxOfOriginalMatrixToSparsePartDataIndexMap.find(originalMatrixIndex)->second);
            return false;
        }
        idxOfOriginalMatrixToSparsePartDataIndexMap[originalMatrixIndex] = idx;
    }

    // Check based on the original matrix, check if the index of the original matrix is correctly stored in blockValue
    for (int row = 0; row < matrix.row(); ++row) {
        if (row + 1 < matrix.rowOffsets().size() && matrix.rowOffsets()[row + 1] - matrix.rowOffsets()[row] == 0) {
            continue;
        }

        // Check if the row exists in `reorderedRows`
        if (rowToIndexOfReorderedRowsMap.find(row) == rowToIndexOfReorderedRowsMap.end()) {
            fprintf(stderr, "Error! Row does not exist in \"reorderedRows\"! row = %d\n", row);
            return false;
        }
        const UIN indexOfReorderedRows = rowToIndexOfReorderedRowsMap[row];
        const UIN rowPanelId = indexOfReorderedRows / ROW_PANEL_SIZE;

        const UIN startIndexOfBlockValuesCurrentRowPanel = blockOffsets[rowPanelId] * BLOCK_SIZE;

        std::unordered_map<UIN, UIN> colToIndexOfReorderedColsMap_currentRow;
        for (int indexOfReorderedCols = denseColOffsets[rowPanelId];
             indexOfReorderedCols < denseColOffsets[rowPanelId + 1];
             ++indexOfReorderedCols) {
            const UIN col = denseCols[indexOfReorderedCols];
            colToIndexOfReorderedColsMap_currentRow[col] = indexOfReorderedCols;
        }

        for (int idxOfOriginalMatrix = matrix.rowOffsets()[row];
             idxOfOriginalMatrix < matrix.rowOffsets()[row + 1];
             ++idxOfOriginalMatrix) {
            const UIN col = matrix.colIndices()[idxOfOriginalMatrix];
            const UIN indexOfReorderedCols = colToIndexOfReorderedColsMap_currentRow[col];
            const UIN startIndexOfColsCurrentRowPanel = denseColOffsets[rowPanelId];
            const UIN colBlockId = (indexOfReorderedCols - startIndexOfColsCurrentRowPanel) / BLOCK_COL_SIZE;

            const UIN localRowId = indexOfReorderedRows % ROW_PANEL_SIZE;
            const UIN localColId = (indexOfReorderedCols - startIndexOfColsCurrentRowPanel) % BLOCK_COL_SIZE;

            const UIN idxOfBlockValues = startIndexOfBlockValuesCurrentRowPanel + colBlockId * BLOCK_SIZE +
                localRowId * BLOCK_COL_SIZE + localColId;

            // Check if the block value is correct
            if (idxOfBlockValues < blockValues.size()
                && blockValues[idxOfBlockValues] != idxOfOriginalMatrix
                && idxOfOriginalMatrixToSparsePartDataIndexMap.find(idxOfOriginalMatrix)
                    == idxOfOriginalMatrixToSparsePartDataIndexMap.end()) {
                fprintf(stderr,
                        "Error! The block value is incorrect!(Check based on the original matrix) row: %u, col: %u, rebell.blockValues()[%u]: %u, idxOfOriginalMatrix: %u, \n",
                        row,
                        col,
                        idxOfBlockValues,
                        blockValues[idxOfBlockValues],
                        idxOfOriginalMatrix);
                return false;
            }
        }
    }

    // Check based on the blockValues, check if the value of blockValue is stored correctly
    for (int idxOfBlockValues = 0; idxOfBlockValues < blockValues.size(); ++idxOfBlockValues) {

        std::pair<UIN, UIN> rowCol = rebell.calculateRowColByBlockValueIndex(idxOfBlockValues);
        const UIN row = rowCol.first;
        const UIN col = rowCol.second;

        if ((row > matrix.row() || col > matrix.col())) {

            // Check if the value is incorrect
            if (blockValues[idxOfBlockValues] != NULL_VALUE) {
                fprintf(stderr,
                        "Error! The value is incorrect!(Check based on the blockValues) idxOfBlockValues: %d\n",
                        idxOfBlockValues);
                return false;
            }
            continue;
        }

        for (int idxOfOriginalMatrix = matrix.rowOffsets()[row]; idxOfOriginalMatrix < matrix.rowOffsets()[row + 1];
             ++idxOfOriginalMatrix) {
            if (matrix.colIndices()[idxOfOriginalMatrix] == col) {

                // Check if the value is missing
                if (blockValues[idxOfBlockValues] == NULL_VALUE) {
                    fprintf(stderr,
                            "Error! Missing value!(Check based on the blockValues) row: %d, col: %d, idxOfBlockValues: %d, idxOfOriginalMatrix: %d\n",
                            row,
                            col,
                            idxOfBlockValues,
                            idxOfOriginalMatrix);

                    return false;
                }

                // Check if the block value is correct
                if (blockValues[idxOfBlockValues] != idxOfOriginalMatrix) {
                    fprintf(stderr,
                            "Error! The block value is incorrect!(Check based on the blockValues) row: %d, col: %d, idxOfBlockValues: %d, idxOfOriginalMatrix: %d\n",
                            row,
                            col,
                            idxOfBlockValues,
                            idxOfOriginalMatrix);

                    return false;
                }
            }

            // Check if a non-existent value appeared in blockValues
            if (idxOfOriginalMatrix == matrix.rowOffsets()[row + 1]
                && blockValues[idxOfBlockValues] != NULL_VALUE) {
                std::cerr << "Error! A non-existent value appeared in blockValues! idxOfBlockValues: %d" <<
                          idxOfBlockValues << std::endl;
                return false;
            }
        }
    }

    return true;
}

bool check_rebell(const sparseMatrix::CSR<float> &matrix, const ReBELL &rebell) {

    const auto [maxDensity, minDensity] = rebell.calculateMaxMinDensity();
    printf("rebell : numDenseBlock = %d, average density = %f%%, max average = %f%%, min average = %f%%\n",
           rebell.getNumDenseBlocks(),
           rebell.calculateAverageDensity() * 100,
           maxDensity * 100,
           minDensity * 100);
    printf("rebell: numSparseBlock = %d\n", rebell.getNumSparseBlocks());

    const auto [modeDensity, frequency] = rebell.calculateDensityMode();
    printf("rebell : mode density = %f%%, frequency = %d\n", modeDensity * 100, frequency);

    const auto [numTiles, averageDensity] = calculateNumTilesAndAverageDensityInOriginalMatrix(matrix);
    printf("Number of tiles before reorder: %d, average density : %f%%\n",
           numTiles, averageDensity * 100);

    bool isCorrect = true;
    if (!check_rowReordering(matrix, rebell)) {
        std::cerr << "Error! The row reordering is incorrect!" << std::endl;
        isCorrect = false;
    }

    if (!check_colReordering(matrix, rebell)) {
        std::cerr << "Error! The col reordering is incorrect!" << std::endl;
        isCorrect = false;
    }

    if (!check_bell(matrix, rebell)) {
        std::cerr << "Error! The bell is incorrect!" << std::endl;
        isCorrect = false;
    }

    return isCorrect;
}

std::pair<UIN, float> calculateNumTilesAndAverageDensityInOriginalMatrix(const sparseMatrix::CSR<float> &matrix) {
    UIN numTiles = 0;
    float totalDensity = 0.0f;

    const UIN numRowTiles = std::ceil(static_cast<float>(matrix.row()) / WMMA_M);
    const UIN numColTiles = std::ceil(static_cast<float>(matrix.col()) / WMMA_N);

#pragma omp parallel for reduction(+ : numTiles, totalDensity) schedule(dynamic)
    for (int rowTileId = 0; rowTileId < numRowTiles; ++rowTileId) {
        for (int colTileId = 0; colTileId < numColTiles; ++colTileId) {
            const UIN startRow = rowTileId * WMMA_M;
            const UIN endRow = std::min(static_cast<UIN>(startRow + WMMA_M), matrix.row());

            const UIN startCol = colTileId * WMMA_N;
            const UIN endCol = std::min(static_cast<UIN>(startCol + WMMA_N), matrix.col());

            UIN numNonZero = 0;
            for (int row = startRow; row < endRow; ++row) {
                for (int idx = matrix.rowOffsets()[row]; idx < matrix.rowOffsets()[row + 1]; ++idx) {
                    const UIN col = matrix.colIndices()[idx];

                    if (col >= startCol && col < endCol) {
                        ++numNonZero;
                    }
                }
            }

            if (numNonZero > 0) {
                ++numTiles;
                totalDensity += static_cast<float>(numNonZero) / (WMMA_M * WMMA_N);
            }
        }
    }

    float averageDensity = (numTiles > 0) ? totalDensity / numTiles : 0.0f;

    return std::make_pair(numTiles, averageDensity);
}