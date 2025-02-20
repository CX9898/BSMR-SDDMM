#include <cstdio>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <numeric>
#include <omp.h>

#include "ReBELL.hpp"
#include "CudaTimeCalculator.cuh"
#include "parallelAlgorithm.cuh"

ReBELL::ReBELL(const sparseMatrix::CSR<float> &matrix, float &time) {

    // Row reordering
    float rowReordering_time;
    {
//        bsa_rowReordering_cpu(matrix, reorderedRows_, rowReordering_time);
//        std::vector<int> reorderedRows =
//            bsa_rowReordering_cpu(matrix, row_similarity_threshold_alpha, BLOCK_SIZE, rowReordering_time);
        int clu_cnt;
        reorderedRows_ =
            bsa_rowReordering_gpu(matrix, row_similarity_threshold_alpha, BLOCK_COL_SIZE, rowReordering_time, clu_cnt);
    }
    printf("rowReordering time : %f ms\n", rowReordering_time);

    numRowPanels_ = std::ceil(static_cast<float>(reorderedRows_.size()) / ROW_PANEL_SIZE);

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();
    // column reordering
    colReordering(matrix, numRowPanels_, reorderedRows(), reorderedColOffsets_, reorderedCols_);
    timeCalculator.endClock();
    float colReordering_time = timeCalculator.getTime();
    printf("colReordering time : %f ms\n", colReordering_time);

    // Calculate the maximum number of column blocks in a row panel
    maxNumColBlocks_ = 0;
    for (int rowPanelId = 0; rowPanelId < numRowPanels_; ++rowPanelId) {
        const UIN numBlocksCurrentRowPanel = std::ceil(
            static_cast<float>(reorderedColOffsets_[rowPanelId + 1] - reorderedColOffsets()[rowPanelId])
                / BLOCK_COL_SIZE);
        maxNumColBlocks_ = std::max(maxNumColBlocks_, numBlocksCurrentRowPanel);
    }

    timeCalculator.startClock();

    // initialize blockRowOffsets_
    const UIN numRowPanel = std::ceil(static_cast<float>(reorderedRows_.size()) / ROW_PANEL_SIZE);
    std::vector<UIN> numBlockInEachRowPanel(numRowPanel);
#pragma omp parallel for
    for (int rowPanelId = 0; rowPanelId < numRowPanel; ++rowPanelId) {
        const UIN numColIndices = reorderedColOffsets_[rowPanelId + 1] - reorderedColOffsets_[rowPanelId];
        numBlockInEachRowPanel[rowPanelId] = std::ceil(static_cast<float>(numColIndices) / BLOCK_COL_SIZE);
    }
    blockRowOffsets_.resize(numRowPanel + 1);
    blockRowOffsets_[0] = 0;
    host::inclusive_scan(numBlockInEachRowPanel.data(),
                         numBlockInEachRowPanel.data() + numBlockInEachRowPanel.size(),
                         blockRowOffsets_.data() + 1);

    // initialize blockValues_
    blockValues_.resize(blockRowOffsets_.back() * BLOCK_SIZE);
    host::fill_n(blockValues_.data(), blockValues_.size(), NULL_VALUE);
#pragma omp parallel for
    for (int indexOfReorderedRows = 0; indexOfReorderedRows < reorderedRows_.size(); ++indexOfReorderedRows) {
        const UIN row = reorderedRows_[indexOfReorderedRows];

        std::unordered_map<UIN, UIN> colToIndexOfOriginalMatrixMap;
        for (int idxOfOriginalMatrix = matrix.rowOffsets()[row]; idxOfOriginalMatrix < matrix.rowOffsets()[row + 1];
             ++idxOfOriginalMatrix) {
            colToIndexOfOriginalMatrixMap[matrix.colIndices()[idxOfOriginalMatrix]] = idxOfOriginalMatrix;
        }

        const UIN rowPanelId = indexOfReorderedRows / ROW_PANEL_SIZE;
        const UIN localRowId = indexOfReorderedRows % ROW_PANEL_SIZE;
        const UIN startIndexOfBlockValuesCurrentRowPanel = blockRowOffsets_[rowPanelId] * BLOCK_SIZE;
        // Iterate over the columns in the row panel
        for (int iter = 0, indexOfReorderedCols = reorderedColOffsets_[rowPanelId];
             indexOfReorderedCols < reorderedColOffsets_[rowPanelId + 1];
             ++iter, ++indexOfReorderedCols) {
            const UIN localColId = iter % BLOCK_COL_SIZE;
            const UIN colBlockId = iter / BLOCK_COL_SIZE;
            const UIN idxOfBlockValues = startIndexOfBlockValuesCurrentRowPanel + colBlockId * BLOCK_SIZE +
                localRowId * BLOCK_COL_SIZE + localColId;

            const UIN col = reorderedCols_[indexOfReorderedCols];
            const auto findIter = colToIndexOfOriginalMatrixMap.find(col);
            if (findIter != colToIndexOfOriginalMatrixMap.end()) {
                blockValues_[idxOfBlockValues] = findIter->second;
            }
        }
    }

    timeCalculator.endClock();
    float bell_time = timeCalculator.getTime();
    printf("bell time : %f ms\n", bell_time);

    time = rowReordering_time + colReordering_time + bell_time;
}

UIN ReBELL::calculateRowPanelIdByBlockValuesIndex(UIN blockValueIndex) const {
    UIN rowPanelId = 0;
    while (rowPanelId + 1 < blockRowOffsets().size()) {
        if (blockValueIndex < blockRowOffsets()[rowPanelId + 1] * BLOCK_SIZE) {
            break;
        }
        ++rowPanelId;
    }
    return rowPanelId;
}

UIN ReBELL::calculateRowPanelIdByColIndex(UIN reorderedColIndex) const {
    UIN rowPanelId = 0;
    while (rowPanelId + 1 < reorderedColOffsets().size()) {
        if (reorderedColIndex < reorderedColOffsets()[rowPanelId + 1]) {
            break;
        }
        ++rowPanelId;
    }
    return rowPanelId;
}

std::pair<UIN, UIN> ReBELL::calculateLocalRowColByBlockValueIndex(UIN blockValueIndex) const {
    const UIN rowPanelId = calculateRowPanelIdByBlockValuesIndex(blockValueIndex);
    const UIN startIndexOfBlockValuesCurrentRowPanel = blockRowOffsets()[rowPanelId] * BLOCK_SIZE;
    const UIN localIndex = (blockValueIndex - startIndexOfBlockValuesCurrentRowPanel) % BLOCK_SIZE;
    const UIN localRowId = localIndex / BLOCK_COL_SIZE;
    const UIN localColId = localIndex % BLOCK_COL_SIZE;
    return std::make_pair(localRowId, localColId);
}

std::pair<UIN, UIN> ReBELL::calculateRowColByBlockValueIndex(UIN blockValueIndex) const {
    const UIN rowPanelId = calculateRowPanelIdByBlockValuesIndex(blockValueIndex);

    const UIN startIndexOfBlockValuesCurrentRowPanel = blockRowOffsets()[rowPanelId] * BLOCK_SIZE;
    const UIN colBlockId = (blockValueIndex - startIndexOfBlockValuesCurrentRowPanel) / BLOCK_SIZE;

    const UIN localIndex = (blockValueIndex - startIndexOfBlockValuesCurrentRowPanel) % BLOCK_SIZE;
    const UIN localRowId = localIndex / BLOCK_COL_SIZE;
    const UIN localColId = localIndex % BLOCK_COL_SIZE;

    const UIN idxOfReorderedRows = rowPanelId * ROW_PANEL_SIZE + localRowId;
    const UIN row = idxOfReorderedRows < reorderedRows().size() ?
        reorderedRows()[idxOfReorderedRows] : NULL_VALUE;

    const UIN idxOfReorderedCols = reorderedColOffsets()[rowPanelId] +
        colBlockId * BLOCK_COL_SIZE + localColId;
    const UIN col = idxOfReorderedCols < reorderedColOffsets()[rowPanelId + 1] ?
        reorderedCols()[idxOfReorderedCols] : NULL_VALUE;

    return std::make_pair(row, col);
}

UIN ReBELL::calculateColBlockIdByBlockValueIndex(UIN blockValueIndex) const {
    const UIN rowPanel = calculateRowPanelIdByBlockValuesIndex(blockValueIndex);
    const UIN startIndexOfBlockValueCurrentRowPanel = blockRowOffsets()[rowPanel] * BLOCK_SIZE;

    return std::ceil((static_cast<float>(blockValueIndex - startIndexOfBlockValueCurrentRowPanel)) / BLOCK_SIZE);
}

float ReBELL::calculateAverageDensity() {
    float totalDensity = 0.0f;
#pragma omp parallel for reduction(+ : totalDensity)
    for (int rowPanelId = 0; rowPanelId < numRowPanels_; ++rowPanelId) {
        const UIN startIndexOfBlockValuesCurrentRowPanel = blockRowOffsets_[rowPanelId] * BLOCK_SIZE;
        const UIN endIndexOfBlockValuesCurrentRowPanel = blockRowOffsets_[rowPanelId + 1] * BLOCK_SIZE;
        UIN numNonZero = 0;
        for (int idxOfBlockValues = startIndexOfBlockValuesCurrentRowPanel;
             idxOfBlockValues < endIndexOfBlockValuesCurrentRowPanel; ++idxOfBlockValues) {
            if (blockValues_[idxOfBlockValues] != NULL_VALUE) {
                ++numNonZero;
            }
            if (idxOfBlockValues % BLOCK_SIZE == BLOCK_SIZE - 1) {
                totalDensity += static_cast<float>(numNonZero) / BLOCK_SIZE;
                numNonZero = 0;
            }
        }
    }

    return totalDensity / getNumBlocks();
}

std::pair<float, float> ReBELL::calculateMaxMinDensity() {
    float maxDensity = std::numeric_limits<float>::min();
    float minDensity = std::numeric_limits<float>::max();

#pragma omp parallel for reduction(max : maxDensity) reduction(min : minDensity)
    for (int rowPanelId = 0; rowPanelId < numRowPanels_; ++rowPanelId) {
        const UIN startIndexOfBlockValuesCurrentRowPanel = blockRowOffsets_[rowPanelId] * BLOCK_SIZE;
        const UIN endIndexOfBlockValuesCurrentRowPanel = blockRowOffsets_[rowPanelId + 1] * BLOCK_SIZE;
        UIN numNonZero = 0;
        for (int idxOfBlockValues = startIndexOfBlockValuesCurrentRowPanel;
             idxOfBlockValues < endIndexOfBlockValuesCurrentRowPanel; ++idxOfBlockValues) {
            if (blockValues_[idxOfBlockValues] != NULL_VALUE) {
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

bool check_rowReordering(const sparseMatrix::CSR<float> &matrix, const struct ReBELL &rebell) {
    bool isCorrect = true;

    std::unordered_map<UIN, UIN> rowToIndexOfReorderedRowsMap;
    for (int indexOfReorderedRows = 0; indexOfReorderedRows < rebell.reorderedRows().size();
         ++indexOfReorderedRows) {
        const UIN row = rebell.reorderedRows()[indexOfReorderedRows];

        // Check if the row is duplicated
        if (rowToIndexOfReorderedRowsMap.find(row) != rowToIndexOfReorderedRowsMap.end()) {
            std::cerr << "Error! Row is duplicated! Duplicated row: " << row << std::endl;
            isCorrect = false;
        }

        rowToIndexOfReorderedRowsMap[row] = indexOfReorderedRows;
    }

    for (int row = 0; row < matrix.row(); ++row) {
        const UIN numColIndices = matrix.rowOffsets()[row + 1] - matrix.rowOffsets()[row];

        if (numColIndices == 0) {

            // Check if empty rows are stored
            if (rowToIndexOfReorderedRowsMap.find(row) != rowToIndexOfReorderedRowsMap.end()) {
                std::cerr << "Error! Empty row is stored! Row: " << row << std::endl;
                isCorrect = false;
            }
            continue;
        }

        // Check if there are any missing rows
        if (rowToIndexOfReorderedRowsMap.find(row) == rowToIndexOfReorderedRowsMap.end()) {
            std::cerr << "Error! Row is missing! Row: " << row << std::endl;
            isCorrect = false;
        }
    }

    // TODO : Check if it is sorted correctly
    {

    }

    return isCorrect;
}

bool check_colReordering(const sparseMatrix::CSR<float> &matrix, const struct ReBELL &rebell) {
    bool isCorrect = true;

    for (int rowPanelId = 0; rowPanelId < rebell.numRowPanels(); ++rowPanelId) {

        const UIN startIdxOfReorderedRowIndicesCurrentRowPanel = rowPanelId * ROW_PANEL_SIZE;
        const UIN endIdxOfReorderedRowIndicesCurrentRowPanel =
            std::min(startIdxOfReorderedRowIndicesCurrentRowPanel + ROW_PANEL_SIZE,
                     static_cast<UIN>(rebell.reorderedRows().size()));

        // Count the number of non-zero elements for each column segment
        std::unordered_map<UIN, UIN> colToNumOfNonZeroMap;
        for (int reorderedRowIndex = startIdxOfReorderedRowIndicesCurrentRowPanel;
             reorderedRowIndex < endIdxOfReorderedRowIndicesCurrentRowPanel;
             ++reorderedRowIndex) { // Loop through the rows in this row panel
            const UIN row = rebell.reorderedRows()[reorderedRowIndex];
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

        std::unordered_set<UIN> colIndicesRecordSet;
        for (int idxOfReorderedColIndices = rebell.reorderedColOffsets()[rowPanelId];
             idxOfReorderedColIndices < rebell.reorderedColOffsets()[rowPanelId + 1];
             ++idxOfReorderedColIndices) {
            const UIN col = rebell.reorderedCols()[idxOfReorderedColIndices];

            // 1) Check if column indexes are duplicated
            if (colIndicesRecordSet.find(col) != colIndicesRecordSet.end()) {
                std::cerr << "Error! Column indexes are duplicated\n" << std::endl;
                isCorrect = false;
            }
            colIndicesRecordSet.insert(col);

            // 2) Check if the column index in the row panel is correct
            if (colToNumOfNonZeroMap.find(col) == colToNumOfNonZeroMap.end()) {
                std::cerr << "Error! Column indexes in the row panel is incorrect!" << std::endl;
                isCorrect = false;
            }

            // 3) Check if the order of column indexes in the row panel is correct
            if (idxOfReorderedColIndices + 1 < rebell.reorderedColOffsets()[rowPanelId + 1] &&
                colToNumOfNonZeroMap[col]
                    < colToNumOfNonZeroMap[rebell.reorderedCols()[idxOfReorderedColIndices + 1]]) {
                std::cerr << "Error! The order of column indexes in the row panel is incorrect!" << std::endl;
                isCorrect = false;
            }
        }

        // 4) Check if the number of column indexes in the row panel is correct
        if (colIndicesRecordSet.size() != colToNumOfNonZeroMap.size()) {
            std::cerr << "Error! The number of column indexes in the row panel is incorrect!" << std::endl;
            isCorrect = false;
        }

        // 5) Check if the number of columns in the row panel is correct
        const UIN numColsCurrentRowPanel = rebell.reorderedColOffsets()[rowPanelId + 1] -
            rebell.reorderedColOffsets()[rowPanelId];
        if (numColsCurrentRowPanel != colToNumOfNonZeroMap.size()) {
            fprintf(stderr, "Error! The number of columns in the row panel is incorrect! rowPanelId: %d\n", rowPanelId);
        }
    }

    return isCorrect;
}

bool check_bell(const sparseMatrix::CSR<float> &matrix, const struct ReBELL &rebell) {
    bool isCorrect = true;

    // Check if the blockRowOffsets is correct
    for (int idxOfBlockRowOffsets = 1; idxOfBlockRowOffsets < rebell.blockRowOffsets().size(); ++idxOfBlockRowOffsets) {
        const UIN rowPanelId = idxOfBlockRowOffsets - 1;
        const UIN numBlockCurrentRowPanel =
            rebell.blockRowOffsets()[idxOfBlockRowOffsets] - rebell.blockRowOffsets()[idxOfBlockRowOffsets - 1];
        const UIN numColsCurrentRowPanel =
            rebell.reorderedColOffsets()[rowPanelId + 1] - rebell.reorderedColOffsets()[rowPanelId];

        // Check if the number of blocks in the row panel is correct
        if (numBlockCurrentRowPanel !=
            static_cast<UIN>(std::ceil(static_cast<float>(numColsCurrentRowPanel) / BLOCK_COL_SIZE))) {
            fprintf(stderr, "Error! The number of blocks in the row panel is incorrect! rowPanelId: %d\n", rowPanelId);
            isCorrect = false;
        }
    }

    std::unordered_set<UIN> blockValuesSet;
    for (UIN iter : rebell.blockValues()) {
        // Check if the block value is duplicated
        if (blockValuesSet.find(iter) != blockValuesSet.end() && iter != NULL_VALUE) {
            fprintf(stderr, "Error! The block value is duplicated! val: %d\n", iter);
            isCorrect = false;
        }
        blockValuesSet.insert(iter);
    }

    std::unordered_map<UIN, UIN> rowToIndexOfReorderedRowsMap;
    for (int indexOfReorderedRows = 0; indexOfReorderedRows < rebell.reorderedRows().size();
         ++indexOfReorderedRows) {
        const UIN row = rebell.reorderedRows()[indexOfReorderedRows];
        rowToIndexOfReorderedRowsMap[row] = indexOfReorderedRows;
    }

    // Check based on the original matrix, check if the index of the original matrix is correctly stored in blockValue
    for (int row = 0; row < matrix.row(); ++row) {
        if (row + 1 < matrix.rowOffsets().size() && matrix.rowOffsets()[row + 1] - matrix.rowOffsets()[row] == 0) {
            continue;
        }

        // Check if the row exists in `reorderedRows`
        if (rowToIndexOfReorderedRowsMap.find(row) == rowToIndexOfReorderedRowsMap.end()) {
            fprintf(stderr, "Error! Row does not exist in \"reorderedRows\"! row = %d\n", row);
            isCorrect = false;
        }
        const UIN indexOfReorderedRows = rowToIndexOfReorderedRowsMap[row];
        const UIN rowPanelId = indexOfReorderedRows / ROW_PANEL_SIZE;

        const UIN startIndexOfBlockValuesCurrentRowPanel = rebell.blockRowOffsets()[rowPanelId] * BLOCK_SIZE;

        std::unordered_map<UIN, UIN> colToIndexOfReorderedColsMap_currentRow;
        for (int indexOfReorderedCols = rebell.reorderedColOffsets()[rowPanelId];
             indexOfReorderedCols < rebell.reorderedColOffsets()[rowPanelId + 1];
             ++indexOfReorderedCols) {
            const UIN col = rebell.reorderedCols()[indexOfReorderedCols];
            colToIndexOfReorderedColsMap_currentRow[col] = indexOfReorderedCols;
        }

        for (int idxOfOriginalMatrix = matrix.rowOffsets()[row];
             idxOfOriginalMatrix < matrix.rowOffsets()[row + 1];
             ++idxOfOriginalMatrix) {
            const UIN col = matrix.colIndices()[idxOfOriginalMatrix];
            const UIN indexOfReorderedCols = colToIndexOfReorderedColsMap_currentRow[col];
            const UIN startIndexOfColsCurrentRowPanel = rebell.reorderedColOffsets()[rowPanelId];
            const UIN colBlockId = (indexOfReorderedCols - startIndexOfColsCurrentRowPanel) / BLOCK_COL_SIZE;

            const UIN localRowId = indexOfReorderedRows % ROW_PANEL_SIZE;
            const UIN localColId = (indexOfReorderedCols - startIndexOfColsCurrentRowPanel) % BLOCK_COL_SIZE;

            const UIN idxOfBlockValues = startIndexOfBlockValuesCurrentRowPanel + colBlockId * BLOCK_SIZE +
                localRowId * BLOCK_COL_SIZE + localColId;

            // Check if the block value is correct
            if (rebell.blockValues()[idxOfBlockValues] != idxOfOriginalMatrix) {
                fprintf(stderr,
                        "Error! The block value is incorrect!(Check based on the original matrix) row: %u, col: %u, rebell.blockValues()[%u]: %u, idxOfOriginalMatrix: %u, \n",
                        row,
                        col,
                        idxOfBlockValues,
                        rebell.blockValues()[idxOfBlockValues],
                        idxOfOriginalMatrix);
                isCorrect = false;
            }
        }
    }

    // Check based on the blockValues, check if the value of blockValue is stored correctly
    for (int idxOfBlockValues = 0; idxOfBlockValues < rebell.blockValues().size(); ++idxOfBlockValues) {

        std::pair<UIN, UIN> rowCol = rebell.calculateRowColByBlockValueIndex(idxOfBlockValues);
        const UIN row = rowCol.first;
        const UIN col = rowCol.second;

        if ((row > matrix.row() || col > matrix.col())) {

            // Check if the value is incorrect
            if (rebell.blockValues()[idxOfBlockValues] != NULL_VALUE) {
                fprintf(stderr,
                        "Error! The value is incorrect!(Check based on the blockValues) idxOfBlockValues: %d\n",
                        idxOfBlockValues);
                isCorrect = false;
            }
            continue;
        }

        for (int idxOfOriginalMatrix = matrix.rowOffsets()[row]; idxOfOriginalMatrix < matrix.rowOffsets()[row + 1];
             ++idxOfOriginalMatrix) {
            if (matrix.colIndices()[idxOfOriginalMatrix] == col) {

                // Check if the value is missing
                if (rebell.blockValues()[idxOfBlockValues] == NULL_VALUE) {
                    fprintf(stderr,
                            "Error! Missing value!(Check based on the blockValues) row: %d, col: %d, idxOfBlockValues: %d, idxOfOriginalMatrix: %d\n",
                            row,
                            col,
                            idxOfBlockValues,
                            idxOfOriginalMatrix);

                    isCorrect = false;
                    break;
                }

                // Check if the block value is correct
                if (rebell.blockValues()[idxOfBlockValues] != idxOfOriginalMatrix) {
                    fprintf(stderr,
                            "Error! The block value is incorrect!(Check based on the blockValues) row: %d, col: %d, idxOfBlockValues: %d, idxOfOriginalMatrix: %d\n",
                            row,
                            col,
                            idxOfBlockValues,
                            idxOfOriginalMatrix);

                    isCorrect = false;
                    break;
                }
            }

            // Check if a non-existent value appeared in blockValues
            if (idxOfOriginalMatrix == matrix.rowOffsets()[row + 1]
                && rebell.blockValues()[idxOfBlockValues] != NULL_VALUE) {
                std::cerr << "Error! A non-existent value appeared in blockValues! idxOfBlockValues: %d" <<
                          idxOfBlockValues << std::endl;
                isCorrect = false;
            }
        }
    }

    return isCorrect;
}

bool check_rebell(const sparseMatrix::CSR<float> &matrix, const struct ReBELL &rebell) {
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
    printf("Total tiles: %d\n", numRowTiles * numColTiles);

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