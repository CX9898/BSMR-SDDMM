#include <cstdio>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <numeric>

#include "ReBELL.hpp"
#include "CudaTimeCalculator.cuh"
#include "parallelAlgorithm.cuh"

ReBELL::ReBELL(const sparseMatrix::CSR<float> &matrix) {

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
    const UIN numRowPanel = std::ceil(static_cast<float>(reorderedRows_.size()) / ROW_PANEL_SIZE);
    std::vector<UIN> numBlockInEachRowPanel(numRowPanel);
#pragma omp parallel for
    for (int rowPanelId = 0; rowPanelId < numRowPanel; ++rowPanelId) {
        const UIN numColIndices = reorderedColsOffset_[rowPanelId + 1] - reorderedColsOffset_[rowPanelId];
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
    for (int indexOfReorderedRow = 0; indexOfReorderedRow < reorderedRows_.size(); ++indexOfReorderedRow) {
        const UIN row = reorderedRows_[indexOfReorderedRow];

        std::unordered_map<UIN, UIN> colToIndexMap;
        for (int idxOfOriginalMatrix = matrix.rowOffsets_[row]; idxOfOriginalMatrix < matrix.rowOffsets_[row + 1];
             ++idxOfOriginalMatrix) {
            colToIndexMap[matrix.colIndices_[idxOfOriginalMatrix]] = idxOfOriginalMatrix;
        }

        const UIN rowPanelId = indexOfReorderedRow / ROW_PANEL_SIZE;
        const UIN localRowId = indexOfReorderedRow % ROW_PANEL_SIZE;
        const UIN startIndexOfBlockValuesCurrentRowPanel = blockRowOffsets_[rowPanelId] * BLOCK_SIZE;
        // Iterate over the columns in the row panel
        for (int iter = 0, indexOfReorderedCol = reorderedColsOffset_[rowPanelId];
             indexOfReorderedCol < reorderedColsOffset_[rowPanelId + 1];
             ++iter, ++indexOfReorderedCol) {
            const UIN localColId = iter % BLOCK_COL_SIZE;
            const UIN colBlockId = iter / BLOCK_COL_SIZE;
            const UIN idxOfBlockValues = startIndexOfBlockValuesCurrentRowPanel + colBlockId * BLOCK_SIZE +
                localRowId * BLOCK_COL_SIZE + localColId;

            const UIN col = reorderedCols_[indexOfReorderedCol];
            const auto findIter = colToIndexMap.find(col);
            if (findIter != colToIndexMap.end()) {
                blockValues_[idxOfBlockValues] = findIter->second;
            }
        }
    }
}

UIN ReBELL::calculateRowPanelId(UIN blockValueIndex) const {
    UIN rowPanelId = 0;
    while (rowPanelId + 1 < blockRowOffsets().size()) {
        if (blockValueIndex < blockRowOffsets()[rowPanelId + 1] * BLOCK_SIZE) {
            break;
        }
        ++rowPanelId;
    }
    return rowPanelId;
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

    for (int row = 0; row < matrix.row_; ++row) {
        const UIN numColIndices = matrix.rowOffsets_[row + 1] - matrix.rowOffsets_[row];

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
        const UIN endIdxOfReorderedRowIndicesCurrentRowPanel = std::min(
            startIdxOfReorderedRowIndicesCurrentRowPanel + ROW_PANEL_SIZE,
            static_cast<UIN>(rebell.reorderedRows().size()));

        // Count the number of non-zero elements for each column segment
        std::unordered_map<UIN, UIN> colAndNumOfNonZeroMap;
        for (int reorderedRowIndex = startIdxOfReorderedRowIndicesCurrentRowPanel;
             reorderedRowIndex < endIdxOfReorderedRowIndicesCurrentRowPanel;
             ++reorderedRowIndex) { // Loop through the rows in this row panel
            const UIN row = rebell.reorderedRows()[reorderedRowIndex];
            for (UIN idx = matrix.rowOffsets_[row]; idx < matrix.rowOffsets_[row + 1];
                 ++idx) { // Loop through the columns in this row
                const UIN col = matrix.colIndices_[idx];
                if (colAndNumOfNonZeroMap.find(col) == colAndNumOfNonZeroMap.end()) {
                    colAndNumOfNonZeroMap[col] = 1;
                } else {
                    ++colAndNumOfNonZeroMap[col];
                }
            }
        }

        std::unordered_set<UIN> colIndicesRecordSet;
        for (int idxOfReorderedColIndices = rebell.reorderedColsOffset()[rowPanelId];
             idxOfReorderedColIndices < rebell.reorderedColsOffset()[rowPanelId + 1];
             ++idxOfReorderedColIndices) {
            const UIN col = rebell.reorderedCols()[idxOfReorderedColIndices];

            // 1) Check if column indexes are duplicated
            if (colIndicesRecordSet.find(col) != colIndicesRecordSet.end()) {
                std::cerr << "Error! Column indexes are duplicated\n" << std::endl;
                isCorrect = false;
            }
            colIndicesRecordSet.insert(col);

            // 2) Check if the column index in the row panel is correct
            if (colAndNumOfNonZeroMap.find(col) == colAndNumOfNonZeroMap.end()) {
                std::cerr << "Error! Column indexes in the row panel is incorrect!" << std::endl;
                isCorrect = false;
            }

            // 3) Check if the order of column indexes in the row panel is correct
            if (idxOfReorderedColIndices + 1 < rebell.reorderedColsOffset()[rowPanelId + 1] &&
                colAndNumOfNonZeroMap[col]
                    < colAndNumOfNonZeroMap[rebell.reorderedCols()[idxOfReorderedColIndices + 1]]) {
                std::cerr << "Error! The order of column indexes in the row panel is incorrect!" << std::endl;
                isCorrect = false;
            }
        }

        // 4) Check if the number of column indexes in the row panel is correct
        if (colIndicesRecordSet.size() != colAndNumOfNonZeroMap.size()) {
            std::cerr << "Error! The number of column indexes in the row panel is incorrect!" << std::endl;
            isCorrect = false;
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
            rebell.reorderedColsOffset()[rowPanelId + 1] - rebell.reorderedColsOffset()[rowPanelId];

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
    for (int row = 0; row < matrix.row_; ++row) {
        if (row + 1 < matrix.row_ && matrix.rowOffsets_[row + 1] - matrix.rowOffsets_[row] == 0) {
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
        for (int indexOfReorderedCols = rebell.reorderedColsOffset()[rowPanelId];
             indexOfReorderedCols < rebell.reorderedColsOffset()[rowPanelId + 1];
             ++indexOfReorderedCols) {
            const UIN col = rebell.reorderedCols()[indexOfReorderedCols];
            colToIndexOfReorderedColsMap_currentRow[col] = indexOfReorderedCols;
        }

        for (int idxOfOriginalMatrix = matrix.rowOffsets_[row];
             idxOfOriginalMatrix < matrix.rowOffsets_[row + 1];
             ++idxOfOriginalMatrix) {
            const UIN col = matrix.colIndices_[idxOfOriginalMatrix];
            const UIN indexOfReorderedCols = colToIndexOfReorderedColsMap_currentRow[col];
            const UIN colBlockId = indexOfReorderedCols / BLOCK_COL_SIZE;

            const UIN localRowId = indexOfReorderedRows % ROW_PANEL_SIZE;
            const UIN localColId = indexOfReorderedCols % BLOCK_COL_SIZE;

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

        const UIN rowPanelId = rebell.calculateRowPanelId(idxOfBlockValues);

        const UIN startIndexOfBlockValuesCurrentRowPanel = rebell.blockRowOffsets()[rowPanelId] * BLOCK_SIZE;
        const UIN colBlockId = (idxOfBlockValues - startIndexOfBlockValuesCurrentRowPanel) / BLOCK_SIZE;

        const UIN localRowId = (idxOfBlockValues - startIndexOfBlockValuesCurrentRowPanel) %
            BLOCK_SIZE / BLOCK_COL_SIZE;
        const UIN localColId = (idxOfBlockValues - startIndexOfBlockValuesCurrentRowPanel) % BLOCK_COL_SIZE;

        const UIN idxOfReorderedRows = rowPanelId * ROW_PANEL_SIZE + localRowId;
        const UIN row = idxOfReorderedRows < rebell.reorderedRows().size() ?
            rebell.reorderedRows()[idxOfReorderedRows] : NULL_VALUE;

        const UIN idxOfReorderedCols = rebell.reorderedColsOffset()[rowPanelId] +
            colBlockId * BLOCK_COL_SIZE + localColId;
        const UIN col = idxOfReorderedCols < rebell.reorderedCols().size() ?
            rebell.reorderedCols()[idxOfReorderedCols] : NULL_VALUE;

        if ((row > matrix.row_ || col > matrix.col_)) {

            // Check if the value is incorrect
            if (rebell.blockValues()[idxOfBlockValues] != NULL_VALUE) {
                fprintf(stderr,
                        "Error! The value is incorrect!(Check based on the blockValues) idxOfBlockValues: %d\n",
                        idxOfBlockValues);
                isCorrect = false;
            }
            continue;
        }

        for (int idxOfOriginalMatrix = matrix.rowOffsets_[row]; idxOfOriginalMatrix < matrix.rowOffsets_[row + 1];
             ++idxOfOriginalMatrix) {
            if (matrix.colIndices_[idxOfOriginalMatrix] == col) {

                // Check if the value is missing
                if (rebell.blockValues()[idxOfBlockValues] == NULL_VALUE) {
                    fprintf(stderr,
                            "Error! Missing value!(Check based on the blockValues) row: %d, col: %d, idxOfBlockValues: %d, idxOfOriginalMatrix: %d, localCalId = %d\n",
                            row,
                            col,
                            idxOfBlockValues,
                            idxOfOriginalMatrix,
                            localColId);

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
            if (idxOfOriginalMatrix == matrix.rowOffsets_[row + 1]
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