#include <cstdio>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <numeric>

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
    const UIN numRowPanel = std::ceil(static_cast<float>(reorderedRows_.size()) / row_panel_size);
    std::vector<UIN> numBlockInEachRowPanel(numRowPanel);
#pragma omp parallel for
    for (int rowPanelId = 0; rowPanelId < numRowPanel; ++rowPanelId) {
        const UIN numColIndices = reorderedColsOffset_[rowPanelId + 1] - reorderedColsOffset_[rowPanelId];
        numBlockInEachRowPanel[rowPanelId] = std::ceil(static_cast<float>(numColIndices) / block_col_size);
    }
    blockRowOffsets_.resize(numRowPanel + 1);
    blockRowOffsets_[0] = 0;
    host::inclusive_scan(numBlockInEachRowPanel.data(),
                         numBlockInEachRowPanel.data() + numBlockInEachRowPanel.size(),
                         blockRowOffsets_.data() + 1);

    // initialize blockValues_
    blockValues_.resize(blockRowOffsets_.back() * block_size);
    host::fill_n(blockValues_.data(), blockValues_.size(), NULL_VALUE);
#pragma omp parallel for
    for (int indexOfReorderedRow = 0; indexOfReorderedRow < reorderedRows_.size(); ++indexOfReorderedRow) {
        const UIN row = reorderedRows_[indexOfReorderedRow];

        std::unordered_map<UIN, UIN> colToIndexMap;
        for (int idxOfOriginalMatrix = matrix.rowOffsets_[row]; idxOfOriginalMatrix < matrix.rowOffsets_[row + 1];
             ++idxOfOriginalMatrix) {
            colToIndexMap[matrix.colIndices_[idxOfOriginalMatrix]] = idxOfOriginalMatrix;
        }

        const UIN rowPanelId = indexOfReorderedRow / row_panel_size;
        const UIN localRowId = indexOfReorderedRow % row_panel_size;
        const UIN startIndexOfBlockValuesCurrentRowPanel = blockRowOffsets_[rowPanelId] * block_size;
        // Iterate over the columns in the row panel
        for (int iter = 0, indexOfReorderedCol = reorderedColsOffset_[rowPanelId];
             indexOfReorderedCol < reorderedColsOffset_[rowPanelId + 1];
             ++iter, ++indexOfReorderedCol) {
            const UIN localColId = iter % block_col_size;
            const UIN colBlockId = iter / block_col_size;
            const UIN idxOfBlockValues = startIndexOfBlockValuesCurrentRowPanel + colBlockId * block_size +
                localRowId * block_col_size + localColId;

            const UIN col = reorderedCols_[indexOfReorderedCol];
            const auto findIter = colToIndexMap.find(col);
            if (findIter != colToIndexMap.end()) {
                blockValues_[idxOfBlockValues] = findIter->second;
            }
        }
    }
}

bool check_rowReordering(const sparseDataType::CSR<float> &matrix, const struct ReBELL &rebell) {
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

bool check_colReordering(const sparseDataType::CSR<float> &matrix, const struct ReBELL &rebell) {
    bool isCorrect = true;

    for (int rowPanelId = 0; rowPanelId < rebell.numRowPanels(); ++rowPanelId) {

        const UIN startIdxOfReorderedRowIndicesCurrentRowPanel = rowPanelId * row_panel_size;
        const UIN endIdxOfReorderedRowIndicesCurrentRowPanel = std::min(
            startIdxOfReorderedRowIndicesCurrentRowPanel + row_panel_size,
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

bool check_bell(const sparseDataType::CSR<float> &matrix, const struct ReBELL &rebell) {
    bool isCorrect = true;

    std::unordered_set<UIN> blockValuesSet;
    for(UIN iter : blockValuesSet){
        // Check if the block value is duplicated
        if(blockValuesSet.find(iter) != blockValuesSet.end() && iter != NULL_VALUE){
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

    // Check based on the original matrix
    for (int row = 0; row < matrix.row_; ++row) {
        if (row + 1 < matrix.row_ && matrix.rowOffsets_[row + 1] - matrix.rowOffsets_[row] == 0) {
            continue;
        }

        const UIN indexOfReorderedRows = rowToIndexOfReorderedRowsMap[row];
        const UIN rowPanelId = indexOfReorderedRows / row_panel_size;

        const UIN startIndexOfBlockValuesCurrentRowPanel = rebell.blockRowOffsets()[rowPanelId] * block_size;

        std::unordered_map<UIN, UIN> colToReorderedColsMap;
        for (int indexOfReorderedCols = rebell.reorderedColsOffset()[rowPanelId];
             indexOfReorderedCols < rebell.reorderedColsOffset()[rowPanelId + 1];
             ++indexOfReorderedCols) {
            const UIN col = rebell.reorderedCols()[indexOfReorderedCols];
            colToReorderedColsMap[col] = indexOfReorderedCols;
        }

        for (int idxOfOriginalMatrix = matrix.rowOffsets_[row];
             idxOfOriginalMatrix < matrix.rowOffsets_[row + 1];
             ++idxOfOriginalMatrix) {
            const UIN col = matrix.colIndices_[idxOfOriginalMatrix];
            const UIN indexOfReorderedCols = colToReorderedColsMap[col];
            const UIN colBlockId = indexOfReorderedCols / block_col_size;

            const UIN localRowId = indexOfReorderedRows % row_panel_size;
            const UIN localColId = indexOfReorderedCols % block_col_size;

            const UIN idxOfBlockValues = startIndexOfBlockValuesCurrentRowPanel + colBlockId * block_size +
                localRowId * block_col_size + localColId;

            // Check if the block value is correct
            if (rebell.blockValues()[idxOfBlockValues] != idxOfOriginalMatrix) {
                fprintf(stderr,
                        "Error! The block value is incorrect!(Check based on the original matrix) row: %d, col: %d, idxOfBlockValues: %d, idxOfOriginalMatrix: %d\n",
                        row,
                        col,
                        idxOfBlockValues,
                        idxOfOriginalMatrix);
                isCorrect = false;
            }
        }
    }

    // Check based on the blockValues
    for (int idxOfBlockValues = 0; idxOfBlockValues < rebell.blockValues().size(); ++idxOfBlockValues) {

        UIN rowPanelId = 0;
        while (rowPanelId + 1 < rebell.numRowPanels()) {
            if (idxOfBlockValues < rebell.blockRowOffsets()[rowPanelId + 1] * block_size) {
                break;
            }
            ++rowPanelId;
        }

        const UIN startIndexOfBlockValuesCurrentRowPanel = rebell.blockRowOffsets()[rowPanelId] * block_size;
        const UIN colBlockId = (idxOfBlockValues - startIndexOfBlockValuesCurrentRowPanel) / block_size;

        const UIN
            localRowId = (idxOfBlockValues - startIndexOfBlockValuesCurrentRowPanel) % block_size / block_col_size;
        const UIN localColId = (idxOfBlockValues - startIndexOfBlockValuesCurrentRowPanel) % block_col_size;

        const UIN row = rebell.reorderedRows()[rowPanelId * row_panel_size + localRowId];
        const UIN col =
            rebell.reorderedCols()[rebell.reorderedColsOffset()[rowPanelId] + colBlockId * block_col_size + localColId];

        for (int idxOfOriginalMatrix = matrix.rowOffsets_[row]; idxOfOriginalMatrix < matrix.rowOffsets_[row + 1];
             ++idxOfOriginalMatrix) {
            if (matrix.colIndices_[idxOfOriginalMatrix] == col) {

                // Check if the value is missing
                if(rebell.blockValues()[idxOfBlockValues] == NULL_VALUE){
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

bool check_rebell(const sparseDataType::CSR<float> &matrix, const struct ReBELL &rebell) {
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