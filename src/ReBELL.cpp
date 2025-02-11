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
    for (int rowPanelIdx = 0; rowPanelIdx < numRowPanel; ++rowPanelIdx) {
        const UIN numColIndices = reorderedColsOffset_[rowPanelIdx + 1] - reorderedColsOffset_[rowPanelIdx];
        numBlockInEachRowPanel[rowPanelIdx] = std::ceil(static_cast<float>(numColIndices) / block_col_size);
    }
    blockRowOffsets_.resize(numRowPanel + 1);
    blockRowOffsets_[0] = 0;
    host::inclusive_scan(numBlockInEachRowPanel.data(),
                         numBlockInEachRowPanel.data() + numBlockInEachRowPanel.size(),
                         blockRowOffsets_.data() + 1);

    // initialize blockValues_
    blockValues_.resize(blockRowOffsets_.back() * block_size);
    host::fill_n(blockValues_.data(), blockValues_.size(), MAX_UIN);
#pragma omp parallel for
    for (int indexOfReorderedRow = 0; indexOfReorderedRow < reorderedRows_.size(); ++indexOfReorderedRow) {
        const UIN row = reorderedRows_[indexOfReorderedRow];

        std::unordered_map<UIN, UIN> colToIndexMap;
        for (int idxOfOriginalMatrix = matrix.rowOffsets_[row]; idxOfOriginalMatrix < matrix.rowOffsets_[row + 1];
             ++idxOfOriginalMatrix) {
            colToIndexMap[matrix.colIndices_[idxOfOriginalMatrix]] = idxOfOriginalMatrix;
        }

        const UIN rowPanelIdx = indexOfReorderedRow / row_panel_size;
        const UIN localRowIdx = indexOfReorderedRow % row_panel_size;
        const UIN startIndexOfBlockValuesCurrentRowPanel = blockRowOffsets_[rowPanelIdx] * block_size;
        // Iterate over the columns in the row panel
        for (int iter = 0, indexOfReorderedCol = reorderedColsOffset_[rowPanelIdx];
             indexOfReorderedCol < reorderedColsOffset_[rowPanelIdx + 1];
             ++iter, ++indexOfReorderedCol) {
            const UIN localColIdx = iter % block_col_size;
            const UIN colBlockId = iter / block_col_size;
            const UIN idxOfBlockValues = startIndexOfBlockValuesCurrentRowPanel + colBlockId * block_size +
                localRowIdx * block_col_size + localColIdx;

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

    std::unordered_map<UIN, UIN> rowToReorderedRowIndexMap;
    for (int reorderedRowIndex = 0; reorderedRowIndex < rebell.reorderedRows().size();
         ++reorderedRowIndex) {
        const UIN row = rebell.reorderedRows()[reorderedRowIndex];

        // Check if the row is duplicated
        if (rowToReorderedRowIndexMap.find(row) != rowToReorderedRowIndexMap.end()) {
            std::cerr << "Error! Row is duplicated! Duplicated row: " << row << std::endl;
            isCorrect = false;
        }

        rowToReorderedRowIndexMap[row] = reorderedRowIndex;
    }

    for (int row = 0; row < matrix.row_; ++row) {
        const UIN numColIndices = matrix.rowOffsets_[row + 1] - matrix.rowOffsets_[row];

        if (numColIndices == 0) {

            // Check if empty rows are stored
            if (rowToReorderedRowIndexMap.find(row) != rowToReorderedRowIndexMap.end()) {
                std::cerr << "Error! Empty row is stored! Row: " << row << std::endl;
                isCorrect = false;
            }
            continue;
        }

        // Check if there are any missing rows
        if (rowToReorderedRowIndexMap.find(row) == rowToReorderedRowIndexMap.end()) {
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

    for (int rowPanelIdx = 0; rowPanelIdx < rebell.numRowPanels(); ++rowPanelIdx) {

        const UIN startIdxOfReorderedRowIndicesCurrentRowPanel = rowPanelIdx * row_panel_size;
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
        for (int idxOfReorderedColIndices = rebell.reorderedColsOffset()[rowPanelIdx];
             idxOfReorderedColIndices < rebell.reorderedColsOffset()[rowPanelIdx + 1];
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
            if (idxOfReorderedColIndices + 1 < rebell.reorderedColsOffset()[rowPanelIdx + 1] &&
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

    std::unordered_map<UIN, UIN> rowToReorderedRowIndexMap;
    for (int reorderedRowIndex = 0; reorderedRowIndex < rebell.reorderedRows().size();
         ++reorderedRowIndex) {
        const UIN row = rebell.reorderedRows()[reorderedRowIndex];
        rowToReorderedRowIndexMap[row] = reorderedRowIndex;
    }

    for (int row = 0; row < matrix.row_; ++row) {
        UIN reorderedRowIndex = 0;
        while (reorderedRowIndex < rebell.reorderedRows().size() &&
            rebell.reorderedRows()[reorderedRowIndex] != row) {
            ++reorderedRowIndex;
        }

        for (int idxOfOriginalMatrix = matrix.rowOffsets_[row];
             idxOfOriginalMatrix < matrix.rowOffsets_[row + 1];
             ++idxOfOriginalMatrix) {
            const UIN col = matrix.colIndices_[idxOfOriginalMatrix];
        }
    }

//    for (int rowPanelIdx = 0; rowPanelIdx < rebell.numRowPanels_; ++rowPanelIdx) {
//        for (int reorderedRowIndex = rebell.reorderedRowIndices_[rowPanelIdx * row_panel_size];)
//    }

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