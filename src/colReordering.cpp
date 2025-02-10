#include <iostream>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <unordered_set>

#include "ReBELL.hpp"
#include "TensorCoreConfig.cuh"
#include "parallelAlgorithm.cuh"

bool check_colReordering(const sparseDataType::CSR<float> &matrix, const struct ReBELL &rebell) {

    for (int rowPanelIdx = 0; rowPanelIdx < rebell.numRowPanels_; ++rowPanelIdx) {

        const UIN startIdxOfReorderedRowIndicesInThisRowPanel = rowPanelIdx * row_panel_size;
        const UIN endIdxOfReorderedRowIndicesInThisRowPanel = std::min(
            startIdxOfReorderedRowIndicesInThisRowPanel + row_panel_size,
            static_cast<UIN>(rebell.reorderedRowIndices_.size()));

        // Count the number of non-zero elements for each column segment
        std::unordered_map<UIN, UIN> colAndNumOfNonZeroMap;
        for (int idxOfReorderedRowIndices = startIdxOfReorderedRowIndicesInThisRowPanel;
             idxOfReorderedRowIndices < endIdxOfReorderedRowIndicesInThisRowPanel;
             ++idxOfReorderedRowIndices) { // Loop through the rows in this row panel
            const UIN row = rebell.reorderedRowIndices_[idxOfReorderedRowIndices];
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
        for (int idxOfReorderedColIndices = rebell.reorderedColIndicesOffset_[rowPanelIdx];
             idxOfReorderedColIndices < rebell.reorderedColIndicesOffset_[rowPanelIdx + 1];
             ++idxOfReorderedColIndices) {
            const UIN col = rebell.reorderedColIndices_[idxOfReorderedColIndices];

            // 1) Check if column indexes are duplicated
            if (colIndicesRecordSet.find(col) != colIndicesRecordSet.end()) {
                std::cerr << "Error! Column indexes are duplicated\n" << std::endl;
                return false;
            }
            colIndicesRecordSet.insert(col);

            // 2) Check if the column index in the row panel is correct
            if (colAndNumOfNonZeroMap.find(col) == colAndNumOfNonZeroMap.end()) {
                std::cerr << "Error! Column indexes in the row panel is incorrect!" << std::endl;
                return false;
            }

            // 3) Check if the order of column indexes in the row panel is correct
            if (idxOfReorderedColIndices + 1 < rebell.reorderedColIndicesOffset_[rowPanelIdx + 1] &&
                colAndNumOfNonZeroMap[col]
                    < colAndNumOfNonZeroMap[rebell.reorderedColIndices_[idxOfReorderedColIndices + 1]]) {
                std::cerr << "Error! The order of column indexes in the row panel is incorrect!" << std::endl;
                return false;
            }
        }

        // 4) Check if the number of column indexes in the row panel is correct
        if (colIndicesRecordSet.size() != colAndNumOfNonZeroMap.size()) {
            std::cerr << "Error! The number of column indexes in the row panel is incorrect!" << std::endl;
            return false;
        }
    }

    return true;
}

void ReBELL::colReordering(const sparseDataType::CSR<float> &matrix) {
    numRowPanels_ = std::ceil(static_cast<float>(reorderedRowIndices_.size()) / row_panel_size);
    std::vector<UIN> numOfNonZeroColSegmentInEachRowPanel(numRowPanels_, 0);
    std::vector<std::vector<UIN>>
        colIndicesInEachRowPanel_sparse(numRowPanels_, std::vector<UIN>(matrix.col_)); // Containing empty columns
#pragma omp parallel for
    for (int rowPanelIdx = 0; rowPanelIdx < numRowPanels_; ++rowPanelIdx) {
        const UIN startIdxOfReorderedRowIndicesInThisRowPanel = rowPanelIdx * row_panel_size;
        const UIN endIdxOfReorderedRowIndicesInThisRowPanel = std::min(
            startIdxOfReorderedRowIndicesInThisRowPanel + row_panel_size,
            static_cast<UIN>(reorderedRowIndices_.size()));

        // Count the number of non-zero elements for each column segment
        std::vector<UIN> numOfNonZeroInEachColSegment(matrix.col_, 0);
        for (int idxOfReorderedRowIndices = startIdxOfReorderedRowIndicesInThisRowPanel;
             idxOfReorderedRowIndices < endIdxOfReorderedRowIndicesInThisRowPanel;
             ++idxOfReorderedRowIndices) { // Loop through the rows in this row panel
            const UIN row = reorderedRowIndices_[idxOfReorderedRowIndices];
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

    reorderedColIndicesOffset_.resize(numRowPanels_ + 1);
    reorderedColIndicesOffset_[0] = 0;
    host::inclusive_scan(numOfNonZeroColSegmentInEachRowPanel.data(),
                         numOfNonZeroColSegmentInEachRowPanel.data() + numOfNonZeroColSegmentInEachRowPanel.size(),
                         reorderedColIndicesOffset_.data() + 1);

    reorderedColIndices_.resize(reorderedColIndicesOffset_[numRowPanels_]);
#pragma omp parallel for
    for (int rowPanelIdx = 0; rowPanelIdx < numRowPanels_; ++rowPanelIdx) {
        std::copy(colIndicesInEachRowPanel_sparse[rowPanelIdx].begin(),
                  colIndicesInEachRowPanel_sparse[rowPanelIdx].begin()
                      + numOfNonZeroColSegmentInEachRowPanel[rowPanelIdx],
                  reorderedColIndices_.begin() + reorderedColIndicesOffset_[rowPanelIdx]);
    }
}