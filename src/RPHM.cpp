#include <cstdio>
#include <cmath>
#include <unordered_map>
#include <map>
#include <unordered_set>
#include <limits>
#include <numeric>
#include <array>
#include <omp.h>

#include "BSMR.hpp"
#include "CudaTimeCalculator.cuh"
#include "parallelAlgorithm.cuh"
#include "sddmmKernel.cuh"

BSMR::BSMR(const sparseMatrix::CSR<float> &matrix,
           const float similarityThreshold,
           const float blockDensityThreshold,
           const int numIterations) {
    // Row reordering
    float rowReordering_time = 0.0f;
    const UIN blockSize = calculateBlockSize(matrix);
    //    noReorderRow(matrix, reorderedRows_, rowReordering_time);
    for (int iter = 0; iter < numIterations; ++iter) {
        float oneIterationTime = 0.0f;
        reorderedRows_ = bsa_rowReordering_gpu(matrix,
                                               similarityThreshold,
                                               blockSize,
                                               numClusters_,
                                               oneIterationTime);
        rowReordering_time += oneIterationTime;
    }
    rowReordering_time /= numIterations;

    rowReorderingTime_ = rowReordering_time;
    // printf("rowReordering time : %f ms\n", rowReordering_time);

    numRowPanels_ = std::ceil(static_cast<float>(reorderedRows_.size()) / ROW_PANEL_SIZE);
    // printf("numRowPanels : %d\n", numRowPanels_);

    // Column reordering
    float colReordering_time = 0.0f;
    for (int iter = 0; iter < numIterations; ++iter) {
        float oneIterationTime = 0.0f;
        colReordering_cpu(matrix,
                          numRowPanels_,
                          reorderedRows_,
                          blockDensityThreshold,
                          denseCols_,
                          denseColOffsets_,
                          sparseCols_,
                          sparseColOffsets_,
                          sparseValueOffsets_,
                          oneIterationTime);
        colReordering_time += oneIterationTime;
    }
    colReordering_time /= numIterations;

    colReorderingTime_ = colReordering_time;
    // printf("colReordering time : %f ms\n", colReordering_time);

    reorderingTime_ = rowReordering_time + colReordering_time;
}

RPHM::RPHM(sparseMatrix::CSR<float> &matrix, const BSMR &bsmr) {
    std::vector<UIN> blockOffsets;
    std::vector<UIN> blockValues;

    std::vector<UIN> sparseValues;
    std::vector<UIN> sparseRelativeRows;
    std::vector<UIN> sparseColIndices;

    numRowPanels_ = bsmr.numRowPanels();

    // Calculate the maximum number of dense column blocks in a row panel
    maxNumDenseColBlocks_ = 0;
    //#pragma omp parallel for reduction(max : maxNumDenseColBlocks_)
    for (int rowPanelId = 0; rowPanelId < numRowPanels_; ++rowPanelId) {
        const UIN numBlocksCurrentRowPanel = std::ceil(
            static_cast<float>(bsmr.denseColOffsets()[rowPanelId + 1] - bsmr.denseColOffsets()[rowPanelId])
            / BLOCK_COL_SIZE);
        maxNumDenseColBlocks_ = std::max(maxNumDenseColBlocks_, numBlocksCurrentRowPanel);
    }

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();

    // initialize blockRowOffsets_
    std::vector<UIN> numBlockInEachRowPanel(numRowPanels_);
#pragma omp parallel for
    for (int rowPanelId = 0; rowPanelId < numRowPanels_; ++rowPanelId) {
        const UIN numColIndices = bsmr.denseColOffsets()[rowPanelId + 1] - bsmr.denseColOffsets()[rowPanelId];
        numBlockInEachRowPanel[rowPanelId] = std::ceil(static_cast<float>(numColIndices) / BLOCK_COL_SIZE);
    }

    blockOffsets.resize(numRowPanels_ + 1);
    blockOffsets[0] = 0;
    host::inclusive_scan(numBlockInEachRowPanel.data(),
                         numBlockInEachRowPanel.data() + numBlockInEachRowPanel.size(),
                         blockOffsets.data() + 1);

    sparseRelativeRows.resize(bsmr.sparseValueOffsets().back());
    sparseValues.resize(bsmr.sparseValueOffsets().back());
    sparseColIndices.resize(bsmr.sparseValueOffsets().back());

    // initialize blockValues_
    blockValues.resize(blockOffsets.back() * BLOCK_SIZE);
    host::fill_n(blockValues.data(), blockValues.size(), NULL_VALUE);
#pragma omp parallel for
    for (int indexOfReorderedRows = 0; indexOfReorderedRows < bsmr.reorderedRows().size(); ++indexOfReorderedRows) {
        const UIN row = bsmr.reorderedRows()[indexOfReorderedRows];

        std::unordered_map<UIN, UIN> colToIndexOfOriginalMatrixMap;
        for (int idxOfOriginalMatrix = matrix.rowOffsets()[row]; idxOfOriginalMatrix < matrix.rowOffsets()[row + 1];
             ++idxOfOriginalMatrix) {
            colToIndexOfOriginalMatrixMap[matrix.colIndices()[idxOfOriginalMatrix]] = idxOfOriginalMatrix;
        }

        const UIN rowPanelId = indexOfReorderedRows / ROW_PANEL_SIZE;
        const UIN localRowId = indexOfReorderedRows % ROW_PANEL_SIZE;
        const UIN startIndexOfBlockValuesCurrentRowPanel = blockOffsets[rowPanelId] * BLOCK_SIZE;

        // Iterate over the dense columns in the row panel
        for (int count = 0, indexOfReorderedCols = bsmr.denseColOffsets()[rowPanelId];
             indexOfReorderedCols < bsmr.denseColOffsets()[rowPanelId + 1];
             ++count, ++indexOfReorderedCols) {
            const UIN localColId = count % BLOCK_COL_SIZE;
            const UIN colBlockId = count / BLOCK_COL_SIZE;
            const UIN idxOfBlockValues = startIndexOfBlockValuesCurrentRowPanel + colBlockId * BLOCK_SIZE +
                                         localRowId * BLOCK_COL_SIZE + localColId;

            const UIN col = bsmr.denseCols()[indexOfReorderedCols];
            const auto findIter = colToIndexOfOriginalMatrixMap.find(col);
            if (findIter != colToIndexOfOriginalMatrixMap.end()) {
                blockValues[idxOfBlockValues] = findIter->second;
            }
        }
    }

    // Initialize sparse part data
#pragma omp parallel for
    for (int rowPanelId = 0; rowPanelId < numRowPanels_; ++rowPanelId) {
        std::unordered_map<UIN, std::vector<std::array<UIN, 2> > > colToRelativeRowAndOriginIndexMap;

        const UIN startIndex = rowPanelId * ROW_PANEL_SIZE;
        const UIN endIndex = std::min(startIndex + ROW_PANEL_SIZE, static_cast<UIN>(bsmr.reorderedRows().size()));

        for (int indexOfReorderedRows = startIndex; indexOfReorderedRows < endIndex; ++indexOfReorderedRows) {
            const UIN row = bsmr.reorderedRows()[indexOfReorderedRows];
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
        const UIN startSparsePartIndex = bsmr.sparseValueOffsets()[rowPanelId];
        // Iterate over the sparse columns in the row panel
        for (int indexOfReorderedCols = bsmr.sparseColOffsets()[rowPanelId];
             indexOfReorderedCols < bsmr.sparseColOffsets()[rowPanelId + 1]; ++indexOfReorderedCols) {
            const UIN col = bsmr.sparseCols()[indexOfReorderedCols];

            const auto findIter = colToRelativeRowAndOriginIndexMap.find(col);
            if (findIter != colToRelativeRowAndOriginIndexMap.end()) {
                for (const std::array<UIN, 2> &iter: findIter->second) {
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
    //#pragma omp parallel for reduction(max : maxNumSparseColBlocks_)
    for (int rowPanelId = 0; rowPanelId < numRowPanels_; ++rowPanelId) {
        const UIN numSparseData = bsmr.sparseValueOffsets()[rowPanelId + 1] - bsmr.sparseValueOffsets()[rowPanelId];
        const UIN numBlocksCurrentRowPanel = std::ceil(
            static_cast<float>(numSparseData) / sddmm_sparse_block_each_thread_block_counts_the_number_Of_data);
        maxNumSparseColBlocks_ = std::max(maxNumSparseColBlocks_, numBlocksCurrentRowPanel);
    }

    timeCalculator.endClock();
    float rphm_time = timeCalculator.getTime();
    // printf("rphm time : %f ms\n", rphm_time);

    // Copy data to device
    h2d(reorderedRows_, bsmr.reorderedRows());
    h2d(denseColOffsets_, bsmr.denseColOffsets());
    h2d(denseCols_, bsmr.denseCols());
    h2d(blockOffsets_, blockOffsets);
    h2d(blockValues_, blockValues);
    h2d(sparseValueOffsets_, bsmr.sparseValueOffsets());
    h2d(sparseValues_, sparseValues);
    h2d(sparseRelativeRows_, sparseRelativeRows);
    h2d(sparseColIndices_, sparseColIndices);
}

UIN RPHM::getNumSparseBlocks() const {
    return sparseValueOffsets().back_data()
           / static_cast<float>(sddmm_sparse_block_each_thread_block_counts_the_number_Of_data);
}

UIN RPHM::calculateRowPanelIdByBlockValuesIndex(UIN blockValueIndex) const {
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

UIN RPHM::calculateRowPanelIdByColIndex(UIN reorderedColIndex) const {
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

std::pair<UIN, UIN> RPHM::calculateLocalRowColByBlockValueIndex(UIN blockValueIndex) const {
    const UIN rowPanelId = calculateRowPanelIdByBlockValuesIndex(blockValueIndex);
    const UIN startIndexOfBlockValuesCurrentRowPanel = blockOffsets()[rowPanelId] * BLOCK_SIZE;
    const UIN localIndex = (blockValueIndex - startIndexOfBlockValuesCurrentRowPanel) % BLOCK_SIZE;
    const UIN localRowId = localIndex / BLOCK_COL_SIZE;
    const UIN localColId = localIndex % BLOCK_COL_SIZE;
    return std::make_pair(localRowId, localColId);
}

std::pair<UIN, UIN> RPHM::calculateRowColByBlockValueIndex(UIN blockValueIndex) const {
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
    const UIN row = idxOfReorderedRows < reorderedRows.size() ? reorderedRows[idxOfReorderedRows] : NULL_VALUE;

    const UIN idxOfReorderedCols = denseColOffsets[rowPanelId] +
                                   colBlockId * BLOCK_COL_SIZE + localColId;
    const UIN col = idxOfReorderedCols < denseColOffsets[rowPanelId + 1] ? denseCols[idxOfReorderedCols] : NULL_VALUE;

    return std::make_pair(row, col);
}

UIN RPHM::calculateColBlockIdByBlockValueIndex(UIN blockValueIndex) const {
    std::vector<UIN> blockOffsets;
    d2h(blockOffsets, blockOffsets_);

    const UIN rowPanel = calculateRowPanelIdByBlockValuesIndex(blockValueIndex);
    const UIN startIndexOfBlockValueCurrentRowPanel = blockOffsets[rowPanel] * BLOCK_SIZE;

    return std::ceil((static_cast<float>(blockValueIndex - startIndexOfBlockValueCurrentRowPanel)) / BLOCK_SIZE);
}

float RPHM::calculateDenseBlockAverageDensity() const {
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

std::pair<float, float> RPHM::calculateMaxMinDensity() const {
    std::vector<UIN> blockOffsets;
    std::vector<UIN> blockValues;
    d2h(blockOffsets, blockOffsets_);
    d2h(blockValues, blockValues_);

    float maxDensity = std::numeric_limits<float>::min();
    float minDensity = std::numeric_limits<float>::max();

    //#pragma omp parallel for reduction(max : maxDensity) reduction(min : minDensity)
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

std::pair<float, UIN> RPHM::calculateDensityMode() const {
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
    for (const auto &densityAndNum: densityToNumMap) {
        if (maxNum < densityAndNum.second) {
            maxNum = densityAndNum.second;
            modeDensity = static_cast<float>(densityAndNum.first) / divisor;
        }
    }

    return std::make_pair(modeDensity, maxNum);
}

bool check_rowReordering(const sparseMatrix::CSR<float> &matrix, const struct RPHM &rphm) {
    std::vector<UIN> reorderedRows;
    d2h(reorderedRows, rphm.reorderedRows());

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

bool check_colReordering(const sparseMatrix::CSR<float> &matrix, const BSMR &bsmr, const struct RPHM &rphm,
                         const float denseColSegmentThreshold) {
    std::vector<UIN> reorderedRows;
    std::vector<UIN> denseColOffsets;
    std::vector<UIN> denseCols;
    std::vector<UIN> sparseValueOffsets;
    std::vector<UIN> sparseRelativeRows;
    std::vector<UIN> sparseColIndices;

    d2h(reorderedRows, rphm.reorderedRows());
    d2h(denseColOffsets, rphm.denseColOffsets());
    d2h(denseCols, rphm.denseCols());
    d2h(sparseValueOffsets, rphm.sparseValueOffsets());
    d2h(sparseRelativeRows, rphm.sparseRelativeRows());
    d2h(sparseColIndices, rphm.sparseColIndices());

    for (int rowPanelId = 0; rowPanelId < rphm.numRowPanels(); ++rowPanelId) {
        const UIN startIdxOfReorderedRowIndicesCurrentRowPanel = rowPanelId * ROW_PANEL_SIZE;
        const UIN endIdxOfReorderedRowIndicesCurrentRowPanel =
                std::min(startIdxOfReorderedRowIndicesCurrentRowPanel + ROW_PANEL_SIZE,
                         static_cast<UIN>(reorderedRows.size()));

        // Count the number of non-zero elements for each column segment, and store the row and column indices in the current row panel
        std::unordered_map<UIN, UIN> colToNumOfNonZeroMap;
        std::unordered_set<UIN> rowIndicesCurrentRowPanelSet;
        for (int reorderedRowIndex = startIdxOfReorderedRowIndicesCurrentRowPanel;
             reorderedRowIndex < endIdxOfReorderedRowIndicesCurrentRowPanel;
             ++reorderedRowIndex) {
            // Loop through the rows in this row panel
            const UIN row = reorderedRows[reorderedRowIndex];
            rowIndicesCurrentRowPanelSet.insert(row);
            for (UIN idx = matrix.rowOffsets()[row]; idx < matrix.rowOffsets()[row + 1];
                 ++idx) {
                // Loop through the columns in this row
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

            // Check if the column index is not a dense column.
            if (colToNumOfNonZeroMap.find(col)->second < denseColSegmentThreshold) {
                fprintf(stderr,
                        "Error! Column index is not a dense column! rowPanelId: %d, col: %d\n",
                        rowPanelId,
                        col);
                return false;
            }
        }

        // check sparse column segment
        std::unordered_set<UIN> sparseColIndicesRecordSet;
        int numConsecutiveDenseCols = 0;
        for (int idxOfReorderedColIndices = bsmr.sparseColOffsets()[rowPanelId];
             idxOfReorderedColIndices < bsmr.sparseColOffsets()[rowPanelId + 1];
             ++idxOfReorderedColIndices) {
            const UIN col = bsmr.sparseCols()[idxOfReorderedColIndices];

            // Check if column indexes are duplicated
            if (sparseColIndicesRecordSet.find(col) != sparseColIndicesRecordSet.end()) {
                std::cerr << "Error! Column indexes are duplicated\n" << std::endl;
                return false;
            }
            sparseColIndicesRecordSet.insert(col);

            // Check if the column index in the row panel is correct
            if (colToNumOfNonZeroMap.find(col) == colToNumOfNonZeroMap.end()) {
                fprintf(stderr,
                        "Error! Column index not in current row panel! rowPanelId: %d, col: %d\n",
                        rowPanelId, col);
                return false;
            }

            // Check if there are any groups of dense blocks
            if (colToNumOfNonZeroMap.find(col)->second < denseColSegmentThreshold) {
                // If the column is not a dense column, reset the consecutive dense column count
                numConsecutiveDenseCols = 0;
            } else {
                // If the column is a dense column, increment the consecutive dense column count
                ++numConsecutiveDenseCols;
            }
            if (numConsecutiveDenseCols >= BLOCK_COL_SIZE) {
                // If there are more than one consecutive dense columns, it is an error
                fprintf(stderr,
                        "Error! 16 consecutive dense columns in sparse tile! rowPanelId: %d, idxOfReorderedColIndices: %d\n",
                        rowPanelId,
                        idxOfReorderedColIndices);
                return false;
            }
        }

        // Check if there are any dense columns in the sparse column segment
        for (const auto &denseCol: denseColIndicesRecordSet) {
            if (sparseColIndicesRecordSet.find(denseCol) != sparseColIndicesRecordSet.end()) {
                printf(" Error! Dense column index is also in sparse column segment! "
                       "rowPanelId: %d, denseCol: %d\n", rowPanelId, denseCol);
                return false;
            }
        }

        // check sparse part data
        for (int idx = sparseValueOffsets[rowPanelId];
             idx < sparseValueOffsets[rowPanelId + 1];
             ++idx) {
            const UIN relativeRow = sparseRelativeRows[idx];
            const UIN row = reorderedRows[rowPanelId * ROW_PANEL_SIZE + relativeRow];
            const UIN col = sparseColIndices[idx];

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

bool check_rphm(const sparseMatrix::CSR<float> &matrix, const struct RPHM &rphm) {
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
    d2h(reorderedRows, rphm.reorderedRows());
    d2h(denseColOffsets, rphm.denseColOffsets());
    d2h(denseCols, rphm.denseCols());
    d2h(blockOffsets, rphm.blockOffsets());
    d2h(blockValues, rphm.blockValues());
    d2h(sparseValueOffsets, rphm.sparseValueOffsets());
    d2h(sparseValues, rphm.sparseValues());
    d2h(sparseRelativeRows, rphm.sparseRelativeRows());
    d2h(sparseColIndices, rphm.sparseColIndices());

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
    for (UIN iter: blockValues) {
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
                        "Error! The block value is incorrect!(Check based on the original matrix) row: %u, col: %u, rphm.blockValues()[%u]: %u, idxOfOriginalMatrix: %u, \n",
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
        std::pair<UIN, UIN> rowCol = rphm.calculateRowColByBlockValueIndex(idxOfBlockValues);
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

void evaluationReordering(const sparseMatrix::CSR<float> &matrix, const BSMR &bsmr, Logger &logger) {
    int numDenseBlocks = 0;
    float totalDensity = 0.0f;
    int numDenseThreadBlocks = 0;
    int numSparseThreadBlocks = 0;

    // row panel loop
    for (int rowPanelId = 0; rowPanelId < bsmr.numRowPanels(); ++rowPanelId) {
        const int numDenseBlocksInCurrentRowPanel =
                std::ceil(
                    (bsmr.denseColOffsets()[rowPanelId + 1] - bsmr.denseColOffsets()[rowPanelId]) /
                    static_cast<float>(BLOCK_COL_SIZE));
        const int numSparseBlocksInCurrentRowPanel =
                std::ceil(
                    (bsmr.sparseColOffsets()[rowPanelId + 1] - bsmr.sparseColOffsets()[rowPanelId]) /
                    static_cast<float>(sddmm_sparse_block_each_thread_block_counts_the_number_Of_data));

        numDenseThreadBlocks += std::ceil(
            static_cast<float>(numDenseBlocksInCurrentRowPanel) / each_thread_block_counts_the_number_Of_dense_blocks);
        numSparseThreadBlocks += numSparseBlocksInCurrentRowPanel;

        // Maps each block ID to a set of column indices contained in that block
        std::vector<std::unordered_set<UIN> > blockToColumnSet(
            numDenseBlocksInCurrentRowPanel + numSparseBlocksInCurrentRowPanel);
        std::vector<UIN> nnzInEachBlock(
            numDenseBlocksInCurrentRowPanel + numSparseBlocksInCurrentRowPanel, 0);
        // dense column segment loop
        for (int indexOfReorderedCols = bsmr.denseColOffsets()[rowPanelId];
             indexOfReorderedCols < bsmr.denseColOffsets()[rowPanelId + 1];
             ++indexOfReorderedCols) {
            const UIN col = bsmr.denseCols()[indexOfReorderedCols];

            // Calculate the block id
            const UIN startIndexOfColsCurrentRowPanel = bsmr.denseColOffsets()[rowPanelId];
            const UIN colBlockId = (indexOfReorderedCols - startIndexOfColsCurrentRowPanel) / BLOCK_COL_SIZE;

            blockToColumnSet[colBlockId].insert(col);
        }

        const UIN startIndexOfReorderedRowsCurrentRowPanel = rowPanelId * ROW_PANEL_SIZE;
        const UIN endIndexOfReorderedRowsCurrentRowPanel =
                std::min(startIndexOfReorderedRowsCurrentRowPanel + ROW_PANEL_SIZE,
                         static_cast<UIN>(bsmr.reorderedRows().size()));
        // row index loop
        for (int indexOfReorderedRows = startIndexOfReorderedRowsCurrentRowPanel;
             indexOfReorderedRows < endIndexOfReorderedRowsCurrentRowPanel; ++indexOfReorderedRows) {
            const UIN row = bsmr.reorderedRows()[indexOfReorderedRows];

            // column index loop
            for (int idx = matrix.rowOffsets()[row]; idx < matrix.rowOffsets()[row + 1]; ++idx) {
                const UIN col = matrix.colIndices()[idx];

                // Calculate the block id
                for (int blockId = 0; blockId < blockToColumnSet.size(); ++blockId) {
                    if (blockToColumnSet[blockId].find(col) != blockToColumnSet[blockId].end()) {
                        ++nnzInEachBlock[blockId];
                    }
                }
            }
        }

        // Calculate the average density in the current row panel
        for (int blockId = 0; blockId < blockToColumnSet.size(); ++blockId) {
            const float blockSize = static_cast<float>(ROW_PANEL_SIZE * BLOCK_COL_SIZE);
            if (nnzInEachBlock[blockId] > 0) {
                const float density = static_cast<float>(nnzInEachBlock[blockId]) / blockSize;
                totalDensity += density;

                const float densityThreshold = logger.delta_;
                if (density >= densityThreshold) {
                    ++numDenseBlocks;
                }
            }
        }
    }

    const auto [numDenseBlocksInOriginalMatrix, averageDensityInOriginalMatrix] =
            calculateNumDenseBlocksAndAverageDensityInOriginalMatrix(logger.delta_, matrix);

    logger.numDenseBlock_ = numDenseBlocks;
    logger.averageDensity_ = totalDensity / numDenseBlocks;
    logger.numDenseThreadBlocks_ = numDenseThreadBlocks;
    logger.numSparseThreadBlocks_ = numSparseThreadBlocks;
    logger.originalNumDenseBlock_ = numDenseBlocksInOriginalMatrix;
    logger.originalAverageDensity_ = averageDensityInOriginalMatrix;
}

bool check_rphm(const sparseMatrix::CSR<float> &matrix, const BSMR &bsmr, const RPHM &rphm,
                const float denseColSegmentThreshold) {
    bool isCorrect = true;
    if (!check_rowReordering(matrix, rphm)) {
        std::cerr << "Error! The row reordering is incorrect!" << std::endl;
        isCorrect = false;
    }

    if (!check_colReordering(matrix, bsmr, rphm, denseColSegmentThreshold)) {
        std::cerr << "Error! The col reordering is incorrect!" << std::endl;
        isCorrect = false;
    }

    if (!check_rphm(matrix, rphm)) {
        std::cerr << "Error! The rphm is incorrect!" << std::endl;
        isCorrect = false;
    }

    return isCorrect;
}

std::pair<UIN, float> calculateNumDenseBlocksAndAverageDensityInOriginalMatrix(
    const float densityThreshold, const sparseMatrix::CSR<float> &matrix) {
    const int numRowPanels = std::ceil(static_cast<float>(matrix.row()) / ROW_PANEL_SIZE);
    const int numColBlocks = std::ceil(static_cast<float>(matrix.col()) / BLOCK_COL_SIZE);
    UIN numDenseBlocks = 0;
    float totalDensity = 0.0f;
    for (int rowPanel = 0; rowPanel < numRowPanels; ++rowPanel) {
        for (int colBlock = 0; colBlock < numColBlocks; ++colBlock) {
            const UIN startRow = rowPanel * ROW_PANEL_SIZE;
            const UIN endRow = std::min(static_cast<UIN>(startRow + ROW_PANEL_SIZE), matrix.row());

            const UIN startCol = colBlock * BLOCK_COL_SIZE;
            const UIN endCol = std::min(static_cast<UIN>(startCol + BLOCK_COL_SIZE), matrix.col());

            UIN numNonZero = 0;
            for (int row = startRow; row < endRow; ++row) {
                for (int idx = matrix.rowOffsets()[row]; idx < matrix.rowOffsets()[row + 1]; ++idx) {
                    const UIN col = matrix.colIndices()[idx];

                    if (col >= startCol && col < endCol) {
                        ++numNonZero;
                    }
                }
            }
            const float blockSize = static_cast<float>((endRow - startRow) * (endCol - startCol));
            if (numNonZero > 0) {
                const float density = static_cast<float>(numNonZero) / (blockSize);
                if (density >= densityThreshold) {
                    totalDensity += density;
                    ++numDenseBlocks;
                }
            }
        }
    }

    float averageDensity = (numDenseBlocks > 0) ? totalDensity / numDenseBlocks : 0.0f;

    return std::make_pair(numDenseBlocks, averageDensity);
}
