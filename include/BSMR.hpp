#pragma once

#include <Logger.hpp>

#include "devVector.cuh"
#include "Matrix.hpp"

constexpr UIN ROW_PANEL_SIZE = WMMA_M;
constexpr UIN BLOCK_COL_SIZE = WMMA_N;
constexpr UIN BLOCK_SIZE = ROW_PANEL_SIZE * BLOCK_COL_SIZE;

/**
 * @className: BSMR
 * @classInterpretation: Reorder the rows and columns of a sparse matrix and divide it into dense tiled and sparse tiled.
 * @MemberVariables:
 * `reorderedRows_`: Store the reordered row indexes.
 * `denseCols_`: Store the reordered dense column indexes for each row panel in order.
 * `denseColOffsets_`: Offset array of reordered dense column array in each row panel.
 *
 **/
class BSMR{
public:
    BSMR() = default;

    BSMR(const float similarityThreshold,
         const float blockDensityThreshold,
         const sparseMatrix::CSR<float>& matrix,
         const int numIterations = 1);

    void rowReordering(const float similarityThreshold,
                       const sparseMatrix::CSR<float>& matrix,
                       const int numIterations = 1);

    void colReordering(const float blockDensityThreshold,
                       const sparseMatrix::CSR<float>& matrix,
                       const std::vector<UIN>& reorderedRows = std::vector<UIN>(),
                       const int numIterations = 1);

    int numRowPanels() const{ return numRowPanels_; }
    const std::vector<UIN>& reorderedRows() const{ return reorderedRows_; }
    const std::vector<UIN>& denseCols() const{ return denseCols_; }
    const std::vector<UIN>& denseColOffsets() const{ return denseColOffsets_; }
    const std::vector<UIN>& sparseCols() const{ return sparseCols_; }
    const std::vector<UIN>& sparseColOffsets() const{ return sparseColOffsets_; }
    const std::vector<UIN>& sparseValueOffsets() const{ return sparseValueOffsets_; }
    int numClusters() const{ return numClusters_; }
    float rowReorderingTime() const{ return rowReorderingTime_; }
    float colReorderingTime() const{ return colReorderingTime_; }
    float reorderingTime() const{ return rowReorderingTime_ + colReorderingTime_; }

private:
    int numRowPanels_ = 0;
    std::vector<UIN> reorderedRows_;
    std::vector<UIN> denseCols_;
    std::vector<UIN> denseColOffsets_;
    std::vector<UIN> sparseCols_;
    std::vector<UIN> sparseColOffsets_;
    std::vector<UIN> sparseValueOffsets_;

    int numClusters_ = 1;
    float rowReorderingTime_ = 0.0f;
    float colReorderingTime_ = 0.0f;
};

/**
 * @className: RPHM
 * @classInterpretation: Store dense tiled in BELL format, and sparse tiled in COO format.
 * @MemberVariables:
 * `reorderedRows_`: Store the reordered row indexes.
 * `denseCols_`: Store the reordered dense column indexes for each row panel in order.
 * `denseColOffsets_`: Offset array of reordered dense column array in each row panel.
 * `blockValues_`: BELL format. Stores the index of the original matrix element.
 * `blockOffsets_`: BELL format. Stores the number of column blocks in each row panel.
 * `sparseDataOffsets_`: size of the number of row panels + 1. Stores the number of data in each row panel.
 * `sparseData_`: values in COO format.
 * `sparseRelativeRows_`: row indices in COO format, but relative to the row panel.
 * `sparseCols_`: column indices in COO format.
 **/
class RPHM{
public:
    RPHM() = default;

    RPHM(const sparseMatrix::CSR<float>& matrix, const BSMR& bsmr);

    UIN numRowPanels() const{ return numRowPanels_; }
    UIN maxNumDenseColBlocksInRowPanel() const{ return maxNumDenseColBlocksInRowPanel_; }
    UIN maxNumSparseColBlocksInRowPanel() const{ return maxNumSparseColBlocksInRowPanel_; }
    UIN numDenseThreadBlocks() const{ return numDenseThreadBlocks_; }
    UIN numSparseThreadBlocks() const{ return numSparseThreadBlocks_; }
    const dev::vector<UIN>& reorderedRows() const{ return reorderedRows_; }
    const dev::vector<UIN>& denseCols() const{ return denseCols_; }
    const dev::vector<UIN>& blockValues() const{ return blockValues_; }
    const dev::vector<UIN>& blockOffsets() const{ return blockOffsets_; }
    const dev::vector<UIN>& sparseValueOffsets() const{ return sparseValueOffsets_; }
    const dev::vector<UIN>& sparseValues() const{ return sparseValues_; }
    const dev::vector<UIN>& sparseRelativeRows() const{ return sparseRelativeRows_; }
    const dev::vector<UIN>& sparseColIndices() const{ return sparseColIndices_; }

    const dev::vector<UIN>& denseRowPanelIds() const{ return denseRowPanelIds_; }
    const dev::vector<UIN> denseColBlockIters() const{ return denseColBlockIters_; }
    const dev::vector<UIN>& sparseRowPanelIds() const{ return sparseRowPanelIds_; }
    const dev::vector<UIN>& sparseColBlockIters() const{ return sparseColBlockIters_; }

    float time() const{ return reorderingTime_; }

    // Calculate the rowPanelID by blockValueIndex
    UIN calculateRowPanelIdByBlockValuesIndex(UIN blockValueIndex) const;

    // Calculate the localRow and localCol by blockValueIndex
    std::pair<UIN, UIN> calculateLocalRowColByBlockValueIndex(UIN blockValueIndex) const;

    // Calculate the row and col by blockValueIndex
    std::pair<UIN, UIN> calculateRowColByBlockValueIndex(UIN blockValueIndex) const;

    // Calculate the colBlockId in row panel by blockValueIndex
    UIN calculateColBlockIdByBlockValueIndex(UIN blockValueIndex) const;

    UIN getNumDenseBlocks() const{ return blockOffsets().back_data(); }

    UIN getNumSparseBlocks() const;

    // Calculate the average density of all blocks
    float calculateDenseBlockAverageDensity() const;

    // Calculate the maximum and minimum density of all blocks
    std::pair<float, float> calculateMaxMinDensity() const;

    // Calculate the mode density and its frequency among all blocks.
    std::pair<float, UIN> calculateDensityMode() const;

private:
    UIN numRowPanels_ = 0;
    UIN maxNumDenseColBlocksInRowPanel_ = 0;
    UIN maxNumSparseColBlocksInRowPanel_ = 0;
    UIN numDenseThreadBlocks_ = 0;
    UIN numSparseThreadBlocks_ = 0;

    // Reordered row indexes
    dev::vector<UIN> reorderedRows_;

    // Dense block data
    dev::vector<UIN> denseCols_;
    dev::vector<UIN> blockOffsets_;
    dev::vector<UIN> blockValues_;

    // Sparse block data
    dev::vector<UIN> sparseValueOffsets_;
    dev::vector<UIN> sparseValues_;
    dev::vector<UIN> sparseRelativeRows_;
    dev::vector<UIN> sparseColIndices_;

    // Row panel IDs and column block iterators
    dev::vector<UIN> denseRowPanelIds_;
    dev::vector<UIN> denseColBlockIters_;
    dev::vector<UIN> sparseRowPanelIds_;
    dev::vector<UIN> sparseColBlockIters_;

    float reorderingTime_ = 0.0f;
};

void noReorderRow(const sparseMatrix::CSR<float>& matrix, std::vector<UIN>& reorderedRows, float& time);

/**
 * @funcitonName: rowReordering_cpu
 * @functionInterpretation: Sort rows by row similarity
 * @input:
 * `matrix`: Sparse matrix data in CSR format.
 * @output: Update `reorderingRows_`.
 **/
void rowReordering_cpu(const sparseMatrix::CSR<float>& matrix, std::vector<UIN>& rows, float& time);

void rowReordering_gpu(const sparseMatrix::CSR<float>& matrix,
                       const float similarity_threshold_alpha,
                       const int blockSize,
                       std::vector<UIN>& reorderedRows,
                       float& time);

UIN calculateBlockSize(const sparseMatrix::CSR<float>& matrix);

std::vector<int> bsa_rowReordering_cpu(const sparseMatrix::CSR<float>& matrix,
                                       const float similarity_threshold_alpha,
                                       const int block_size,
                                       float& reordering_time);

std::vector<UIN> bsa_rowReordering_gpu(const sparseMatrix::CSR<float>& matrix,
                                       const float alpha,
                                       const UIN block_size,
                                       int& num_clusters,
                                       float& reordering_time);

/**
 * @funcitonName: colReordering
 * @functionInterpretation: Divide rows into row panels and columns reordered in each row panel.
 * @input:
 * `matrix`: Sparse matrix data in CSR format.
 * @output: Update `reorderingColsOffset_` and `reorderingCols_`.
 **/
void colReordering(const sparseMatrix::CSR<float>& matrix,
                   const UIN numRowPanels,
                   const std::vector<UIN>& reorderedRows,
                   std::vector<UIN>& reorderedCols,
                   std::vector<UIN>& reorderedColOffsets);

/**
 * @funcitonName: colReordering
 * @functionInterpretation: Divide rows into row panels and columns reordered in each row panel. After the columns reordered, the columns are divided into dense and sparse residual columns.
 * @input:
 * `matrix`: Sparse matrix data in CSR format.
 * `reorderedRows` : Reordered row index array.
 * @output:
 **/
void colReordering_cpu(const sparseMatrix::CSR<float>& matrix,
                       const UIN numRowPanels,
                       const std::vector<UIN>& reorderedRows,
                       const float blockDensityThreshold,
                       std::vector<UIN>& denseCols,
                       std::vector<UIN>& denseColOffsets,
                       std::vector<UIN>& sparseCols,
                       std::vector<UIN>& sparseColOffsets,
                       std::vector<UIN>& sparseDataOffsets,
                       float& time);

void colReordering_gpu(const sparseMatrix::CSR<float>& matrix,
                       const UIN numRowPanels,
                       const std::vector<UIN>& reorderedRows,
                       const UIN dense_column_segment_threshold,
                       std::vector<UIN>& denseCols,
                       std::vector<UIN>& denseColOffsets,
                       std::vector<UIN>& sparseCols,
                       std::vector<UIN>& sparseColOffsets,
                       std::vector<UIN>& sparseDataOffsets,
                       float& time);

// Error checking
bool check_rphm(const sparseMatrix::CSR<float>& matrix,
                const BSMR& bsmr,
                const RPHM& rphm,
                const float denseColSegmentThreshold);

// Calculate the number of tiles and average density in the original matrix
std::pair<UIN, float> calculateNumDenseBlocksAndAverageDensityInOriginalMatrix(
    const float densityThreshold,
    const sparseMatrix::CSR<float>& matrix);

void evaluationReordering(const sparseMatrix::CSR<float>& matrix, const BSMR& bsmr, Logger& logger);
