#pragma once

#include "Matrix.hpp"

constexpr float row_similarity_threshold_alpha = 0.3f;

constexpr UIN ROW_PANEL_SIZE = WMMA_M;
constexpr UIN BLOCK_COL_SIZE = WMMA_N;
constexpr UIN BLOCK_SIZE = ROW_PANEL_SIZE * BLOCK_COL_SIZE;

constexpr UIN NULL_VALUE = MAX_UIN;

/**
 * @className: ReBELL
 * @classInterpretation: Reorder the rows and columns of a sparse matrix and store it in BELL format
 * @MemberVariables:
 * `reorderedRows_`: Store the reordered row indexes.
 * `reorderedCols_`: Store the reordered column indexes for each row panel in order.
 * `reorderedColOffsets_`: Offset array of reordered  column array in each row panel.
 * `blockValues_`: BELL format. Stores the index of the original matrix element.
 * `blockRowOffsets_`: BELL format. Stores the number of column blocks in each row panel
 **/
class ReBELL {
 public:
  ReBELL(const sparseMatrix::CSR<float> &matrix, float &time);

  UIN numRowPanels() const { return numRowPanels_; }
  UIN maxNumDenseColBlocks() const { return maxNumDenseColBlocks_; }
  UIN maxNumSparseColBlocks() const { return maxNumSparseColBlocks_; }
  const std::vector<UIN> &reorderedRows() const { return reorderedRows_; }
  const std::vector<UIN> &reorderedCols() const { return reorderedCols_; }
  const std::vector<UIN> &reorderedColOffsets() const { return reorderedColOffsets_; }
  const std::vector<UIN> &blockValues() const { return blockValues_; }
  const std::vector<UIN> &blockRowOffsets() const { return blockRowOffsets_; }
  const std::vector<UIN> &sparsePartDataOffsets() const { return sparsePartDataOffsets_; }
  const std::vector<UIN> &sparsePartData() const { return sparsePartData_; }
  const std::vector<UIN> &sparsePartRelativeRows() const { return sparsePartRelativeRows_; }
  const std::vector<UIN> &sparsePartColIndices() const { return sparsePartColIndices_; }

  UIN dense_column_segment_threshold() const { return dense_column_segment_threshold_; }

  // Calculate the rowPanelID by blockValueIndex
  UIN calculateRowPanelIdByBlockValuesIndex(UIN blockValueIndex) const;

  // Calculate the rowPanelID by reorderedColIndex
  UIN calculateRowPanelIdByColIndex(UIN reorderedColIndex) const;

  // Calculate the localRow and localCol by blockValueIndex
  std::pair<UIN, UIN> calculateLocalRowColByBlockValueIndex(UIN blockValueIndex) const;

  // Calculate the row and col by blockValueIndex
  std::pair<UIN, UIN> calculateRowColByBlockValueIndex(UIN blockValueIndex) const;

  // Calculate the colBlockId in row panel by blockValueIndex
  UIN calculateColBlockIdByBlockValueIndex(UIN blockValueIndex) const;

  UIN getNumDenseBlocks() const { return blockRowOffsets().back(); }
  UIN getNumSparseBlocks() const;

  // Calculate the average density of all blocks
  float calculateAverageDensity();

  // Calculate the maximum and minimum density of all blocks
  std::pair<float, float> calculateMaxMinDensity();

  // Calculate the mode density and its frequency among all blocks.
  std::pair<float, UIN> calculateDensityMode();

 private:
  UIN numRowPanels_;
  UIN maxNumDenseColBlocks_;
  UIN maxNumSparseColBlocks_;

  // Dense part data
  std::vector<UIN> reorderedRows_;
  std::vector<UIN> reorderedColOffsets_;
  std::vector<UIN> reorderedCols_;
  std::vector<UIN> blockRowOffsets_;
  std::vector<UIN> blockValues_;

  // Sparse part data
  std::vector<UIN> sparsePartDataOffsets_;
  std::vector<UIN> sparsePartData_;
  std::vector<UIN> sparsePartRelativeRows_;
  std::vector<UIN> sparsePartColIndices_;

  UIN dense_column_segment_threshold_;
};

/**
 * @funcitonName: rowReordering_cpu
 * @functionInterpretation: Sort rows by row similarity
 * @input:
 * `matrix`: Sparse matrix data in CSR format.
 * @output: Update `reorderingRows_`.
 **/
void rowReordering_cpu(const sparseMatrix::CSR<float> &matrix, std::vector<UIN> &rows, float &time);

std::vector<int> bsa_rowReordering_cpu(const sparseMatrix::CSR<float> &matrix,
                                       const float similarity_threshold_alpha,
                                       const int block_size,
                                       float &reordering_time);

std::vector<UIN> bsa_rowReordering_gpu(const sparseMatrix::CSR<float> &matrix,
                                       float alpha,
                                       UIN block_size,
                                       float &reordering_time);

/**
 * @funcitonName: colReordering
 * @functionInterpretation: Divide rows into row panels and columns reordered in each row panel.
 * @input:
 * `matrix`: Sparse matrix data in CSR format.
 * @output: Update `reorderingColsOffset_` and `reorderingCols_`.
 **/
void colReordering(const sparseMatrix::CSR<float> &matrix,
                   const UIN numRowPanels,
                   const std::vector<UIN> &reorderedRows,
                   std::vector<UIN> &reorderedCols,
                   std::vector<UIN> &reorderedColOffsets);

/**
 * @funcitonName: colReordering
 * @functionInterpretation: Divide rows into row panels and columns reordered in each row panel. After the columns reordered, the columns are divided into dense and sparse residual columns.
 * @input:
 * `matrix`: Sparse matrix data in CSR format.
 * `reorderedRows` : Reordered row index array.
 * @output:
 **/
void colReordering(const sparseMatrix::CSR<float> &matrix,
                   const UIN numRowPanels,
                   const std::vector<UIN> &reorderedRows,
                   const UIN dense_column_segment_threshold,
                   std::vector<UIN> &denseCols,
                   std::vector<UIN> &denseColOffsets,
                   std::vector<UIN> &sparseCols,
                   std::vector<UIN> &sparseColOffsets,
                   std::vector<UIN> &sparsePartDataOffsets);

// Error checking
bool check_rebell(const sparseMatrix::CSR<float> &matrix, const struct ReBELL &rebell);

// Calculate the number of tiles and average density in the original matrix
std::pair<UIN, float> calculateNumTilesAndAverageDensityInOriginalMatrix(const sparseMatrix::CSR<float> &matrix);