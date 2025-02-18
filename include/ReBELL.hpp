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
 * `reorderedRows_`: Reordered row index array.
 * `reorderedCols_`: Reordered col index array in each row panel.
 * `reorderedRowPanelOffset_`: Offset array of col array in each row panel.
 * `blockValues_`: BELL format. Stores the index of the original matrix element.
 * `blockRowOffsets_`: BELL format. Stores the number of column blocks in each row panel
 **/
class ReBELL {
 public:
  ReBELL(const sparseMatrix::CSR<float> &matrix, float &time);

  UIN numRowPanels() const { return numRowPanels_; }
  UIN maxNumColBlocks() const { return maxNumColBlocks_; }
  const std::vector<UIN> &reorderedRows() const { return reorderedRows_; }
  const std::vector<UIN> &reorderedCols() const { return reorderedCols_; }
  const std::vector<UIN> &reorderedColOffsets() const { return reorderedColOffsets_; }
  const std::vector<UIN> &blockValues() const { return blockValues_; }
  const std::vector<UIN> &blockRowOffsets() const { return blockRowOffsets_; }

  // Calculate the rowPanelID by blockValueIndex
  UIN calculateRowPanelIdByBlockValuesIndex(UIN blockValueIndex) const;

  // Calculate the rowPanelID by reorderedColIndex
  UIN calculateRowPanelIdByColIndex(UIN reorderedColIndex) const;

  // Calculate the localRow and localCol by blockValueIndex
  std::pair<UIN, UIN> calculateLocalRowColByBlockValueIndex(UIN blockValueIndex) const;

  // Calculate the row and col by blockValueIndex
  std::pair<UIN, UIN> calculateRowColByBlockValueIndex(UIN blockValueIndex) const;

  UIN getNumBlocks() const { return blockRowOffsets().back(); }

  // Calculate the average density of all blocks
  float calculateAverageDensity();

  // Calculate the maximum and minimum density of all blocks
  std::pair<float, float> calculateMaxMinDensity();

 private:
  UIN numRowPanels_;
  UIN maxNumColBlocks_;

  std::vector<UIN> reorderedRows_;
  std::vector<UIN> reorderedCols_;
  std::vector<UIN> reorderedColOffsets_;

  std::vector<UIN> blockValues_;
  std::vector<UIN> blockRowOffsets_;
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

std::vector<int> bsa_rowReordering_gpu(const sparseMatrix::CSR<float> &matrix,
                                       float alpha,
                                       UIN block_size,
                                       float &reordering_time,
                                       int &cluster_cnt);

/**
 * @funcitonName: colReordering
 * @functionInterpretation: Divide rows into row panels and sort the columns in each row panel.
 * @input:
 * `matrix`: Sparse matrix data in CSR format.
 * @output: Update `reorderingColsOffset_` and `reorderingCols_`.
 **/
void colReordering(const sparseMatrix::CSR<float> &matrix,
                   const UIN numRowPanels,
                   const std::vector<UIN> &reorderedRows,
                   std::vector<UIN> &reorderedColOffsets,
                   std::vector<UIN> &reorderedCols);

// Error checking
bool check_rebell(const sparseMatrix::CSR<float> &matrix, const struct ReBELL &rebell);

// Calculate the number of tiles in the original matrix
UIN calculateNumTilesInOriginalMatrix(const sparseMatrix::CSR<float> &matrix);