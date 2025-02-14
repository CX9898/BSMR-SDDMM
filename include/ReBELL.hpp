#pragma once

#include "Matrix.hpp"

constexpr float row_similarity_threshold_alpha = 0.1f;

constexpr UIN ROW_PANEL_SIZE = WMMA_M;
constexpr UIN BLOCK_COL_SIZE = WMMA_N;
constexpr UIN BLOCK_SIZE = ROW_PANEL_SIZE * BLOCK_COL_SIZE;

constexpr UIN NULL_VALUE = MAX_UIN;

/**
 * @className: ReBELL
 * @classInterpretation:
 * @MemberVariables:
 * `reorderedRows_`: Sorted row index array.
 * `reorderedCols_`: Sorted col index array in each row panel.
 * `reorderedRowPanelOffset_`: Offset array of col array in each row panel.
 * `blockValues_`: BELL format. Stores the index of the original matrix element.
 * `blockRowOffsets_`: BELL format. Stores the number of column blocks in each row panel
 **/
class ReBELL {
 public:
  ReBELL(const sparseMatrix::CSR<float> &matrix, float& time);

  UIN numRowPanels() const { return numRowPanels_; }
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
  std::pair<UIN, UIN> calculateLocalRowColByColIndex(UIN blockValueIndex) const;

  // Calculate the row and col by blockValueIndex
  std::pair<UIN, UIN> calculateRowColByBlockValueIndex(UIN blockValueIndex) const;

 private:
  UIN numRowPanels_;
  std::vector<UIN> reorderedRows_;
  std::vector<UIN> reorderedCols_;
  std::vector<UIN> reorderedColOffsets_;

  std::vector<UIN> blockValues_;
  std::vector<UIN> blockRowOffsets_;

  /**
   * @funcitonName: rowReordering
   * @functionInterpretation: Sort rows by row similarity
   * @input:
   * `matrix`: Sparse matrix data in CSR format.
   * @output: Update `reorderingRows_`.
   **/
  void rowReordering(const sparseMatrix::CSR<float> &matrix);

  /**
   * @funcitonName: colReordering
   * @functionInterpretation: Divide rows into row panels and sort the columns in each row panel.
   * @input:
   * `matrix`: Sparse matrix data in CSR format.
   * @output: Update `reorderingColsOffset_` and `reorderingCols_`.
   **/
  void colReordering(const sparseMatrix::CSR<float> &matrix);
};

// Error checking
bool check_rebell(const sparseMatrix::CSR<float> &matrix, const struct ReBELL &rebell);