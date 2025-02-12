#pragma once

#include "Matrix.hpp"

constexpr float row_similarity_threshold_alpha = 0.1f;

constexpr UIN row_panel_size = WMMA_M;
constexpr UIN block_col_size = WMMA_N;
constexpr UIN block_size = row_panel_size * block_col_size;

constexpr UIN NULL_VALUE = MAX_UIN;

/**
 * @className: ReBELL
 * @classInterpretation:
 * @MemberVariables:
 * `reorderedRows_`: Sorted row index array.
 * `reorderedCols_`: Offset array of col index array in each row panel.
 * `reorderedColsOffset_`: Sorted col index array in each row panel.
 * `blockValues_`:
 * `blockRowOffsets_`:
 **/
class ReBELL {
 public:
  ReBELL(const sparseDataType::CSR<float> &matrix);

  UIN numRowPanels() const { return numRowPanels_; }
  const std::vector<UIN> &reorderedRows() const { return reorderedRows_; }
  const std::vector<UIN> &reorderedCols() const { return reorderedCols_; }
  const std::vector<UIN> &reorderedColsOffset() const { return reorderedColsOffset_; }
  const std::vector<UIN> &blockValues() const { return blockValues_; }
  const std::vector<UIN> &blockRowOffsets() const { return blockRowOffsets_; }

  // Calculate the rowPanelID by blockValueIndex
  UIN calculateRowPanelId(UIN blockValueIndex) const;

 private:
  UIN numRowPanels_;
  std::vector<UIN> reorderedRows_;
  std::vector<UIN> reorderedCols_;
  std::vector<UIN> reorderedColsOffset_;

  std::vector<UIN> blockValues_;
  std::vector<UIN> blockRowOffsets_;

  /**
   * @funcitonName: rowReordering
   * @functionInterpretation: Sort rows by row similarity
   * @input:
   * `matrix`: Sparse matrix data in CSR format.
   * @output: Update `reorderingRows_`.
   **/
  void rowReordering(const sparseDataType::CSR<float> &matrix);

  /**
   * @funcitonName: colReordering
   * @functionInterpretation: Divide rows into row panels and sort the columns in each row panel.
   * @input:
   * `matrix`: Sparse matrix data in CSR format.
   * @output: Update `reorderingColsOffset_` and `reorderingCols_`.
   **/
  void colReordering(const sparseDataType::CSR<float> &matrix);
};

bool check_rebell(const sparseDataType::CSR<float> &matrix, const struct ReBELL &rebell);