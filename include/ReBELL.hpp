#pragma once

#include "Matrix.hpp"

constexpr float row_similarity_threshold_alpha = 0.1f;

constexpr UIN row_panel_size = WMMA_M;
constexpr UIN block_col_size = WMMA_N;
constexpr UIN block_size = row_panel_size * block_col_size;

/**
 * @structName: ReBELL
 * @structInterpretation:
 * @MemberVariables:
 * `reorderedRowIndices_`: Sorted row index array.
 * `reorderedColIndices_`: Offset array of col index array in each row panel.
 * `reorderedColIndicesOffset_`: Sorted col index array in each row panel.
 * `blockValues_`:
 * `blockRowOffsets_`:
 **/
struct ReBELL {
 public:
  ReBELL() = default;
  ReBELL(const sparseDataType::CSR<float> &matrix);

  UIN numRowPanels_;
  std::vector<UIN> reorderedRowIndices_;
  std::vector<UIN> reorderedColIndices_;
  std::vector<UIN> reorderedColIndicesOffset_;

  std::vector<UIN> blockValues_;
  std::vector<UIN> blockRowOffsets_;

 private:
  /**
   * @funcitonName: rowReordering
   * @functionInterpretation: Sort rows by row similarity
   * @input:
   * `matrix`: Sparse matrix data in CSR format.
   * @output: Update `rowIndices_` in the ReorderedMatrix structure.
   **/
  void rowReordering(const sparseDataType::CSR<float> &matrix);

  /**
   * @funcitonName: colReordering
   * @functionInterpretation: Divide rows into row panels and sort the columns in each row panel.
   * @input:
   * `matrix`: Sparse matrix data in CSR format.
   * And `rowIndices_` in the ReorderedMatrix struct.
   * @output: Update `colIndicesOffset_` and `colIndicesInEachRowPanel_` in the ReorderedMatrix structure.
   **/
  void colReordering(const sparseDataType::CSR<float> &matrix);
};

bool check_rowReordering(const sparseDataType::CSR<float> &matrix, const struct ReBELL &rebell);
bool check_colReordering(const sparseDataType::CSR<float> &matrix, const struct ReBELL &rebell);