#pragma once

#include "Matrix.hpp"

const float row_similarity_threshold_alpha = 0.3f;

const int row_panel_size = WMMA_M;
const int block_col_size = WMMA_N;
const int block_size = row_panel_size * block_col_size;

/**
 * @structName: ReorderedMatrix
 * @structInterpretation: Main entry function. Calls `row_reordering` and `col_reordering` to reorder rows and columns respectively.
 * @MemberVariables:
 * `rowIndices_`: Sorted row index array.
 * `colIndicesOffset_`: Offset array of col index array in each row panel.
 * `colIndicesInEachRowPanel_`: Sorted col index array in each row panel.
 **/
struct ReorderedMatrix {
  std::vector<UIN> reorderedRowIndices_;
  std::vector<UIN> reorderedColIndices_;
  std::vector<UIN> reorderedColIndicesOffset_;
};

struct ReBELL: public ReorderedMatrix{
  ReBELL() = default;
  ReBELL(const sparseDataType::CSR<float> &csrMatrix);

  std::vector<UIN> blockValues_;
  std::vector<UIN> blockRowOffsets_;
};


/**
 * @funcitonName: reordering
 * @functionInterpretation: Main entry function. Calls `row_reordering` and `col_reordering` to reorder rows and columns respectively.
 * @input:
 * `matrix`: Sparse matrix data in CSR format.
 * @output: ReorderedMatrix struct
 **/
ReorderedMatrix reordering(const sparseDataType::CSR<float> &matrix);

/**
 * @funcitonName: row_reordering
 * @functionInterpretation: Sort rows by row similarity
 * @input:
 * `matrix`: Sparse matrix data in CSR format.
 * @output: Update `rowIndices_` in the ReorderedMatrix structure.
 **/
void row_reordering(const sparseDataType::CSR<float> &matrix, struct ReorderedMatrix &reorderedMatrix);

/**
 * @funcitonName: col_reordering
 * @functionInterpretation: Divide rows into row panels and sort the columns in each row panel.
 * @input:
 * `matrix`: Sparse matrix data in CSR format.
 * And `rowIndices_` in the ReorderedMatrix struct.
 * @output: Update `colIndicesOffset_` and `colIndicesInEachRowPanel_` in the ReorderedMatrix structure.
 **/
void col_reordering(const sparseDataType::CSR<float> &matrix, struct ReorderedMatrix &reorderedMatrix);