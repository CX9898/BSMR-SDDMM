#include "reordering.hpp"

ReorderedMatrix reordering(const sparseDataType::CSR<float> &matrix) {
    ReorderedMatrix reorderedMatrix;
    row_reordering(matrix, reorderedMatrix);
    col_reordering(matrix, reorderedMatrix);
    return reorderedMatrix;
}