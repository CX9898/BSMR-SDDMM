#include "reordering.hpp"

ReorderedMatrix reordering(const sparseDataType::CSR &matrix) {
    ReorderedMatrix reorderedMatrix;
    row_reordering(matrix, reorderedMatrix);
    col_reordering(matrix, reorderedMatrix);
    return reorderedMatrix;
}