#include "reordering.hpp"

void reordering(const sparseDataType::CSR &matrix) {
    std::vector<UIN> sortedRowIndex = row_reordering(matrix);
    col_reordering(matrix, sortedRowIndex);
}