#pragma once

#include "Matrix.hpp"

#define COL_BLOCK_SIZE 32

const float similarity_threshold_alpha = 0.3f;

void row_reordering(const sparseDataType::CSR &matrix);

void col_reordering(const sparseDataType::CSR &matrix);