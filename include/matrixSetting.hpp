#pragma once

#include "wmmaSetting.hpp"

// Must be multiples of WMMA supported dimensions for WMMA code to work
const int MATRIX_M = 515 * WMMA_M;
const int MATRIX_N = 515 * WMMA_N;
const int MATRIX_K = 515 * WMMA_K;

const int MATRIX_A_SIZE = MATRIX_M * MATRIX_K;
const int MATRIX_B_SIZE = MATRIX_K * MATRIX_N;
const int MATRIX_C_SIZE = MATRIX_M * MATRIX_N;