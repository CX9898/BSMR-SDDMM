#pragma once

#include <cuda_fp16.h>

using MATRIX_A_TYPE = __half;
using MATRIX_B_TYPE = __half;
using MATRIX_C_TYPE = float;

// The dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;