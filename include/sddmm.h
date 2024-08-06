#pragma once

#include "util.h"

int mainTmp(int argc, char *argv[]);

void preprocessing(const isratnisa::Matrix S,
                   const std::vector<float> &matrixAValues,
                   const std::vector<float> &matrixBValues,
                   float *matrixP);

void sddmm_GPU(const isratnisa::Matrix S,
               const isratnisa::TiledMatrix tS,
               float *matrixP,
               const vector<float> W, // 稠密矩阵 W, host 端数据
               const vector<float> H // 稠密矩阵 H, host 端数据
);
