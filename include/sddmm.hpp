#pragma once

#include "Matrix.hpp"
#include "Logger.hpp"
#include "Options.hpp"

// Using reordering method for sddmm operations
void sddmm(const Options& options,
           const Matrix<float>& matrixA,
           const Matrix<float>& matrixB,
           sparseMatrix::CSR<float>& matrixP,
           Logger& logger);

void sddmm_testMode(const Options& options,
                    sparseMatrix::CSR<float>& matrixP);

// Error check
bool checkSddmm(const Matrix<float>& matrixA,
                const Matrix<float>& matrixB,
                const sparseMatrix::CSR<float>& matrixS,
                const sparseMatrix::CSR<float>& matrixP);

/** CPU 参考 SDDMM 与 matrixP.values()（GPU）对比；稀疏模式与 matrixP 一致，保证与重排后存储顺序对齐 */
void fillLoggerCpuGpuAccuracyCompare(Logger& logger,
                                     const Matrix<float>& matrixA,
                                     const Matrix<float>& matrixB,
                                     const sparseMatrix::CSR<float>& matrixP);

/**
 * 在与 BSMR 相同的 CSR 结构上再跑 cuSPARSE SDDMM（FP32），将 BSMR(matrixP) 与 cuSPARSE 的逐元差异统计写入 logger。
 * 相对 Frobenius 等指标的分母为 cuSPARSE 结果向量的范数。
 */
void fillLoggerBsmrVsCuSparseAccuracyCompare(Logger& logger,
                                             const Matrix<float>& matrixA,
                                             const Matrix<float>& matrixB,
                                             const sparseMatrix::CSR<float>& matrixP_bsmr);
