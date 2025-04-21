#include <omp.h>

#include "host.hpp"

template<typename T>
void dmm_cpu(const Matrix<T> &matrixA,
             const Matrix<T> &matrixB,
             Matrix<T> &matrixC) {
    if (matrixA.col() != matrixB.row() ||
        matrixA.row() != matrixC.row() ||
        matrixB.col() != matrixC.col()) {
        std::cerr << "The storage of the three matrices does not match" << std::endl;
        return;
    }
    const int K = matrixA.col();
#pragma omp parallel for
    for (int mtxCIdx = 0; mtxCIdx < matrixC.size(); ++mtxCIdx) {
        const int row = matrixC.rowOfValueIndex(mtxCIdx);
        const int col = matrixC.colOfValueIndex(mtxCIdx);
        float val = 0.0f;
        for (int kIter = 0; kIter < K; ++kIter) {
            const auto valA = matrixA.getOneValueForMultiplication(
                MatrixMultiplicationOrder::left_multiplication,
                row, col, kIter);
            const auto valB = matrixB.getOneValueForMultiplication(
                MatrixMultiplicationOrder::right_multiplication,
                row, col, kIter);
            val += valA * valB;
        }
        matrixC[mtxCIdx] = val;
    }
}

template void dmm_cpu<int>(const Matrix<int> &matrixA,
                           const Matrix<int> &matrixB,
                           Matrix<int> &matrixC);
template void dmm_cpu<float>(const Matrix<float> &matrixA,
                             const Matrix<float> &matrixB,
                             Matrix<float> &matrixC);
template void dmm_cpu<double>(const Matrix<double> &matrixA,
                              const Matrix<double> &matrixB,
                              Matrix<double> &matrixC);

template<typename T>
void sddmm_cpu(
    const Matrix<T> &matrixA,
    const Matrix<T> &matrixB,
    const sparseMatrix::CSR<T> &matrixS,
    sparseMatrix::CSR<T> &matrixP) {
    if (matrixA.col() != matrixB.row() ||
        matrixA.row() != matrixP.row() ||
        matrixB.col() != matrixP.col()) {
        std::cerr << "The storage of the three matrices does not match" << std::endl;
        return;
    }
    const int K = matrixA.col();
#pragma omp parallel for
    for (int row = 0; row < matrixS.row(); ++row) {
        for (int matrixSIdx = matrixS.rowOffsets()[row]; matrixSIdx < matrixS.rowOffsets()[row + 1]; ++matrixSIdx) {
            const size_t col = matrixS.colIndices()[matrixSIdx];

            float val = 0.0f;
            for (int kIter = 0; kIter < K; ++kIter) {
                const auto valA = matrixA.getOneValueForMultiplication(
                    MatrixMultiplicationOrder::left_multiplication,
                    row, col, kIter);
                const auto valB = matrixB.getOneValueForMultiplication(
                    MatrixMultiplicationOrder::right_multiplication,
                    row, col, kIter);
                val += valA * valB;
            }

            matrixP.setValues()[matrixSIdx] = val;
        }
    }
}

template void sddmm_cpu<int>(const Matrix<int> &matrixA,
                             const Matrix<int> &matrixB,
                             const sparseMatrix::CSR<int> &matrixS,
                             sparseMatrix::CSR<int> &matrixP);

template void sddmm_cpu<float>(const Matrix<float> &matrixA,
                               const Matrix<float> &matrixB,
                               const sparseMatrix::CSR<float> &matrixS,
                               sparseMatrix::CSR<float> &matrixP);

template void sddmm_cpu<double>(const Matrix<double> &matrixA,
                                const Matrix<double> &matrixB,
                                const sparseMatrix::CSR<double> &matrixS,
                                sparseMatrix::CSR<double> &matrixP);

template<typename T>
void sddmm_cpu(
    const Matrix<T> &matrixA,
    const Matrix<T> &matrixB,
    const sparseMatrix::COO<T> &matrixS,
    sparseMatrix::COO<T> &matrixP) {
    if (matrixA.col() != matrixB.row() ||
        matrixA.row() != matrixP.row() ||
        matrixB.col() != matrixP.col()) {
        std::cerr << "The storage of the three matrices does not match" << std::endl;
        return;
    }
    const int K = matrixA.col();
#pragma omp parallel for
    for (int matrixSIdx = 0; matrixSIdx < matrixS.nnz(); ++matrixSIdx) {
        const size_t row = matrixS.rowIndices()[matrixSIdx];
        const size_t col = matrixS.colIndices()[matrixSIdx];

        float val = 0.0f;
        for (int kIter = 0; kIter < K; ++kIter) {
            const auto valA = matrixA.getOneValueForMultiplication(
                MatrixMultiplicationOrder::left_multiplication,
                row, col, kIter);
            const auto valB = matrixB.getOneValueForMultiplication(
                MatrixMultiplicationOrder::right_multiplication,
                row, col, kIter);
            val += valA * valB;
        }

//        val *= matrixS.values()[matrixSIdx];
        matrixP.setValues()[matrixSIdx] = val;
    }
}

template void sddmm_cpu<int>(const Matrix<int> &matrixA,
                             const Matrix<int> &matrixB,
                             const sparseMatrix::COO<int> &matrixS,
                             sparseMatrix::COO<int> &matrixP);

template void sddmm_cpu<float>(const Matrix<float> &matrixA,
                               const Matrix<float> &matrixB,
                               const sparseMatrix::COO<float> &matrixS,
                               sparseMatrix::COO<float> &matrixP);

template void sddmm_cpu<double>(const Matrix<double> &matrixA,
                                const Matrix<double> &matrixB,
                                const sparseMatrix::COO<double> &matrixS,
                                sparseMatrix::COO<double> &matrixP);