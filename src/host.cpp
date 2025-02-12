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

void sddmm_cpu_coo_isratnisa(
    const Matrix<float> &matrixA,
    const Matrix<float> &matrixB,
    const SparseMatrix<float> &matrixS,
    SparseMatrix<float> &matrixP) {
    const int K = matrixA.col();

    // reduction(+:rmse)
    double start_time = omp_get_wtime();
    omp_set_dynamic(0);
    omp_set_num_threads(28);
#pragma omp parallel for //reduction(+:tot)
    for (int idx = 0; idx < matrixS.nnz(); ++idx) {
        float sm = 0;
        const int row = matrixS.rowIndices()[idx];
        const int col = matrixS.colIndices()[idx];
        for (int t = 0; t < K; ++t)
            sm += matrixA.values()[row * K + t] * matrixB.values()[col * K + t];
        matrixP.setValues()[idx] = sm;//* val_ind[ind];
        // cout << "ind " << row<<" "<<col << ":: "  <<" "<< p_ind[ind] << " = " << sm <<" * "<< val_ind[ind]<< endl;
    }
    double CPU_time = omp_get_wtime() - start_time;
    //correctness check

    printf("\nomp time CPU : %.4f \n\n", CPU_time * 1000);
}

template<typename T>
void sddmm_cpu(
    const Matrix<T> &matrixA,
    const Matrix<T> &matrixB,
    const SparseMatrix<T> &matrixS,
    SparseMatrix<T> &matrixP) {
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

        val *= matrixS.values()[matrixSIdx];
        matrixP.setValues()[matrixSIdx] = val;
    }
}

template void sddmm_cpu<int>(const Matrix<int> &matrixA,
                             const Matrix<int> &matrixB,
                             const SparseMatrix<int> &matrixS,
                             SparseMatrix<int> &matrixP);

template void sddmm_cpu<float>(const Matrix<float> &matrixA,
                               const Matrix<float> &matrixB,
                               const SparseMatrix<float> &matrixS,
                               SparseMatrix<float> &matrixP);

template void sddmm_cpu<double>(const Matrix<double> &matrixA,
                                const Matrix<double> &matrixB,
                                const SparseMatrix<double> &matrixS,
                                SparseMatrix<double> &matrixP);

template<typename T>
void sddmm_cpu(
    const Matrix<T> &matrixA,
    const Matrix<T> &matrixB,
    const sparseMatrix::CSR<T> &matrixS,
    sparseMatrix::CSR<T> &matrixP) {
    if (matrixA.col() != matrixB.row() ||
        matrixA.row() != matrixP.row_ ||
        matrixB.col() != matrixP.col_) {
        std::cerr << "The storage of the three matrices does not match" << std::endl;
        return;
    }
    const int K = matrixA.col();
#pragma omp parallel for
    for (int row = 0; row < matrixS.row_; ++row) {
        for (int matrixSIdx = matrixS.rowOffsets_[row]; matrixSIdx < matrixS.rowOffsets_[row + 1]; ++matrixSIdx) {
            const size_t col = matrixS.colIndices_[matrixSIdx];

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

            val *= matrixS.values_[matrixSIdx];
            matrixP.values_[matrixSIdx] = val;
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