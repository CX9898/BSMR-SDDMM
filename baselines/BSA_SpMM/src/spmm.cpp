#include "spmm.h"

int _spmm_bellpack(intT A_rows, intT A_cols, intT block_size, intT A_ell_rows, intT A_ell_cols, intT *A_ellColInd, DataT_H *A_ellValues, DataT_H *B, intT B_cols, DataT_C *C, float &elapsed_time)
{
    cudaDataType_t data_type_AB = CUDA_R_16F;
    cudaDataType_t data_type_C = CUDA_R_32F;
    cudaDataType_t compute_type = CUDA_R_32F;

    unsigned int B_rows = A_cols;
    unsigned int size_B = B_rows * B_cols;

    unsigned int C_rows = A_rows;
    unsigned int C_cols = B_cols;
    unsigned int size_C = C_rows * C_cols;

    intT num_blocks = A_ell_rows * A_ell_cols;
    intT ellValues_cols = A_ell_cols * block_size;

    intT ldb = B_cols;
    intT ldc = C_cols;

    intT *dA_ellColInd;
    DataT_H *dA_ellValues;

    float alpha = 1.0f;
    float beta = 1.0f;

    checkCudaErrors(cudaMalloc((void **)&dA_ellColInd, A_ell_rows * A_ell_cols * sizeof(intT)));
    checkCudaErrors(cudaMalloc((void **)&dA_ellValues, block_size * block_size * num_blocks * sizeof(DataT_H)));

    DataT *d_B;
    DataT_C *d_C;

    checkCudaErrors(cudaMalloc((void **)&d_B, size_B * sizeof(DataT)));
    checkCudaErrors(cudaMalloc((void **)&d_C, size_C * sizeof(DataT_C)));

    checkCudaErrors(cudaMemcpy(d_C, C, size_C * sizeof(DataT), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(dA_ellColInd, A_ellColInd, A_ell_rows * A_ell_cols * sizeof(intT), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dA_ellValues, A_ellValues, block_size * block_size * num_blocks * sizeof(DataT_H), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_B, B, size_B * sizeof(DataT_H), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    checkCudaErrors(cusparseCreate(&handle));
    checkCudaErrors(cusparseCreateBlockedEll(
        &matA,
        A_rows, A_cols, block_size, ellValues_cols,
        dA_ellColInd, dA_ellValues,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
        data_type_AB));

    checkCudaErrors(cusparseCreateDnMat(&matB, B_rows, B_cols, ldb, d_B, data_type_AB, CUSPARSE_ORDER_ROW));
    checkCudaErrors(cusparseCreateDnMat(&matC, C_rows, C_cols, ldc, d_C, data_type_C, CUSPARSE_ORDER_ROW));

    size_t bufferSize = 0;
    void *dBuffer = NULL;

    checkCudaErrors(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, compute_type,
        CUSPARSE_SPMM_BLOCKED_ELL_ALG1, &bufferSize));

    checkCudaErrors(cudaMalloc(&dBuffer, bufferSize));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    checkCudaErrors(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, compute_type,
        CUSPARSE_SPMM_BLOCKED_ELL_ALG1, dBuffer));

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    checkCudaErrors(cusparseDestroySpMat(matA));
    checkCudaErrors(cusparseDestroyDnMat(matB));
    checkCudaErrors(cusparseDestroyDnMat(matC));

    checkCudaErrors(cudaMemcpy(C, d_C, size_C * sizeof(DataT_C), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(dBuffer));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(dA_ellValues));

    checkCudaErrors(cusparseDestroy(handle));

    return 0;
}

void spmm_bellpack(BSA_HYBRID &bsa_lhs, ARR &rhs, DataT_C *matC, float &elapsed_time, bool compress_rows)
{
    intT block_size = bsa_lhs.block_size;
    intT starting_row_panel = 0;
    // compress (pass the emptry rows)
    if (compress_rows)
    {
        for (intT i = 0; i < bsa_lhs.ell_rows; i++)
        {
            bool is_empty_row = true;
            for (intT j = 0; j < bsa_lhs.ell_cols; j++)
            {
                if (bsa_lhs.ellColInd[i * bsa_lhs.ell_cols + j] != -1)
                {
                    is_empty_row = false;
                    break;
                }
            }
            if (!is_empty_row)
            {
                starting_row_panel = i;
                break;
            }
        }

        intT new_ell_rows = bsa_lhs.ell_rows - starting_row_panel;
        intT new_lhs_rows = bsa_lhs.rows - (starting_row_panel * block_size);
        intT C_cols = rhs.cols;

        intT *new_ellColInd = bsa_lhs.ellColInd + (starting_row_panel * bsa_lhs.ell_cols);
        DataT *new_result_mat = matC + (starting_row_panel * block_size) * C_cols;
        DataT_H *new_ellValues = bsa_lhs.h_ellValues + (starting_row_panel * bsa_lhs.ell_cols * block_size * block_size);
        _spmm_bellpack(new_lhs_rows, bsa_lhs.cols, bsa_lhs.block_size, new_ell_rows, bsa_lhs.ell_cols, new_ellColInd, new_ellValues, rhs.h_mat, rhs.cols, new_result_mat, elapsed_time);
    }
    else
    {
        _spmm_bellpack(bsa_lhs.rows, bsa_lhs.cols, bsa_lhs.block_size, bsa_lhs.ell_rows, bsa_lhs.ell_cols, bsa_lhs.ellColInd, bsa_lhs.h_ellValues, rhs.h_mat, rhs.cols, matC, elapsed_time);
    }
}

int _spmm_csr(intT A_rows, intT A_cols, intT csr_total_nonzeros, intT *rowptr, intT *colidx, DataT *values, DataT *B, intT B_cols, DataT *C, float &elapsed_time)
{
    cudaDataType_t data_type_AB = CUDA_R_32F;
    cudaDataType_t data_type_C = CUDA_R_32F;

    unsigned int size_rowptr = (A_rows + 1);
    unsigned int size_colidx = (csr_total_nonzeros);
    unsigned int size_values = (csr_total_nonzeros);

    unsigned int B_rows = A_cols;
    unsigned int size_B = B_rows * B_cols;

    unsigned int C_rows = A_rows;
    unsigned int C_cols = B_cols;
    unsigned int size_C = C_rows * C_cols;

    intT *d_rowptr, *d_colidx;
    DataT *d_values;

    intT ldb = B_cols;
    intT ldc = C_cols;

    float alpha = 1.0f;
    float beta = 1.0f;

    checkCudaErrors(cudaMalloc((void **)&d_rowptr, size_rowptr * sizeof(intT)));
    checkCudaErrors(cudaMalloc((void **)&d_colidx, size_colidx * sizeof(intT)));
    checkCudaErrors(cudaMalloc((void **)&d_values, size_values * sizeof(DataT)));

    DataT *d_B;
    DataT_C *d_C;

    checkCudaErrors(cudaMalloc((void **)&d_B, size_B * sizeof(DataT)));
    checkCudaErrors(cudaMalloc((void **)&d_C, size_C * sizeof(DataT)));

    checkCudaErrors(cudaMemcpy(d_rowptr, rowptr, size_rowptr * sizeof(intT), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_colidx, colidx, size_colidx * sizeof(intT), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_values, values, size_values * sizeof(DataT), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMemcpy(d_B, B, size_B * sizeof(DataT), cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;

    checkCudaErrors(cusparseCreate(&handle));

    checkCudaErrors(
        cusparseCreateCsr(
            &matA, A_rows, A_cols, csr_total_nonzeros, d_rowptr, d_colidx, d_values,
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, data_type_AB));

    checkCudaErrors(
        cusparseCreateDnMat(&matB, B_rows, B_cols, ldb, d_B, data_type_AB, CUSPARSE_ORDER_ROW));
    checkCudaErrors(
        cusparseCreateDnMat(&matC, C_rows, C_cols, ldc, d_C, data_type_C, CUSPARSE_ORDER_ROW));

    size_t bufferSize = 0;
    void *dBuffer = NULL;

    checkCudaErrors(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta,
        matC, data_type_C, CUSPARSE_SPMM_ALG_DEFAULT,
        &bufferSize));

    checkCudaErrors(cudaMalloc(&dBuffer, bufferSize));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    checkCudaErrors(cudaEventRecord(start, 0));

    checkCudaErrors(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matA, matB, &beta, matC, data_type_C,
        CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    checkCudaErrors(cusparseDestroySpMat(matA));
    checkCudaErrors(cusparseDestroyDnMat(matB));
    checkCudaErrors(cusparseDestroyDnMat(matC));

    checkCudaErrors(cusparseDestroy(handle));
    checkCudaErrors(cudaMemcpy(C, d_C, size_C * sizeof(DataT_C), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_values));
    checkCudaErrors(cudaFree(d_rowptr));
    checkCudaErrors(cudaFree(d_colidx));
    return 0;
}

void spmm_csr(BSA_HYBRID &bsa_lhs, ARR &rhs, DataT_C *matC, float &elapsed_time)
{
    _spmm_csr(bsa_lhs.rows, bsa_lhs.cols, bsa_lhs.csr_total_nonzeros, bsa_lhs.csr_rowptr, bsa_lhs.csr_colidx, bsa_lhs.csr_values, rhs.mat, rhs.cols, matC, elapsed_time);
}

void cusparse_spmm(CSR &lhs, ARR &rhs, ARR &result_mat, LOGGER &logger, intT n_repetitions, bool pattern_only)
{
    vector<float> arr_csr_spmm_time(n_repetitions);
    float csr_spmm_time;
    if (pattern_only)
    {
        lhs.values = new DataT[lhs.rows * lhs.cols];
        for (int i = 0; i < lhs.rows * lhs.cols; i++)
        {
            lhs.values[i] = 1.0f;
        }
    }
    for (int i = 0; i < n_repetitions; i++)
    {
        _spmm_csr(lhs.rows, lhs.cols, lhs.total_nonzeros, lhs.rowptr, lhs.colidx, lhs.values, rhs.mat, rhs.cols, result_mat.mat, csr_spmm_time);
        arr_csr_spmm_time[i] = csr_spmm_time;
    }

    logger.avg_csr_spmm_time = avg(arr_csr_spmm_time);
    logger.avg_total_spmm_time = avg(arr_csr_spmm_time);
}

int _cublas_gemm(const DataT *A, intT A_rows, intT A_cols, DataT *B, intT B_cols, DataT_C *C, float &elapsed_time)
{
    cudaDataType_t data_type_AB = CUDA_R_32F;
    cudaDataType_t data_type_C = CUDA_R_32F;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_16F;

    cublasGemmAlgo_t cuda_algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    intT B_rows = A_cols;

    intT C_rows = A_rows;
    intT C_cols = B_cols;

    intT size_A = A_rows * A_cols;
    intT size_B = B_rows * B_cols;
    intT size_C = C_rows * C_cols;

    cublasHandle_t handle;

    checkCudaErrors(cublasCreate(&handle));

    DataT *d_A, *d_B;
    DataT_C *d_C;

    checkCudaErrors(cudaMalloc((void **)&d_A, size_A * sizeof(DataT)));
    checkCudaErrors(cudaMalloc((void **)&d_B, size_B * sizeof(DataT)));
    checkCudaErrors(cudaMalloc((void **)&d_C, size_C * sizeof(DataT)));

    checkCudaErrors(cudaMemcpy(d_A, A, size_A * sizeof(DataT), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, B, size_B * sizeof(DataT), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    intT m = A_rows;
    intT n = B_cols;
    intT k = A_cols;

    intT lda = A_rows;
    intT ldb = B_rows;
    intT ldc = C_rows;

    float alpha = 1.0f;
    float beta = 0.0f;

    checkCudaErrors(cublasGemmEx(
        handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k, &alpha,
        d_A, data_type_AB, lda,
        d_B, data_type_AB, ldb,
        &beta,
        d_C, data_type_C, ldc,
        compute_type, cuda_algo));

    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    checkCudaErrors(cudaMemcpy(C, d_C, size_C * sizeof(DataT_C), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();

    checkCudaErrors(cudaFree(d_C));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));

    checkCudaErrors(cublasDestroy(handle));

    return 0;
}

void cublas_gemm(CSR &lhs, ARR &rhs, ARR &result_mat, LOGGER &logger, intT n_repetitions, bool pattern_only)
{
    vector<float> arr_gemm_time(n_repetitions);
    float gemm_time;
    DataT *arr_lhs = new DataT[lhs.rows * lhs.cols];
    memset(arr_lhs, 0, lhs.rows * lhs.cols * sizeof(DataT));
    if (pattern_only)
    {
        for (intT r = 0; r < lhs.rows; r++)
        {
            intT start_pos = lhs.rowptr[r];
            intT end_pos = lhs.rowptr[r + 1];
            for (intT nz = start_pos; nz < end_pos; nz++)
            {
                intT col = lhs.colidx[nz];
                if (pattern_only)
                    arr_lhs[col * lhs.rows + r] = 1;
                else
                    arr_lhs[col * lhs.rows + r] = lhs.values[nz];
            }
        }
    }
    for (int i = 0; i < n_repetitions; i++)
    {
        _cublas_gemm(arr_lhs, lhs.rows, lhs.cols, rhs.mat, rhs.cols, result_mat.mat, gemm_time);
        arr_gemm_time[i] = gemm_time;
    }

    logger.avg_total_spmm_time = avg(arr_gemm_time);
    delete[] arr_lhs;
}

void bsa_spmm(BSA_HYBRID &bsa_lhs, ARR &rhs, ARR &result_mat, LOGGER &logger, intT n_repetitions, bool compress_rows, bool valid)
{
    vector<float> arr_csr_spmm_time, arr_bellpack_spmm_time, arr_total_spmm_time;
    float csr_spmm_time, bellpack_spmm_time, total_spmm_time;

    for (int i = 0; i < n_repetitions; i++)
    {
        memset(result_mat.mat, 0, sizeof(result_mat.rows * result_mat.cols) * sizeof(DataT));
        if (bsa_lhs.csr_total_nonzeros)
        {
            cudaDeviceSynchronize();
            spmm_csr(bsa_lhs, rhs, result_mat.mat, csr_spmm_time);
            arr_csr_spmm_time.push_back(csr_spmm_time);
        }

        if (bsa_lhs.bellpack_total_nonzeros)
        {
            cudaDeviceSynchronize();
            spmm_bellpack(bsa_lhs, rhs, result_mat.mat, bellpack_spmm_time, compress_rows);
            arr_bellpack_spmm_time.push_back(bellpack_spmm_time);
            // printf("elapsed: %f\n", bellpack_spmm_time);
        }

        total_spmm_time = bellpack_spmm_time + csr_spmm_time;
        arr_total_spmm_time.push_back(total_spmm_time);
    }

    if (valid)
    {
#pragma omp parallel for num_threads(12)
        for (int i = 0; i < result_mat.rows; i++)
        {
            intT row = bsa_lhs.row_permutation[i];
            for (int j = 0; j < result_mat.cols; j++)
            {
                swap(result_mat.mat[i * result_mat.cols + j], result_mat.mat[i * result_mat.cols + j]);
            }
        }
    }

    logger.avg_csr_spmm_time = avg(arr_csr_spmm_time);
    logger.avg_bellpack_spmm_time = avg(arr_bellpack_spmm_time);
    logger.avg_total_spmm_time = avg(arr_total_spmm_time);
}