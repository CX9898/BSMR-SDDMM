#include <cuda_runtime.h>
#include <string>
#include <iostream>

#include "RoDeSddmm.h"
#include "matrix_utils.h"
#include "Options.hpp"

__global__ void FillValues(int n, float *array, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    array[idx] = val;
}

int main(int argc, char *argv[]) {

    int ITER = 10;
    // cudaSetDevice(0);

    Options options(argc, argv);
    const size_t k = options.K();
    const std::string file_path = options.inputFile();

    printf("[File : %s]\n", options.inputFile().c_str());
    printf("[K : %zu]\n", k);

    if (k == 32) {
        const int SEG_LENGTH = 512;

        cudaStream_t stream1, stream2;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);

        double gflops = 0.0f;

        SPC::SparseMatrix sm1(file_path, SPC::SORTED, 1);


        sm1.RowDivide2Segment(SEG_LENGTH, 4, 32);

        // printf("NR1 : %d , NR2 : %d\n",sm1.n_segs,sm1.n_segs_residue);

        SPC::CudaSparseMatrix<float> c_sm(sm1);
        int m = c_sm.Rows(), n = c_sm.Columns();

        absl::BitGen bitgen;
        SPC::CudaMatrix<float> d_B1(m, k, &bitgen);
        SPC::CudaMatrix<float> d_B2(n, k, &bitgen);

        int size = c_sm.Nonzeros();

        float *d_C;
        cudaMalloc((void **) &d_C, sizeof(float) * size);

        float *d_C1;
        cudaMalloc((void **) &d_C1, sizeof(float) * size);

        float *d_C2;
        cudaMalloc((void **) &d_C2, sizeof(float) * size);

        float *d_C3;
        cudaMalloc((void **) &d_C3, size * sizeof(float));

        float *d_C4;
        cudaMalloc((void **) &d_C4, size * sizeof(float));

        float *d_C5;
        cudaMalloc((void **) &d_C5, size * sizeof(float));

        float *diff;
        cudaMalloc((void **) &diff, sizeof(float) * 1);

        FillValues<<<(size + 31) / 32, 32>>>(size, d_C, 0.0f);
        FillValues<<<(size + 31) / 32, 32>>>(size, d_C1, 0.0f);
        FillValues<<<(size + 31) / 32, 32>>>(size, d_C2, 0.0f);
        // FillValues<<<(size+31)/32,32>>>(size,d_C3,0.0f);
        // FillValues<<<(size+31)/32,32>>>(size,d_C4,0.0f);

#ifdef VALIDATE
        StandCall(m,n,k,
                c_sm.RowIndices(),c_sm.RowOffsets(),c_sm.ColumnIndices(),c_sm.Values(),
                d_B1.Values(),d_B2.Values(),d_C);
#endif
        float tot_ms;
        cudaEvent_t event1, event2;
        cudaEventCreate(&event1);
        cudaEventCreate(&event2);

        cudaDeviceSynchronize();
        cudaEventRecord(event1, 0);

        for (int i = 0; i < ITER; ++i)
            RoDeSDDMM_n32(c_sm.n_segs, c_sm.n_segs_residue, n, k,
                          c_sm.seg_row_indices, c_sm.seg_row_indices_residue, c_sm.seg_st_offsets,
                          c_sm.RowOffsets(), c_sm.ColumnIndices(), c_sm.Values(),
                          d_B1.Values(), d_B2.Values(), d_C2, stream1, stream2);

        cudaEventRecord(event2, 0);

        cudaEventSynchronize(event1);
        cudaEventSynchronize(event2);
        cudaEventElapsedTime(&tot_ms, event1, event2);
        cudaDeviceSynchronize();

        gflops = (double) ITER * (double) c_sm.Nonzeros() * 2 * k / tot_ms / 1000000;
        printf("[RoDe_gflops : %.2f ]\n", gflops);


#ifdef VALIDATE
        MatrixDiff<<<(size+31)/32,32>>>(size,diff,d_C,d_C1);
        MatrixDiff<<<(size+31)/32,32>>>(size,diff,d_C,d_C2);
        // MatrixDiff<<<(size+31)/32,32>>>(size,diff,d_C,d_C3);
        // MatrixDiff<<<(size+31)/32,32>>>(size,diff,d_C,d_C4);
        // MatrixDiff<<<(size+31)/32,32>>>(size,diff,d_C,d_C5);
#endif

        cudaFree(d_C);
        cudaFree(d_C1);
        cudaFree(d_C2);
        cudaFree(d_C3);
        cudaFree(d_C4);
        cudaFree(d_C5);
        cudaFree(diff);
    }

    if (k == 128) {
        const int SEG_LENGTH = 32;
        const int BN = 128;

        cudaStream_t stream1, stream2;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);

        double gflops = 0.0f;

        SPC::SparseMatrix sm1(file_path, SPC::SORTED, 1);


        sm1.RowDivide2Segment(SEG_LENGTH, 4, 32);

        // printf("NR1 : %d , NR2 : %d\n",sm1.n_segs,sm1.n_segs_residue);

        SPC::CudaSparseMatrix<float> c_sm(sm1);
        int m = c_sm.Rows(), n = c_sm.Columns();

        absl::BitGen bitgen;
        SPC::CudaMatrix<float> d_B1(m, k, &bitgen);
        SPC::CudaMatrix<float> d_B2(n, k, &bitgen);

        int size = c_sm.Nonzeros();

        float *d_C;
        cudaMalloc((void **) &d_C, sizeof(float) * size);

        float *d_C1;
        cudaMalloc((void **) &d_C1, sizeof(float) * size);

        float *d_C2;
        cudaMalloc((void **) &d_C2, sizeof(float) * size);

        float *d_C3;
        cudaMalloc((void **) &d_C3, size * sizeof(float));

        float *d_C4;
        cudaMalloc((void **) &d_C4, size * sizeof(float));

        float *d_C5;
        cudaMalloc((void **) &d_C5, size * sizeof(float));

        float *diff;
        cudaMalloc((void **) &diff, sizeof(float) * 1);

        FillValues<<<(size + 31) / 32, 32>>>(size, d_C, 0.0f);
        FillValues<<<(size + 31) / 32, 32>>>(size, d_C1, 0.0f);
        FillValues<<<(size + 31) / 32, 32>>>(size, d_C2, 0.0f);
        // FillValues<<<(size+31)/32,32>>>(size,d_C3,0.0f);
        // FillValues<<<(size+31)/32,32>>>(size,d_C4,0.0f);

#ifdef VALIDATE
        StandCall(m,n,k,
                c_sm.RowIndices(),c_sm.RowOffsets(),c_sm.ColumnIndices(),c_sm.Values(),
                d_B1.Values(),d_B2.Values(),d_C);
#endif

        float tot_ms;
        cudaEvent_t event1, event2;
        cudaEventCreate(&event1);
        cudaEventCreate(&event2);

        cudaDeviceSynchronize();
        cudaEventRecord(event1, 0);

        for (int i = 0; i < ITER; ++i)
            RoDeSDDMM_n128(c_sm.n_segs, c_sm.n_segs_residue, n, k,
                           c_sm.seg_row_indices, c_sm.seg_row_indices_residue, c_sm.seg_st_offsets,
                           c_sm.RowOffsets(), c_sm.ColumnIndices(), c_sm.Values(),
                           d_B1.Values(), d_B2.Values(), d_C2, stream1, stream2);

        cudaEventRecord(event2, 0);

        cudaEventSynchronize(event1);
        cudaEventSynchronize(event2);
        cudaEventElapsedTime(&tot_ms, event1, event2);
        cudaDeviceSynchronize();

        gflops = (double) ITER * (double) c_sm.Nonzeros() * 2 * k / tot_ms / 1000000;
        printf("[RoDe_gflops : %.2f ]\n", gflops);


#ifdef VALIDATE
        MatrixDiff<<<(size+31)/32,32>>>(size,diff,d_C,d_C1);
        MatrixDiff<<<(size+31)/32,32>>>(size,diff,d_C,d_C2);
        // MatrixDiff<<<(size+31)/32,32>>>(size,diff,d_C,d_C3);
        // MatrixDiff<<<(size+31)/32,32>>>(size,diff,d_C,d_C4);
        // MatrixDiff<<<(size+31)/32,32>>>(size,diff,d_C,d_C5);
#endif

        cudaFree(d_C);
        cudaFree(d_C1);
        cudaFree(d_C2);
        cudaFree(d_C3);
        cudaFree(d_C4);
        cudaFree(d_C5);
        cudaFree(diff);
    }


    return 0;
}