#include "matrices.h"
#include "definitions.h"
#include "helper_cuda.h"
#ifndef REORDER_GPU_H
#define REORDER_GPU_H
#define WARP_SIZE 32

void warmup_gpu(void);
void preprocess_alloc(const CSR &mat, int **Encodings_gpu, int **Dispersions_gpu, intT **rowptr_gpu, intT **colidx_gpu, int num_blocks_per_row);
void perform_preprocessing(const CSR &mat,
                           int *Encodings_gpu, intT *Dispersions, intT *Dispersions_gpu,
                           intT *rowptr_gpu, intT *colidx_gpu,
                           int num_blocks_per_row, intT block_size);
void preprocess_release(int *Encodings_gpu, intT *Dispersions_gpu, intT *rowptr, intT *colidx_gpu);
std::vector<intT> get_permutation_gpu(const CSR &mat, std::vector<intT> ascending_idx, int *Encodings, intT *Dispersions, intT num_blocks_per_row, float alpha, intT &cluster_cnt);
#endif