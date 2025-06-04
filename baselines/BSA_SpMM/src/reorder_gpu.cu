#include "reorder_gpu.cuh"

// Miscellaneous Functions
static __global__ void warm_up_gpu_kernel()
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid;
}
void warmup_gpu(void)
{
    for (int i = 0; i < 10; i++)
    {
        warm_up_gpu_kernel<<<1, 32>>>();
    }
    CHECK_CUDA(cudaDeviceSynchronize());
}

template <typename T>
static __inline__ __device__ T warp_reduce_sum(T value)
{
    /* aggregate all value that each thread within a warp holding.*/
    T ret = value;

    for (int w = 1; w < warpSize; w = w << 1)
    {
        T tmp = __shfl_xor_sync(0xffffffff, ret, w);
        ret += tmp;
    }
    return ret;
}
template <typename T>
static __inline__ __device__ T reduce_sum(T value, T *shm)
{
    unsigned int stride;
    unsigned int tid = threadIdx.x;
    T tmp = warp_reduce_sum(value); // perform warp shuffle first for less utilized shared memory

    unsigned int block_warp_id = tid / warpSize;
    unsigned int lane = tid % warpSize;
    if (lane == 0)
        shm[block_warp_id] = tmp;
    __syncthreads();
    for (stride = blockDim.x / (2 * warpSize); stride >= 1; stride = stride >> 1)
    {
        if (block_warp_id < stride && lane == 0)
        {
            shm[block_warp_id] += shm[block_warp_id + stride];
        }

        __syncthreads();
    }
    return shm[0];
}

static __global__ void calculate_dispersion_kernel(intT *colidx, intT *rowptr,
                                                   int *weighted_partitions, intT *dispersion_score,
                                                   int num_blocks_per_row, intT col_block_size)
{
    extern __shared__ intT shm[];
    __shared__ intT *encoding;
    __shared__ intT *local_result;
    encoding = (intT *)&shm[0];
    local_result = (intT *)&shm[num_blocks_per_row];
    int row_in_charge = blockIdx.x;
    int row_start = rowptr[row_in_charge];
    int row_nz_count = rowptr[row_in_charge + 1] - row_start;
    // if (row_nz_count == 0)
    //     return;

    for (int i = threadIdx.x; i < num_blocks_per_row; i += blockDim.x)
    {
        encoding[i] = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < row_nz_count; i += blockDim.x)
    {
        intT col_idx = colidx[row_start + i];
        atomicAdd(&encoding[col_idx / col_block_size], 1);
    }
    __syncthreads();

    int store_offset = row_in_charge * num_blocks_per_row;
    int result_tmp = 0;
    int dense_partition_size = 0;
    for (int i = threadIdx.x; i < num_blocks_per_row; i += blockDim.x)
    {
        intT value = encoding[i];
        weighted_partitions[store_offset + i] = value;
        int is_dense_partition = (value != 0);
        dense_partition_size += is_dense_partition;
        result_tmp += is_dense_partition * (col_block_size - value);
    }
    int result = reduce_sum(result_tmp + row_nz_count * dense_partition_size, local_result);

    if (threadIdx.x == 0)
    {
        dispersion_score[row_in_charge] = result;
    }
    else
        return;
}
void preprocess_alloc(const CSR &mat, int **Encodings_gpu, int **Dispersions_gpu, intT **rowptr_gpu, intT **colidx_gpu, int num_blocks_per_row)
{
    CHECK_CUDA(cudaMalloc(Encodings_gpu, sizeof(int) * num_blocks_per_row * mat.rows));
    CHECK_CUDA(cudaMalloc(Dispersions_gpu, sizeof(intT) * mat.rows));
    CHECK_CUDA(cudaMalloc(rowptr_gpu, sizeof(intT) * (mat.rows + 1)));
    CHECK_CUDA(cudaMalloc(colidx_gpu, sizeof(intT) * (mat.total_nonzeros)));

    CHECK_CUDA(cudaMemset(*Encodings_gpu, 0, sizeof(int) * num_blocks_per_row * mat.rows));
    CHECK_CUDA(cudaMemset(*Dispersions_gpu, 0, sizeof(intT) * mat.rows));

    CHECK_CUDA(cudaMemcpy(*rowptr_gpu,
                          mat.rowptr,
                          sizeof(intT) * (mat.rows + 1),
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(*colidx_gpu,
                          mat.colidx,
                          sizeof(intT) * mat.total_nonzeros,
                          cudaMemcpyHostToDevice));
}

void perform_preprocessing(const CSR &mat,
                           int *Encodings_gpu, intT *Dispersions, intT *Dispersions_gpu,
                           intT *rowptr_gpu, intT *colidx_gpu,
                           int num_blocks_per_row, intT block_size)
{
    int blockdim = WARP_SIZE * 4;
    int grid = mat.rows;

    size_t shm_size = num_blocks_per_row * sizeof(intT) + (blockdim * sizeof(intT) / WARP_SIZE);
    calculate_dispersion_kernel<<<grid, blockdim, shm_size>>>(colidx_gpu, rowptr_gpu,
                                                              Encodings_gpu,
                                                              Dispersions_gpu,
                                                              num_blocks_per_row, block_size);
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemcpy(Dispersions,
                          Dispersions_gpu,
                          mat.rows * sizeof(intT),
                          cudaMemcpyDeviceToHost));
    // // used for validation
    // CHECK_CUDA(cudaMemcpy(weighted_partitions,
    //                       weighted_partitions_gpu,
    //                       mat.rows * num_blocks_per_row * sizeof(int),
    //                       cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
}
void preprocess_release(int *Encodings_gpu, intT *Dispersions_gpu, intT *rowptr, intT *colidx_gpu)
{
    CHECK_CUDA(cudaFree(Encodings_gpu));
    CHECK_CUDA(cudaFree(Dispersions_gpu));
    CHECK_CUDA(cudaFree(rowptr));
    CHECK_CUDA(cudaFree(colidx_gpu));
}

static __device__ void mutex_lock(unsigned int *mutex)
{

    if (threadIdx.x == 0)
    {
        unsigned int ns = 8;
        while (atomicCAS(mutex, 0, 1) == 1)
        {
            __nanosleep(ns);
            if (ns < 256)
            {
                ns *= 2;
            }
        }
    }
    __syncthreads();
}

static __device__ void mutex_unlock(unsigned int *mutex)
{
    if (threadIdx.x == 0)
    {
        atomicExch(mutex, 0);
    }
    __syncthreads();
}

static __device__ float calculate_similarity_norm_weighted_jaccard(intT *encoding_rep, intT *encoding_cmp, intT num_blocks_per_row, intT *scratch, float *float_scratch)
{

    float similarity;
    intT sum_of_squares_rep = 0;
    intT sum_of_squares_cmp = 0;

    for (int i = threadIdx.x; i < num_blocks_per_row; i += blockDim.x)
    {
        intT e_rep_i = encoding_rep[i];
        intT e_cmp_i = encoding_cmp[i];

        sum_of_squares_rep += e_rep_i * e_rep_i;
        sum_of_squares_cmp += e_cmp_i * e_cmp_i;
    }
    sum_of_squares_rep = reduce_sum(sum_of_squares_rep, scratch);
    sum_of_squares_cmp = reduce_sum(sum_of_squares_cmp, scratch);

    if (threadIdx.x == 0)
    {
        scratch[0] = sum_of_squares_rep;
        scratch[1] = sum_of_squares_cmp;
    }
    __syncthreads();
    sum_of_squares_rep = scratch[0];
    sum_of_squares_cmp = scratch[1];

    if (sum_of_squares_rep == 0 && sum_of_squares_cmp == 0)
    {
        return 1.0f;
    }
    else if ((sum_of_squares_rep == 0 || sum_of_squares_cmp == 0))
    {
        return 0.0f;
    }
    __syncthreads();

    float norm_rep = sqrt((float)sum_of_squares_rep);
    float norm_cmp = sqrt((float)sum_of_squares_cmp);
    float min_sum = 0.0f;
    float max_sum = 0.0f;

    for (int i = threadIdx.x; i < num_blocks_per_row; i += blockDim.x)
    {
        float sim_rep = ((float)encoding_rep[i]) / norm_rep;
        float sim_cmp = ((float)encoding_cmp[i]) / norm_cmp;
        min_sum += fminf(sim_rep, sim_cmp);
        max_sum += fmaxf(sim_rep, sim_cmp);
    }
    min_sum = reduce_sum(min_sum, float_scratch);
    max_sum = reduce_sum(max_sum, float_scratch);
    __syncthreads();

    if (threadIdx.x == 0) // only the first warp holds valid values, and use only one thread for simple write
    {
        float sim = min_sum / max_sum;
        float_scratch[0] = sim;
    }
    __syncthreads();
    similarity = float_scratch[0];
    return similarity;
}
static __global__ void bsa_clustering(intT *weighted_partitions, const intT cluster_id, intT *ascending_idx, volatile intT *cluster_ids, intT start_idx, int num_rows, intT num_blocks_per_row, float alpha, size_t shm_size, unsigned int *mutexes, intT *cluster_id_to_launch, intT *start_idx_to_launch)
{
    extern __shared__ intT shm[];
    __shared__ intT *encoding_rep;
    __shared__ intT *scratch;
    __shared__ float *float_scratch;
    encoding_rep = shm;
    scratch = &encoding_rep[num_blocks_per_row];
    float_scratch = (float *)&scratch[blockDim.x / warpSize];

    bool next_cluster_created = false;

    mutex_lock(&mutexes[start_idx]);
    cluster_ids[start_idx] = cluster_id;
    for (int i = threadIdx.x; i < num_blocks_per_row; i += blockDim.x)
    {
        encoding_rep[i] = weighted_partitions[ascending_idx[start_idx] * num_blocks_per_row + i];
    }
    __syncthreads();

    mutex_unlock(&mutexes[start_idx]);
    mutex_lock(&mutexes[start_idx + 1]);
    cluster_id_to_launch[0] = -1;
    start_idx_to_launch[0] = -1;

    for (int idx = start_idx + 1; idx < num_rows; idx++)
    {
        volatile intT cluster_tmp = cluster_ids[idx];
        if (cluster_tmp != -1)
        {
            if (idx < num_rows - 1)
            {
                mutex_lock(&mutexes[idx + 1]);
            }
            mutex_unlock(&mutexes[idx]);
            continue;
        }

        intT row = ascending_idx[idx]; // ascending_idx[idx];
        intT *encoding_cmp = &weighted_partitions[row * num_blocks_per_row];
        float similarity;

        similarity = calculate_similarity_norm_weighted_jaccard(encoding_rep, encoding_cmp, num_blocks_per_row, scratch, float_scratch);

        if (threadIdx.x == 0)
        {
            float_scratch[0] = similarity;
        }

        __syncthreads();
        similarity = float_scratch[0];

        if (similarity > alpha)
        {

            if (threadIdx.x == 0)
            {
                cluster_ids[idx] = cluster_id;
            }

            for (intT i = threadIdx.x; i < num_blocks_per_row; i += blockDim.x)
            {
                encoding_rep[i] += encoding_cmp[i];
            }

            __syncthreads();
        }
        else
        {
            if (!next_cluster_created)
            {
                if (threadIdx.x == 0)
                {

                    bsa_clustering<<<1, blockDim.x, shm_size, cudaStreamFireAndForget>>>(weighted_partitions,
                                                                                         cluster_id + 1,
                                                                                         ascending_idx, cluster_ids, idx,
                                                                                         num_rows, num_blocks_per_row, alpha, shm_size,
                                                                                         mutexes, cluster_id_to_launch, start_idx_to_launch);

                    cudaError_t err = cudaGetLastError();
                    scratch[0] = (int)cudaGetLastError();
                    if (err == cudaErrorLaunchPendingCountExceeded)
                    {
                        cluster_id_to_launch[0] = cluster_id + 1;
                        start_idx_to_launch[0] = idx;
                    }
                }
            }

            next_cluster_created = true;
        }

        if (idx < num_rows - 1)
        {
            mutex_lock(&mutexes[idx + 1]);
        }
        mutex_unlock(&mutexes[idx]);
    }
}

std::vector<intT> get_permutation_gpu(const CSR &mat, std::vector<intT> ascending_idx, int *Encodings, intT *Dispersions, intT num_blocks_per_row, float alpha, intT &cluster_cnt)
{
    intT *ascending_idx_head = &ascending_idx[0];
    intT *ascending_idx_gpu;
    intT *cluster_ids, *cluster_ids_gpu;
    unsigned int *mutexes;
    intT *cluster_id_to_launch, *start_idx_to_launch;

    CHECK_CUDA(cudaMalloc((void **)&ascending_idx_gpu, sizeof(intT) * mat.rows));
    CHECK_CUDA(cudaMalloc((void **)&cluster_ids_gpu, sizeof(intT) * mat.rows));
    CHECK_CUDA(cudaMalloc((void **)&mutexes, sizeof(unsigned int) * mat.rows));

    CHECK_CUDA(cudaMallocHost((void **)&cluster_id_to_launch, sizeof(intT), cudaHostAllocMapped));
    CHECK_CUDA(cudaMallocHost((void **)&start_idx_to_launch, sizeof(intT), cudaHostAllocMapped));

    CHECK_CUDA(cudaMemset(mutexes, 0, sizeof(unsigned int) * mat.rows));
    CHECK_CUDA(cudaMemcpy(ascending_idx_gpu, ascending_idx_head, sizeof(intT) * mat.rows, cudaMemcpyHostToDevice));
    cluster_ids = (intT *)malloc(sizeof(intT) * mat.rows);
    CHECK_CUDA(cudaDeviceSynchronize());

    int blockdim;
    if (num_blocks_per_row < 32)
    {
        blockdim = 32;
    }
    else
    {
        int num_scan_iterate = 4;
        int blockdim_candidate = WARP_SIZE * ceil((float)(num_blocks_per_row / num_scan_iterate) / (float)WARP_SIZE);
        blockdim_candidate = blockdim_candidate > 32 ? blockdim_candidate : 32;
        blockdim = 1024 < blockdim_candidate ? 1024 : blockdim_candidate;
    }
    // blockdim = 1024;

    int grid = 1;

    size_t shm_size = (blockdim * sizeof(intT) + blockdim * sizeof(float)) / WARP_SIZE + sizeof(intT) * num_blocks_per_row;

    cudaStream_t initial_stream;
    cudaStreamCreateWithFlags(&initial_stream, cudaStreamNonBlocking);

    memset(cluster_ids, -1, sizeof(intT) * mat.rows);
    int zero_row_idx = 0;

    while (true)
    {
        if (zero_row_idx == mat.rows)
            break;
        if (Dispersions[ascending_idx_head[zero_row_idx]] == 0)
        {
            // printf("%d is zero row next row = %d\n", ascending_idx[zero_row_idx], ascending_idx[zero_row_idx + 1]);
            cluster_ids[zero_row_idx] = 0;
            zero_row_idx++;
        }
        else
            break;
    }
    CHECK_CUDA(cudaMemcpy(cluster_ids_gpu, cluster_ids, sizeof(intT) * mat.rows, cudaMemcpyHostToDevice));

    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32768);

    size_t limit;
    int exponent = 2;
    cudaDeviceGetLimit(&limit, cudaLimitDevRuntimePendingLaunchCount);

    cluster_id_to_launch[0] = 1;
    start_idx_to_launch[0] = zero_row_idx;

    do
    {
        bsa_clustering<<<grid, blockdim, shm_size, initial_stream>>>(Encodings, cluster_id_to_launch[0],
                                                                     ascending_idx_gpu, cluster_ids_gpu,
                                                                     start_idx_to_launch[0], mat.rows, num_blocks_per_row,
                                                                     alpha, shm_size, mutexes,
                                                                     cluster_id_to_launch, start_idx_to_launch);

        CHECK_CUDA(cudaDeviceSynchronize());
        limit = limit * exponent;
        if (cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, limit) != cudaSuccess)
        {
            limit = limit / 2;
            exponent = 1;
            cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, limit);
        }

    } while (cluster_id_to_launch[0] != -1);

    CHECK_CUDA(cudaMemcpy(cluster_ids, cluster_ids_gpu, sizeof(intT) * mat.rows, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());

    auto compare_by_cluster_id = [&cluster_ids](int i, int j)
    {
        return cluster_ids[i] < cluster_ids[j];
    };
    std::vector<intT> indices(mat.rows);
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), compare_by_cluster_id);
    std::vector<intT> permutation(mat.rows);
    for (int i = 0; i < mat.rows; i++)
    {
        permutation[i] = ascending_idx_head[indices[i]];
    }
    cluster_cnt = cluster_ids[indices[mat.rows - 1]] + (int)(zero_row_idx != 0);
    // cluster_cnt = cluster_ids[mat.rows - 1];

    cudaStreamDestroy(initial_stream);

    CHECK_CUDA(cudaFree(mutexes));
    CHECK_CUDA(cudaFree(ascending_idx_gpu));
    CHECK_CUDA(cudaFree(cluster_ids_gpu));
    CHECK_CUDA(cudaFreeHost(cluster_id_to_launch));
    CHECK_CUDA(cudaFreeHost(start_idx_to_launch));

    free(cluster_ids);
    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 2048);

    return permutation;
}
