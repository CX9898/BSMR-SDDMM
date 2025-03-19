#include <cmath>
#include <omp.h>
#include <numeric>
#include <algorithm>
#include <set>
#include <queue>

#include "ReBELL.hpp"
#include "parallelAlgorithm.cuh"
#include "CudaTimeCalculator.cuh"

#define COL_BLOCK_SIZE 32

void noReorderRow(const sparseMatrix::CSR<float> &matrix, std::vector<UIN> &reorderedRows, float &time) {
    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();
    reorderedRows.resize(matrix.row());
    iota(reorderedRows.begin(), reorderedRows.end(), 0);

    std::vector<UIN> numColIndices(matrix.row());
#pragma omp parallel for
    for (int row = 0; row < matrix.row(); ++row) {
        numColIndices[row] = matrix.rowOffsets()[row + 1] - matrix.rowOffsets()[row];
    }

    host::sort_by_key(numColIndices.data(), numColIndices.data() + numColIndices.size(),
                      reorderedRows.data());

    // Remove zero rows
    {
        UIN startIndexOfNonZeroRow = 0;
        while (startIndexOfNonZeroRow < reorderedRows.size()
            && matrix.rowOffsets()[reorderedRows[startIndexOfNonZeroRow] + 1]
                - matrix.rowOffsets()[reorderedRows[startIndexOfNonZeroRow]] == 0) {
            ++startIndexOfNonZeroRow;
        }
        reorderedRows.erase(reorderedRows.begin(), reorderedRows.begin() + startIndexOfNonZeroRow);
    }
    timeCalculator.endClock();
    time = timeCalculator.getTime();
}

void encoding(const sparseMatrix::CSR<float> &matrix, std::vector<std::vector<UIN>> &encodings) {
    const int colBlock = std::ceil(static_cast<float>(matrix.col()) / COL_BLOCK_SIZE);
    encodings.resize(matrix.row());
#pragma omp parallel for schedule(dynamic)
    for (int row = 0; row < matrix.row(); ++row) {
        encodings[row].resize(colBlock);
        for (int idx = matrix.rowOffsets()[row]; idx < matrix.rowOffsets()[row + 1]; ++idx) {
            const int col = matrix.colIndices()[idx];
            ++encodings[row][col / COL_BLOCK_SIZE];
        }
    }
}

void calculateDispersion(const UIN col,
                         const std::vector<std::vector<UIN>> &encodings,
                         std::vector<UIN> &dispersions) {
#pragma omp parallel for schedule(dynamic)
    for (int row = 0; row < encodings.size(); ++row) {
        UIN numOfNonZeroColBlocks = 0;
        UIN zeroFillings = 0;
        for (int colBlockIdx = 0; colBlockIdx < encodings[row].size(); ++colBlockIdx) {
            const UIN numOfNonZeroCols = encodings[row][colBlockIdx];
            if (numOfNonZeroCols != 0) {
                ++numOfNonZeroColBlocks;
                zeroFillings += BLOCK_COL_SIZE - numOfNonZeroCols;
            }
        }
        dispersions[row] = COL_BLOCK_SIZE * numOfNonZeroColBlocks + zeroFillings;
    }
}

// return similarity between two encodings
float clusterComparison(const std::vector<UIN> &encoding_rep, const std::vector<UIN> &encoding_cmp) {
    UIN sum_of_squares_rep = 0;
    UIN sum_of_squares_cmp = 0;
    for (int idx = 0; idx < encoding_rep.size(); ++idx) {
        sum_of_squares_rep += encoding_rep[idx] * encoding_rep[idx];
        sum_of_squares_cmp += encoding_cmp[idx] * encoding_cmp[idx];
    }
    if (sum_of_squares_rep == 0 && sum_of_squares_cmp == 0) {
        return 1.0f;
    } else if ((sum_of_squares_rep == 0 || sum_of_squares_cmp == 0)) {
        return 0.0f;
    }
    float norm_rep = sqrt((float) sum_of_squares_rep);
    float norm_cmp = sqrt((float) sum_of_squares_cmp);
    float min_sum = 0.0f;
    float max_sum = 0.0f;
    for (int idx = 0; idx < encoding_rep.size(); ++idx) {
        float sim_rep = (float) encoding_rep[idx] / norm_rep;
        float sim_cmp = (float) encoding_cmp[idx] / norm_cmp;
        min_sum += fminf(sim_rep, sim_cmp);
        max_sum += fmaxf(sim_rep, sim_cmp);
    }
    return min_sum / max_sum;
}

void clustering(const std::vector<std::vector<UIN>> &encodings,
                const std::vector<UIN> &rows, const UIN startIndexOfNonZeroRow, std::vector<int> &clusterIds) {

//    UIN num = 0;
    for (int idx = startIndexOfNonZeroRow; idx < encodings.size() - 1; ++idx) {
        if (idx > startIndexOfNonZeroRow && clusterIds[rows[idx]] != -1) {
            continue;
        }
        clusterIds[rows[idx]] = idx;
#pragma omp parallel for schedule(dynamic)
        for (int cmpIdx = idx + 1; cmpIdx < encodings.size(); ++cmpIdx) {
            if (clusterIds[rows[cmpIdx]] != -1) {
                continue;
            }
            const float similarity =
                clusterComparison(encodings[rows[startIndexOfNonZeroRow]], encodings[rows[cmpIdx]]);
            if (similarity > row_similarity_threshold_alpha) {
                clusterIds[rows[cmpIdx]] = clusterIds[rows[idx]];
//                ++num;
            }
        }
    }
//    printf("!!! num = %d\n", num);
}

void rowReordering_cpu(const sparseMatrix::CSR<float> &matrix, std::vector<UIN> &rows, float &time) {

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();

    std::vector<std::vector<UIN>> encodings;
    encoding(matrix, encodings);

    std::vector<UIN> dispersions(matrix.row());
    calculateDispersion(matrix.col(), encodings, dispersions);

    std::vector<UIN> ascendingRow(matrix.row()); // Store the original row id
    std::iota(ascendingRow.begin(), ascendingRow.end(), 0); // ascending = {0, 1, 2, 3, ... rows-1}
    std::stable_sort(ascendingRow.begin(),
                     ascendingRow.end(),
                     [&dispersions](size_t i, size_t j) { return dispersions[i] < dispersions[j]; });

    std::vector<int> clusterIds(matrix.row(), -1);
    UIN startIndexOfNonZeroRow = 0;
    while (startIndexOfNonZeroRow < matrix.row() && dispersions[ascendingRow[startIndexOfNonZeroRow]] == 0) {
        clusterIds[ascendingRow[startIndexOfNonZeroRow]] = 0;
        ++startIndexOfNonZeroRow;
    }

    clustering(encodings, ascendingRow, startIndexOfNonZeroRow, clusterIds);

    rows.resize(matrix.row());
    std::iota(rows.begin(),
              rows.end(),
              0); // rowIndices = {0, 1, 2, 3, ... rows-1}
    std::stable_sort(rows.begin(),
                     rows.end(),
                     [&clusterIds](int i, int j) { return clusterIds[i] < clusterIds[j]; });

    // Remove zero rows
    {
        startIndexOfNonZeroRow = 0;
        while (startIndexOfNonZeroRow < matrix.row()
            && matrix.rowOffsets()[rows[startIndexOfNonZeroRow] + 1]
                - matrix.rowOffsets()[rows[startIndexOfNonZeroRow]] == 0) {
            ++startIndexOfNonZeroRow;
        }
        rows.erase(rows.begin(), rows.begin() + startIndexOfNonZeroRow);
    }

    timeCalculator.endClock();
    time = timeCalculator.getTime();
}

float normalized_weighted_jaccard_sim(std::vector<int> A_pattern,
                                      std::vector<int> B_pattern,
                                      int current_group_size,
                                      int block_size) {
    float score_A = 0.0;
    float score_B = 0.0;

    float sim_A = 0.0;
    float sim_B = 0.0;

    float min_sum = 0.0;
    float max_sum = 0.0;

    for (int i = 0; i < B_pattern.size(); i++) {
        score_A += A_pattern[i] * A_pattern[i];
        score_B += B_pattern[i] * B_pattern[i];
    }

    score_A = sqrt(score_A);
    score_B = sqrt(score_B);

    if (score_A == 0 && score_B == 0)
        return 1.0;
    if (score_A == 0 || score_B == 0)
        return 0.0;

    for (int i = 0; i < B_pattern.size(); i++) {
        if (A_pattern[i] == 0 && B_pattern[i] == 0) {
            continue;
        }

        sim_A = A_pattern[i] / score_A;
        sim_B = B_pattern[i] / score_B;

        min_sum += std::min(sim_A, sim_B);
        max_sum += std::max(sim_A, sim_B);
    }

    return (min_sum / max_sum);
}

std::vector<int> merge_rows(std::vector<int> A, std::vector<int> B) {
    std::vector<int> result(A.size());

    for (int i = 0; i < A.size(); i++) {
        result[i] = A[i] + B[i];
    }
    return result;
}

std::vector<int> bsa_rowReordering_cpu(const sparseMatrix::CSR<float> &matrix,
                                       const float similarity_threshold_alpha,
                                       const int block_size,
                                       float &reordering_time) {
    int rows = matrix.row();
    std::vector<int> row_permutation;
    std::priority_queue<std::pair<float, int>> row_queue;
    std::priority_queue<std::pair<float, int>> inner_queue;
    std::vector<std::vector<int>> patterns(rows, std::vector<int>((rows + block_size - 1) / block_size));

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();
    for (int r = 0; r < rows; r++) {
        std::set<int> dense_partition;
        int score = 0;
        int start_pos = matrix.rowOffsets()[r];
        int end_pos = matrix.rowOffsets()[r + 1];
        int nnz = end_pos - start_pos;
        if (nnz == 0) {
            row_permutation.push_back(r);
            continue;
        }

        for (int nz = start_pos; nz < end_pos; nz++) {
            int col = matrix.colIndices()[nz];
            patterns[r][col / block_size]++;
            dense_partition.insert(col / block_size);
        }

        for (int t = 0; t < patterns[r].size(); t++) {
            if (patterns[r][t]) {
                score += block_size - patterns[r][t];
            }
        }

        row_queue.push(std::make_pair(-1 * (score + (float) dense_partition.size() * nnz), -1 * r));
    }

    // usleep(100000);
    int cluster_cnt = 0;
    while (!row_queue.empty()) {
        int current_group_size = 1;
        int i = -1 * row_queue.top().second;
        row_queue.pop();
        cluster_cnt++;

        row_permutation.push_back(i);

        std::vector<int> pattern = patterns[i];
        int j;
        while (!row_queue.empty()) {
            auto j_pair = row_queue.top();
            j = -1 * j_pair.second;

            row_queue.pop();

            std::vector<int> B_pattern = patterns[j];

            float sim = normalized_weighted_jaccard_sim(pattern, B_pattern, current_group_size, block_size);

            if (sim <= similarity_threshold_alpha) {
                inner_queue.push(j_pair);
            } else {
                row_permutation.push_back(j);
                pattern = merge_rows(pattern, B_pattern);
                current_group_size++;
            }
        }

        inner_queue.swap(row_queue);
    }

    // Remove zero rows
    {
        UIN startIndexOfNonZeroRow = 0;
        while (startIndexOfNonZeroRow < row_permutation.size()
            && matrix.rowOffsets()[row_permutation[startIndexOfNonZeroRow] + 1]
                - matrix.rowOffsets()[row_permutation[startIndexOfNonZeroRow]] == 0) {
            ++startIndexOfNonZeroRow;
        }
        row_permutation.erase(row_permutation.begin(), row_permutation.begin() + startIndexOfNonZeroRow);
    }

    timeCalculator.endClock();
    reordering_time = timeCalculator.getTime();
    std::cout << reordering_time << "ms" << std::endl;

    return row_permutation;
}

namespace kernel {

template<typename T>
static __inline__ __device__ T warp_reduce_sum(T value) {
    /* aggregate all value that each thread within a warp holding.*/
    T ret = value;

    for (int w = 1; w < warpSize; w = w << 1) {
        T tmp = __shfl_xor_sync(0xffffffff, ret, w);
        ret += tmp;
    }
    return ret;
}

template<typename T>
static __inline__ __device__ T reduce_sum(T value, T *shm) {
    unsigned int stride;
    unsigned int tid = threadIdx.x;
    T tmp = warp_reduce_sum(value); // perform warp shuffle first for less utilized shared memory

    unsigned int block_warp_id = tid / warpSize;
    unsigned int lane = tid % warpSize;
    if (lane == 0)
        shm[block_warp_id] = tmp;
    __syncthreads();
    for (stride = blockDim.x / (2 * warpSize); stride >= 1; stride = stride >> 1) {
        if (block_warp_id < stride && lane == 0) {
            shm[block_warp_id] += shm[block_warp_id + stride];
        }

        __syncthreads();
    }
    return shm[0];
}

__global__ void calculateDispersion(const UIN *colidx, const UIN *rowptr,
                                    int *weighted_partitions, int *dispersion_score,
                                    int num_blocks_per_row, int col_block_size) {
    extern __shared__ int shm[];
    __shared__ int *encoding;
    __shared__ int *local_result;
    encoding = (int *) &shm[0];
    local_result = (int *) &shm[num_blocks_per_row];
    int row_in_charge = blockIdx.x;
    int row_start = rowptr[row_in_charge];
    int row_nz_count = rowptr[row_in_charge + 1] - row_start;
    // if (row_nz_count == 0)
    //     return;

    for (int i = threadIdx.x; i < num_blocks_per_row; i += blockDim.x) {
        encoding[i] = 0;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < row_nz_count; i += blockDim.x) {
        int col_idx = colidx[row_start + i];
        atomicAdd(&encoding[col_idx / col_block_size], 1);
    }
    __syncthreads();

    int store_offset = row_in_charge * num_blocks_per_row;
    int result_tmp = 0;
    int dense_partition_size = 0;
    for (int i = threadIdx.x; i < num_blocks_per_row; i += blockDim.x) {
        int value = encoding[i];
        weighted_partitions[store_offset + i] = value;
        int is_dense_partition = (value != 0);
        dense_partition_size += is_dense_partition;
        result_tmp += is_dense_partition * (col_block_size - value);
    }
    int result = reduce_sum(result_tmp + row_nz_count * dense_partition_size, local_result);

    if (threadIdx.x == 0) {
        dispersion_score[row_in_charge] = result;
    } else
        return;
}

static __device__ void mutex_lock(unsigned int *mutex) {

    if (threadIdx.x == 0) {
        unsigned int ns = 8;
        while (atomicCAS(mutex, 0, 1) == 1) {
            __nanosleep(ns);
            if (ns < 256) {
                ns *= 2;
            }
        }
    }
    __syncthreads();
}

static __device__ void mutex_unlock(unsigned int *mutex) {
    if (threadIdx.x == 0) {
        atomicExch(mutex, 0);
    }
    __syncthreads();
}

static __device__ float calculate_similarity_norm_weighted_jaccard(const int *encoding_rep,
                                                                   const int *encoding_cmp,
                                                                   int num_blocks_per_row,
                                                                   int *scratch,
                                                                   float *float_scratch) {

    float similarity;
    int sum_of_squares_rep = 0;
    int sum_of_squares_cmp = 0;

    for (int i = threadIdx.x; i < num_blocks_per_row; i += blockDim.x) {
        int e_rep_i = encoding_rep[i];
        int e_cmp_i = encoding_cmp[i];

        sum_of_squares_rep += e_rep_i * e_rep_i;
        sum_of_squares_cmp += e_cmp_i * e_cmp_i;
    }
    sum_of_squares_rep = reduce_sum(sum_of_squares_rep, scratch);
    sum_of_squares_cmp = reduce_sum(sum_of_squares_cmp, scratch);

    if (threadIdx.x == 0) {
        scratch[0] = sum_of_squares_rep;
        scratch[1] = sum_of_squares_cmp;
    }
    __syncthreads();
    sum_of_squares_rep = scratch[0];
    sum_of_squares_cmp = scratch[1];

    if (sum_of_squares_rep == 0 && sum_of_squares_cmp == 0) {
        return 1.0f;
    } else if ((sum_of_squares_rep == 0 || sum_of_squares_cmp == 0)) {
        return 0.0f;
    }
    __syncthreads();

    float norm_rep = sqrt((float) sum_of_squares_rep);
    float norm_cmp = sqrt((float) sum_of_squares_cmp);
    float min_sum = 0.0f;
    float max_sum = 0.0f;

    for (int i = threadIdx.x; i < num_blocks_per_row; i += blockDim.x) {
        float sim_rep = ((float) encoding_rep[i]) / norm_rep;
        float sim_cmp = ((float) encoding_cmp[i]) / norm_cmp;
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

static __global__ void bsa_clustering(const int *weighted_partitions,
                                      const int cluster_id,
                                      int *ascending_idx,
                                      volatile int *cluster_ids,
                                      int start_idx,
                                      int num_rows,
                                      int num_blocks_per_row,
                                      float alpha,
                                      size_t shm_size,
                                      unsigned int *mutexes,
                                      int *cluster_id_to_launch,
                                      int *start_idx_to_launch) {
    extern __shared__ int shm[];
    __shared__ int *encoding_rep;
    __shared__ int *scratch;
    __shared__ float *float_scratch;
    encoding_rep = shm;
    scratch = &encoding_rep[num_blocks_per_row];
    float_scratch = (float *) &scratch[blockDim.x / warpSize];

    bool next_cluster_created = false;

    mutex_lock(&mutexes[start_idx]);
    cluster_ids[start_idx] = cluster_id;
    for (int i = threadIdx.x; i < num_blocks_per_row; i += blockDim.x) {
        encoding_rep[i] = weighted_partitions[ascending_idx[start_idx] * num_blocks_per_row + i];
    }
    __syncthreads();

    mutex_unlock(&mutexes[start_idx]);
    mutex_lock(&mutexes[start_idx + 1]);
    cluster_id_to_launch[0] = -1;
    start_idx_to_launch[0] = -1;

    for (int idx = start_idx + 1; idx < num_rows; idx++) {
        volatile int cluster_tmp = cluster_ids[idx];
        if (cluster_tmp != -1) {
            if (idx < num_rows - 1) {
                mutex_lock(&mutexes[idx + 1]);
            }
            mutex_unlock(&mutexes[idx]);
            continue;
        }

        int row = ascending_idx[idx]; // ascending_idx[idx];
        const int *encoding_cmp = &weighted_partitions[row * num_blocks_per_row];
        float similarity;

        similarity = calculate_similarity_norm_weighted_jaccard(encoding_rep,
                                                                encoding_cmp,
                                                                num_blocks_per_row,
                                                                scratch,
                                                                float_scratch);

        if (threadIdx.x == 0) {
            float_scratch[0] = similarity;
        }

        __syncthreads();
        similarity = float_scratch[0];

        if (similarity > alpha) {

            if (threadIdx.x == 0) {
                cluster_ids[idx] = cluster_id;
            }

            for (int i = threadIdx.x; i < num_blocks_per_row; i += blockDim.x) {
                encoding_rep[i] += encoding_cmp[i];
            }

            __syncthreads();
        } else {
            if (!next_cluster_created) {
                if (threadIdx.x == 0) {

                    bsa_clustering<<<1, blockDim.x, shm_size, cudaStreamFireAndForget>>>(weighted_partitions,
                                                                                         cluster_id + 1,
                                                                                         ascending_idx,
                                                                                         cluster_ids,
                                                                                         idx,
                                                                                         num_rows,
                                                                                         num_blocks_per_row,
                                                                                         alpha,
                                                                                         shm_size,
                                                                                         mutexes,
                                                                                         cluster_id_to_launch,
                                                                                         start_idx_to_launch);

                    cudaError_t err = cudaGetLastError();
                    scratch[0] = (int) cudaGetLastError();
                    if (err == cudaErrorLaunchPendingCountExceeded) {
                        cluster_id_to_launch[0] = cluster_id + 1;
                        start_idx_to_launch[0] = idx;
                    }
                }
            }

            next_cluster_created = true;
        }

        if (idx < num_rows - 1) {
            mutex_lock(&mutexes[idx + 1]);
        }
        mutex_unlock(&mutexes[idx]);
    }
}

} // namespace kernel

void calculateDispersion(const sparseMatrix::CSR<float> &matrix,
                         dev::vector<int> &Encodings_gpu,
                         std::vector<int> &Dispersions,
                         dev::vector<int> Dispersions_gpu,
                         const dev::vector<UIN> &rowptr_gpu,
                         const dev::vector<UIN> &colidx_gpu,
                         int num_blocks_per_row,
                         UIN block_size) {
    int blockdim = WARP_SIZE * 4;
    int grid = matrix.row();

    size_t shm_size = num_blocks_per_row * sizeof(UIN) + (blockdim * sizeof(UIN) / WARP_SIZE);
    kernel::calculateDispersion<<<grid, blockdim, shm_size>>>(colidx_gpu.data(), rowptr_gpu.data(),
        Encodings_gpu.data(),
        Dispersions_gpu.data(),
        num_blocks_per_row, block_size);
    cudaDeviceSynchronize();

    Dispersions = d2h(Dispersions_gpu);
}

std::vector<UIN> get_permutation_gpu(const sparseMatrix::CSR<float> &mat,
                                     std::vector<int> ascending_idx,
                                     const dev::vector<int> &Encodings,
                                     const std::vector<int> &Dispersions,
                                     int num_blocks_per_row,
                                     float alpha,
                                     int &cluster_cnt) {

    std::vector<int> cluster_ids(mat.row(), -1);
    dev::vector<unsigned int> mutexes(mat.row(), 0);
    int *cluster_id_to_launch, *start_idx_to_launch;

    cudaMallocHost((void **) &cluster_id_to_launch, sizeof(int), cudaHostAllocMapped);
    cudaMallocHost((void **) &start_idx_to_launch, sizeof(int), cudaHostAllocMapped);

    cudaDeviceSynchronize();

    dev::vector<int> ascending_idx_gpu(ascending_idx);

    int blockdim;
    if (num_blocks_per_row < 32) {
        blockdim = 32;
    } else {
        int num_scan_iterate = 4;
        int blockdim_candidate = WARP_SIZE * ceil((float) (num_blocks_per_row / num_scan_iterate) / (float) WARP_SIZE);
        blockdim_candidate = blockdim_candidate > 32 ? blockdim_candidate : 32;
        blockdim = 1024 < blockdim_candidate ? 1024 : blockdim_candidate;
    }
    // blockdim = 1024;

    int grid = 1;

    size_t
        shm_size = (blockdim * sizeof(int) + blockdim * sizeof(float)) / WARP_SIZE + sizeof(int) * num_blocks_per_row;

    cudaStream_t initial_stream;
    cudaStreamCreateWithFlags(&initial_stream, cudaStreamNonBlocking);

    int zero_row_idx = 0;
    int *ascending_idx_head = &ascending_idx[0];

    while (true) {
        if (zero_row_idx == mat.row())
            break;
        if (Dispersions[ascending_idx_head[zero_row_idx]] == 0) {
            // printf("%d is zero row next row = %d\n", ascending_idx[zero_row_idx], ascending_idx[zero_row_idx + 1]);
            cluster_ids[zero_row_idx] = 0;
            zero_row_idx++;
        } else
            break;
    }

    dev::vector<int> cluster_ids_gpu(cluster_ids);

    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 32768);

    size_t limit;
    int exponent = 2;
    cudaDeviceGetLimit(&limit, cudaLimitDevRuntimePendingLaunchCount);

    cluster_id_to_launch[0] = 1;
    start_idx_to_launch[0] = zero_row_idx;

    do {
        kernel::bsa_clustering<<<grid, blockdim, shm_size, initial_stream>>>(Encodings.data(),
            cluster_id_to_launch[0],
            ascending_idx_gpu.data(),
            cluster_ids_gpu.data(),
            start_idx_to_launch[0],
            mat.row(),
            num_blocks_per_row,
            alpha,
            shm_size,
            mutexes.data(),
            cluster_id_to_launch,
            start_idx_to_launch);

        cudaDeviceSynchronize();
        limit = limit * exponent;
        if (cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, limit) != cudaSuccess) {
            limit = limit / 2;
            exponent = 1;
            cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, limit);
        }

    } while (cluster_id_to_launch[0] != -1);

    cluster_ids = d2h(cluster_ids_gpu);

    auto compare_by_cluster_id = [&cluster_ids](int i, int j) {
      return cluster_ids[i] < cluster_ids[j];
    };
    std::vector<int> indices(mat.row());
    std::iota(indices.begin(), indices.end(), 0);
    std::stable_sort(indices.begin(), indices.end(), compare_by_cluster_id);
    std::vector<UIN> permutation(mat.row());
    for (int i = 0; i < mat.row(); i++) {
        permutation[i] = ascending_idx_head[indices[i]];
    }
    cluster_cnt = cluster_ids[indices[mat.row() - 1]] + (int) (zero_row_idx != 0);
    // cluster_cnt = cluster_ids[mat.row() - 1];

    cudaStreamDestroy(initial_stream);

    cudaFreeHost(cluster_id_to_launch);
    cudaFreeHost(start_idx_to_launch);

    cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 2048);

    return permutation;
}

std::vector<UIN> bsa_rowReordering_gpu(const sparseMatrix::CSR<float> &matrix,
                                       float alpha,
                                       UIN block_size,
                                       float &reordering_time) {

    std::vector<UIN> row_permutation;
    // int num_blocks_per_row = (lhs.cols + block_size - 1) / block_size;
    int num_blocks_per_row = ceil((float) matrix.col() / (float) block_size);

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();

    /*prepare resources -start*/
    std::vector<int> Dispersions;
    dev::vector<int> Encodings_gpu(num_blocks_per_row * matrix.row(), 0);
    dev::vector<int> Dispersions_gpu(matrix.row(), 0);
    dev::vector<UIN> rowptr_gpu(matrix.rowOffsets());
    dev::vector<UIN> colidx_gpu(matrix.colIndices());
    /*prepare resources -done*/

    /*Preprocessing: calculate Encodings and dispersions -start*/
    calculateDispersion(matrix,
                        Encodings_gpu,
                        Dispersions,
                        Dispersions_gpu,
                        rowptr_gpu,
                        colidx_gpu,
                        num_blocks_per_row,
                        block_size);
    /*Preprocessing: calculate Encodings and dispersions -done*/

    /*Prepare Clustering -start*/
    std::vector<int> ascending(matrix.row());
    iota(ascending.begin(), ascending.end(), 0); // ascending = {0, 1, 2, 3, ... lhs.rows-1}
    stable_sort(ascending.begin(),
                ascending.end(),
                [Dispersions](size_t i, size_t j) { return Dispersions[i] < Dispersions[j]; });
    /*Prepare Clustering -done*/

    /*Perform BSA-reordering via gpu -start*/
    int cluster_cnt = 0;
    row_permutation = get_permutation_gpu(matrix,
                                          ascending,
                                          Encodings_gpu,
                                          Dispersions,
                                          num_blocks_per_row,
                                          alpha,
                                          cluster_cnt);
    /*Perform BSA-reordering via gpu -done*/

    // Remove zero rows
    {
        UIN startIndexOfNonZeroRow = 0;
        while (startIndexOfNonZeroRow < row_permutation.size()
            && matrix.rowOffsets()[row_permutation[startIndexOfNonZeroRow] + 1]
                - matrix.rowOffsets()[row_permutation[startIndexOfNonZeroRow]] == 0) {
            ++startIndexOfNonZeroRow;
        }
        row_permutation.erase(row_permutation.begin(), row_permutation.begin() + startIndexOfNonZeroRow);
    }

    timeCalculator.endClock();
    reordering_time = timeCalculator.getTime();
    // cout << reordering_time << "ms" << endl;

    return row_permutation;
}