#include "reorder.h"
#include "reorder_gpu.cuh"
using namespace std::chrono;

vector<intT> merge_rows(vector<intT> A, vector<intT> B)
{
    vector<intT> result(A.size());

    for (int i = 0; i < A.size(); i++)
    {
        result[i] = A[i] + B[i];
    }
    return result;
}

vector<intT> reorder(CSR &lhs, intT method, float alpha, intT block_size, intT n_repetition, LOGGER &logger)
{
    vector<intT> row_permutation;
    vector<float> reordering_times(n_repetition);
    vector<intT> cluster_cnts(n_repetition);

    float reordering_time;
    intT cluster_cnt;
    Reordering alg = static_cast<Reordering>(method);
    switch (alg)
    {
    case none:
        row_permutation.resize(lhs.rows);
        iota(row_permutation.begin(), row_permutation.end(), 0);
        break;
    case cpu:
        for (int i = 0; i < n_repetition; i++)
        {
            row_permutation = bsa_reordering_cpu(lhs, logger, alpha, block_size, reordering_time);
            // printf("[%d]: elapsed_time: %f\n", i, reordering_time);
            reordering_times[i] = reordering_time;
        }
        break;
    case gpu:
        warmup_gpu();
        for (int i = 0; i < n_repetition; i++)
        {
            row_permutation = bsa_reordering_gpu(lhs, alpha, block_size, reordering_time, cluster_cnt);
            cluster_cnts[i] = cluster_cnt;
            // printf("cluster cnt[%d]: %d\n", i, cluster_cnts[i]);
            // printf("[%d]: elapsed_time: %f\n", i, reordering_time);
            reordering_times[i] = reordering_time;
        }
        logger.cluster_cnt = avg(cluster_cnts);

        break;
    default:
        printf("2 options are supported 0) None 1) CPU 2) GPU\n");
        exit(-1);
        break;
    }
    logger.avg_reordering_time = avg(reordering_times);
    return row_permutation;
}
vector<intT> bsa_reordering_cpu(CSR &lhs, LOGGER &logger, float alpha, intT block_size, float &reordering_time)
{
    intT rows = lhs.rows;
    vector<intT> row_permutation;
    priority_queue<pair<float, intT>> row_queue;
    priority_queue<pair<float, intT>> inner_queue;
    vector<vector<intT>> patterns(rows, vector<intT>((rows + block_size - 1) / block_size));

    auto start = high_resolution_clock::now();
    for (intT r = 0; r < rows; r++)
    {
        set<intT> dense_partition;
        intT score = 0;
        intT start_pos = lhs.rowptr[r];
        intT end_pos = lhs.rowptr[r + 1];
        intT nnz = end_pos - start_pos;
        if (nnz == 0)
        {
            row_permutation.push_back(r);
            continue;
        }

        for (intT nz = start_pos; nz < end_pos; nz++)
        {
            intT col = lhs.colidx[nz];
            patterns[r][col / block_size]++;
            dense_partition.insert(col / block_size);
        }

        for (intT t = 0; t < patterns[r].size(); t++)
        {
            if (patterns[r][t])
            {
                score += block_size - patterns[r][t];
            }
        }

        row_queue.push(make_pair(-1 * (score + (float)dense_partition.size() * nnz), -1 * r));
    }

    // usleep(100000);
    intT cluster_cnt = 0;
    while (!row_queue.empty())
    {
        intT current_group_size = 1;
        intT i = -1 * row_queue.top().second;
        row_queue.pop();
        cluster_cnt++;

        row_permutation.push_back(i);

        vector<intT> pattern = patterns[i];
        intT j;
        while (!row_queue.empty())
        {
            auto j_pair = row_queue.top();
            j = -1 * j_pair.second;

            row_queue.pop();

            vector<intT> B_pattern = patterns[j];

            float sim = normalized_weighted_jaccard_sim(pattern, B_pattern, current_group_size, block_size);

            if (sim <= alpha)
            {
                inner_queue.push(j_pair);
            }
            else
            {
                row_permutation.push_back(j);
                pattern = merge_rows(pattern, B_pattern);
                current_group_size++;
            }
        }

        inner_queue.swap(row_queue);
    }

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    reordering_time = duration.count();
    logger.cluster_cnt = cluster_cnt;
    cout << reordering_time << "ms" << endl;

    return row_permutation;
}
vector<intT> bsa_reordering_gpu(CSR &lhs, float alpha, intT block_size, float &reordering_time, intT &cluster_cnt)
{

    vector<intT> row_permutation;
    // int num_blocks_per_row = (lhs.cols + block_size - 1) / block_size;
    int num_blocks_per_row = ceil((float)lhs.cols / (float)block_size);

    auto start = high_resolution_clock::now();
    /*prepare resources -start*/
    intT *Dispersions = (intT *)malloc(sizeof(int) * lhs.rows);
    int *Encodings_gpu;
    intT *Dispersions_gpu;
    intT *rowptr_gpu;
    intT *colidx_gpu;

    preprocess_alloc(lhs, &Encodings_gpu, &Dispersions_gpu, &rowptr_gpu, &colidx_gpu, num_blocks_per_row);
    /*prepare resources -done*/

    /*Preprocessing: calculate Encodings and dispersions -start*/
    perform_preprocessing(lhs, Encodings_gpu, Dispersions, Dispersions_gpu, rowptr_gpu, colidx_gpu, num_blocks_per_row, block_size);
    /*Preprocessing: calculate Encodings and dispersions -done*/

    /*Prepare Clustering -start*/
    vector<intT> ascending(lhs.rows);
    iota(ascending.begin(), ascending.end(), 0); // ascending = {0, 1, 2, 3, ... lhs.rows-1}
    stable_sort(ascending.begin(), ascending.end(), [Dispersions](size_t i, size_t j)
                { return Dispersions[i] < Dispersions[j]; });
    /*Prepare Clustering -done*/

    /*Perform BSA-reordering via gpu -start*/
    row_permutation = get_permutation_gpu(lhs, ascending, Encodings_gpu, Dispersions, num_blocks_per_row, alpha, cluster_cnt);
    /*Perform BSA-reordering via gpu -done*/

    /* release resources -start*/

    preprocess_release(Encodings_gpu, Dispersions_gpu, rowptr_gpu, colidx_gpu);
    /* release resources -done*/

    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    reordering_time = duration.count();
    // cout << reordering_time << "ms" << endl;

    return row_permutation;
}