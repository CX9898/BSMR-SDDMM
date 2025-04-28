#include <numeric>
#include <cmath>
#include <unordered_map>
#include <omp.h>
#include <cuda_runtime.h>

#include "CudaTimeCalculator.cuh"
#include "ReBELL.hpp"
#include "TensorCoreConfig.cuh"
#include "parallelAlgorithm.cuh"
#include "sddmmKernel.cuh"

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

// blockDim:[512,1,1]
__global__ void calculateNumNonZeroColSegmentsPerRowPanel(const UIN numCols,
                                                          const UIN *__restrict__ rowOffsets,
                                                          const UIN *__restrict__ colIndices,
                                                          const UIN numNonZeroRow,
                                                          const UIN *__restrict__ reorderedRows,
                                                          UIN *__restrict__ numNonZeroColSegmentsPerRowPanel) {

    const UIN rowPanelId = blockIdx.x;

    const UIN warpId = threadIdx.x >> 5;

    const UIN indexOfReorderedRows = rowPanelId * ROW_PANEL_SIZE + warpId;
    if (indexOfReorderedRows >= numNonZeroRow) {
        return;
    }

    const UIN row = reorderedRows[indexOfReorderedRows + warpId];
    const UIN endIdx = rowOffsets[row + 1];

    const UIN laneId = threadIdx.x & 31;

    for (int idx = rowOffsets[row] + laneId; idx < endIdx; idx += WARP_SIZE) {
        const UIN col = colIndices[idx];
    }

}

// blockDim:[512,1,1]
__global__ void calculateNNZPerColSegmentPerPanel(const UIN numCols,
                                                  const UIN *__restrict__ rowOffsets,
                                                  const UIN *__restrict__ colIndices,
                                                  const UIN numNonZeroRow,
                                                  const UIN *__restrict__ reorderedRows,
                                                  const UIN *__restrict__ rowPanelColSegmentOffsets,
                                                  UIN *__restrict__ nnzPerColSegmentPerPanel,
                                                  UIN *__restrict__ colIndicesPerPanel_dev) {

    const UIN rowPanelId = blockIdx.x;

    const UIN warpId = threadIdx.x >> 5;

    const UIN indexOfReorderedRows = rowPanelId * ROW_PANEL_SIZE + warpId;
    if (indexOfReorderedRows >= numNonZeroRow) {
        return;
    }

    const UIN startIdxOfNumColSegments = rowPanelColSegmentOffsets[rowPanelId];
    const UIN endIdxOfNumColSegments = rowPanelColSegmentOffsets[rowPanelId + 1];
    const UIN row = reorderedRows[indexOfReorderedRows];
    const UIN endIdx = rowOffsets[row + 1];

    const UIN laneId = threadIdx.x & 31;

    for (int idx = rowOffsets[row] + laneId; idx < endIdx; idx += WARP_SIZE) {
        const UIN col = colIndices[idx];
    }
}

__global__ void analysisDescendingOrderColSegment(const UIN dense_column_segment_threshold,
                                                  const UIN *__restrict__ rowPanelColSegmentOffsets,
                                                  const UIN *__restrict__ nnzPerColSegmentPerRowPanel,
                                                  UIN *numDenseColsPerRowPanel,
                                                  UIN *numSparseColsPerRowPanel) {

}

__global__ void calculateNNZPerSparseColSegmentPerRowPanel() {

}

} // namespace kernel

void colReordering_gpu(const sparseMatrix::CSR<float> &matrix,
                       const UIN numRowPanels,
                       const std::vector<UIN> &reorderedRows,
                       const UIN dense_column_segment_threshold,
                       std::vector<UIN> &denseCols,
                       std::vector<UIN> &denseColOffsets,
                       std::vector<UIN> &sparseCols,
                       std::vector<UIN> &sparseColOffsets,
                       std::vector<UIN> &sparseDataOffsets,
                       float &time) {

    dev::vector<UIN> rowOffsets_dev(matrix.rowOffsets());
    dev::vector<UIN> colIndices_dev(matrix.colIndices());
    dev::vector<UIN> reorderedRows_dev(reorderedRows);

    dev::vector<UIN> numNonZeroColSegmentsPerRowPanel_dev(numRowPanels, 0);

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();
    kernel::calculateNumNonZeroColSegmentsPerRowPanel<<<numRowPanels, 512>>>(matrix.col(),
        rowOffsets_dev.data(),
        colIndices_dev.data(),
        reorderedRows.size(),
        reorderedRows_dev.data(),
        numNonZeroColSegmentsPerRowPanel_dev.data());
    timeCalculator.endClock();
    float calculateNumOfNonZeroColsInEachRowPanel_time = timeCalculator.getTime();
    printf("calculateNumOfNonZeroColsInEachRowPanel_time: %f ms\n", calculateNumOfNonZeroColsInEachRowPanel_time);

    dev::vector<UIN> rowPanelColSegmentOffsets_dev(numRowPanels + 1);

    timeCalculator.startClock();
    dev::fill_n(rowPanelColSegmentOffsets_dev.data(), 1, 0);
    dev::inclusive_scan(numNonZeroColSegmentsPerRowPanel_dev.data(),
                        numNonZeroColSegmentsPerRowPanel_dev.data() + numNonZeroColSegmentsPerRowPanel_dev.size(),
                        rowPanelColSegmentOffsets_dev.data() + 1);
    timeCalculator.endClock();
    float initRowPanelColOffsets_time = timeCalculator.getTime();
    printf("initRowPanelColOffsets_time: %f ms\n", initRowPanelColOffsets_time);

    const UIN sumColSegments = rowPanelColSegmentOffsets_dev.back_data();
    dev::vector<UIN> nnzPerColSegmentPerRowPanel_dev(sumColSegments, 0);
    dev::vector<UIN> colIndicesPerPanel_dev(sumColSegments);

    timeCalculator.startClock();
    kernel::calculateNNZPerColSegmentPerPanel<<<1, 1>>>(matrix.col(),
        rowOffsets_dev.data(),
        colIndices_dev.data(),
        reorderedRows.size(),
        reorderedRows_dev.data(),
        rowPanelColSegmentOffsets_dev.data(),
        nnzPerColSegmentPerRowPanel_dev.data(),
        colIndicesPerPanel_dev.data());
    timeCalculator.endClock();
    float calculateNNZPerColSegmentPerPanel_time = timeCalculator.getTime();
    printf("calculateNNZPerColSegmentPerPanel_time: %f ms\n", calculateNNZPerColSegmentPerPanel_time);

    std::vector<UIN> rowPanelColSegmentOffsets = d2h(rowPanelColSegmentOffsets_dev);

    timeCalculator.startClock();
    // TODO: 是否可以使用流并行
    for (int rowPanel = 0; rowPanel < numRowPanels; ++rowPanel) {
        const UIN startIdx = rowPanelColSegmentOffsets[rowPanel];
        const UIN endIdx = rowPanelColSegmentOffsets[rowPanel + 1];
        dev::sort_by_key_descending_order(nnzPerColSegmentPerRowPanel_dev.data() + startIdx,
                                          nnzPerColSegmentPerRowPanel_dev.data() + endIdx,
                                          colIndicesPerPanel_dev.data() + startIdx);
    }
    timeCalculator.endClock();
    float sortNNZPerColSegmentPerRowPanel_time = timeCalculator.getTime();
    printf("sortNNZPerColSegmentPerRowPanel_time: %f ms\n", sortNNZPerColSegmentPerRowPanel_time);

    dev::vector<UIN> numDenseColsPerRowPanel_dev(numRowPanels);
    dev::vector<UIN> numSparseColsPerRowPanel_dev(numRowPanels);

    timeCalculator.startClock();
    kernel::analysisDescendingOrderColSegment<<<1, 1>>>(dense_column_segment_threshold,
        rowPanelColSegmentOffsets_dev.data(),
        nnzPerColSegmentPerRowPanel_dev.data(),
        numDenseColsPerRowPanel_dev.data(),
        numSparseColsPerRowPanel_dev.data());
    timeCalculator.endClock();
    float analysisDescendingOrderColSegment_time = timeCalculator.getTime();
    printf("analysisDescendingOrderColSegment_time: %f ms\n", analysisDescendingOrderColSegment_time);

    dev::vector<UIN> denseColOffsets_dev(numRowPanels + 1);
    dev::fill_n(denseColOffsets_dev.data(), 1, 0);
    dev::vector<UIN> sparseColOffsets_dev(numRowPanels + 1);
    dev::fill_n(sparseColOffsets_dev.data(), 1, 0);

    timeCalculator.startClock();
    dev::inclusive_scan(numDenseColsPerRowPanel_dev.data(),
                        numDenseColsPerRowPanel_dev.data() + numDenseColsPerRowPanel_dev.size(),
                        denseColOffsets_dev.data() + 1);
    dev::inclusive_scan(numSparseColsPerRowPanel_dev.data(),
                        numSparseColsPerRowPanel_dev.data() + numSparseColsPerRowPanel_dev.size(),
                        sparseColOffsets_dev.data() + 1);
    timeCalculator.endClock();
    float getOffsets_time = timeCalculator.getTime();
    printf("getOffsets_time: %f ms\n", getOffsets_time);

    dev::vector<UIN> nnzPerSparseColSegmentPerRowPanel_dev(numRowPanels);
    timeCalculator.startClock();
    kernel::calculateNNZPerSparseColSegmentPerRowPanel<<<1, 1>>>();
    timeCalculator.endClock();
    float calculateNNZPerSparseColSegmentPerRowPanel_time = timeCalculator.getTime();
    printf("calculateNNZPerSparseColSegmentPerRowPanel_time: %f ms\n", calculateNNZPerSparseColSegmentPerRowPanel_time);

    dev::vector<UIN> sparseDataOffsets_dev(numRowPanels + 1);
    dev::fill_n(sparseDataOffsets_dev.data(), 1, 0);
    timeCalculator.startClock();
    dev::inclusive_scan(nnzPerSparseColSegmentPerRowPanel_dev.data(),
                        nnzPerSparseColSegmentPerRowPanel_dev.data() + nnzPerSparseColSegmentPerRowPanel_dev.size(),
                        sparseDataOffsets_dev.data() + 1);
    timeCalculator.endClock();
    float getSparseDataOffsets_dev_time = timeCalculator.getTime();
    printf("getSparseDataOffsets_dev_time: %f ms\n", getSparseDataOffsets_dev_time);

    denseColOffsets = d2h(denseColOffsets_dev);
    sparseColOffsets = d2h(sparseColOffsets_dev);

    sparseDataOffsets = d2h(sparseDataOffsets_dev);

    denseCols.resize(denseColOffsets[numRowPanels]);
    sparseCols.resize(sparseColOffsets[numRowPanels]);

    std::vector<UIN> numDenseColsPerRowPanel = d2h(numDenseColsPerRowPanel_dev);
    std::vector<UIN> colIndicesPerPanel = d2h(colIndicesPerPanel_dev);
#pragma omp parallel for
    for (int rowPanelId = 0; rowPanelId < numRowPanels; ++rowPanelId) {
        UIN *colsCurrentRowPanelPtr = colIndicesPerPanel.data() + rowPanelColSegmentOffsets[rowPanelId];
        UIN *colsCurrentRowPanelEndPtr = colIndicesPerPanel.data() + rowPanelColSegmentOffsets[rowPanelId + 1];

        UIN *denseColsCurrentRowPanelPtr = colsCurrentRowPanelPtr;
        UIN *denseColsCurrentRowPanelEndPtr = colsCurrentRowPanelPtr + numDenseColsPerRowPanel[rowPanelId];
        std::copy(denseColsCurrentRowPanelPtr, denseColsCurrentRowPanelEndPtr,
                  denseCols.data() + denseColOffsets[rowPanelId]);

        std::copy(denseColsCurrentRowPanelEndPtr, colsCurrentRowPanelEndPtr,
                  sparseCols.begin() + sparseColOffsets[rowPanelId]);
    }

    time = calculateNumOfNonZeroColsInEachRowPanel_time + initRowPanelColOffsets_time
        + calculateNNZPerColSegmentPerPanel_time + sortNNZPerColSegmentPerRowPanel_time
        + analysisDescendingOrderColSegment_time + getOffsets_time
        + calculateNNZPerSparseColSegmentPerRowPanel_time + getSparseDataOffsets_dev_time;
}

// return the number of dense column segments and the number of sparse column segments
std::pair<UIN, UIN> analysisDescendingOrderColSegment(const UIN dense_column_segment_threshold,
                                                      const std::vector<UIN> &numOfNonZeroInEachColSegment) {
    UIN numNonZeroColSegment = 0;
    UIN numDenseColSegment = 0;

    while (numNonZeroColSegment < numOfNonZeroInEachColSegment.size()
        && numOfNonZeroInEachColSegment[numNonZeroColSegment] > 0) {

        if (numOfNonZeroInEachColSegment[numNonZeroColSegment] >= dense_column_segment_threshold) {
            ++numDenseColSegment;
        }

        ++numNonZeroColSegment;
    }

    // 优化密集块平均密度
//    const UIN remainderNumber = numDenseColSegment % each_thread_block_counts_the_number_Of_cols;
//    numDenseColSegment -= remainderNumber;
//    if (remainderNumber > each_thread_block_counts_the_number_Of_cols / 2) {
//        numDenseColSegment = std::min(static_cast<UIN>(numOfNonZeroInEachColSegment.size()),
//                                      numDenseColSegment + each_thread_block_counts_the_number_Of_cols);
//    }

    const UIN numSparseColSegment = numNonZeroColSegment - numDenseColSegment;
    return std::make_pair(numDenseColSegment, numSparseColSegment);
}

// Divide rows into row panels and columns reordered in each row panel. After the columns reordered, the columns are divided into dense and sparse residual columns.
void colReordering_cpu(const sparseMatrix::CSR<float> &matrix,
                       const UIN numRowPanels,
                       const std::vector<UIN> &reorderedRows,
                       const UIN dense_column_segment_threshold,
                       std::vector<UIN> &denseCols,
                       std::vector<UIN> &denseColOffsets,
                       std::vector<UIN> &sparseCols,
                       std::vector<UIN> &sparseColOffsets,
                       std::vector<UIN> &sparseDataOffsets,
                       float &time) {
    std::vector<UIN> numOfDenseColSegmentInEachRowPanel(numRowPanels, 0);
    std::vector<UIN> numOfSparseColSegmentInEachRowPanel(numRowPanels, 0);
    std::vector<std::vector<UIN>> nonZeroColsInEachRowPanel(numRowPanels);
    std::vector<UIN> numOfSparsePartDataInEachRowPanel(numRowPanels, 0);

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();

#pragma omp parallel for schedule(dynamic)
    for (int rowPanelId = 0; rowPanelId < numRowPanels; ++rowPanelId) {
        const UIN startIdxOfReorderedRowsCurrentRowPanel = rowPanelId * ROW_PANEL_SIZE;
        const UIN endIdxOfReorderedRowsCurrentRowPanel = std::min(
            startIdxOfReorderedRowsCurrentRowPanel + ROW_PANEL_SIZE,
            static_cast<UIN>(reorderedRows.size()));

        // Count the number of non-zero elements for each column segment
        std::vector<UIN> numOfNonZeroInEachColSegment(matrix.col(), 0);
        for (int reorderedRowIndex = startIdxOfReorderedRowsCurrentRowPanel;
             reorderedRowIndex < endIdxOfReorderedRowsCurrentRowPanel;
             ++reorderedRowIndex) { // Loop through the rows in this row panel
            const UIN row = reorderedRows[reorderedRowIndex];
            for (UIN idx = matrix.rowOffsets()[row]; idx < matrix.rowOffsets()[row + 1];
                 ++idx) { // Loop through the columns in this row
                const UIN col = matrix.colIndices()[idx];
                ++numOfNonZeroInEachColSegment[col];
            }
        }

        std::vector<UIN> colIndices_sparse(numOfNonZeroInEachColSegment.size()); // Containing empty columns
        host::sequence(colIndices_sparse.data(), colIndices_sparse.data() + colIndices_sparse.size(), 0);

        // 计算具有非零元素的列的数量
        size_t numNonZeroCols = host::count_if_positive(numOfNonZeroInEachColSegment.data(),
                                                        numOfNonZeroInEachColSegment.data()
                                                            + numOfNonZeroInEachColSegment.size());
        std::vector<UIN> numOfNonZeroInEachColSegment_dense(numNonZeroCols);
        std::vector<UIN> colIndices_dense(numNonZeroCols);

        host::copy_if_positive(numOfNonZeroInEachColSegment.data(),
                               numOfNonZeroInEachColSegment.data() + numOfNonZeroInEachColSegment.size(),
                               numOfNonZeroInEachColSegment.data(),
                               numOfNonZeroInEachColSegment_dense.data());
        host::copy_if_positive(colIndices_sparse.data(),
                               colIndices_sparse.data() + colIndices_sparse.size(),
                               numOfNonZeroInEachColSegment.data(),
                               colIndices_dense.data());

        host::sort_by_key_descending_order(numOfNonZeroInEachColSegment_dense.data(),
                                           numOfNonZeroInEachColSegment_dense.data()
                                               + numOfNonZeroInEachColSegment_dense.size(),
                                           colIndices_dense.data());

        nonZeroColsInEachRowPanel[rowPanelId] = colIndices_dense;

        const auto [numDenseColSegment, numSparseColSegment] =
            analysisDescendingOrderColSegment(dense_column_segment_threshold, numOfNonZeroInEachColSegment_dense);

        UIN numSparsePartData = 0;
        for (int i = numDenseColSegment; i < numDenseColSegment + numSparseColSegment; ++i) {
            numSparsePartData += numOfNonZeroInEachColSegment_dense[i];
        }
        numOfDenseColSegmentInEachRowPanel[rowPanelId] = numDenseColSegment;
        numOfSparseColSegmentInEachRowPanel[rowPanelId] = numSparseColSegment;
        numOfSparsePartDataInEachRowPanel[rowPanelId] = numSparsePartData;
    }

    // Initialize the sparsePartDataOffsets
    sparseDataOffsets.resize(numRowPanels + 1);
    sparseDataOffsets[0] = 0;
    host::inclusive_scan(numOfSparsePartDataInEachRowPanel.data(),
                         numOfSparsePartDataInEachRowPanel.data() + numOfSparsePartDataInEachRowPanel.size(),
                         sparseDataOffsets.data() + 1);

    // Initialize the denseColOffsets
    denseColOffsets.resize(numRowPanels + 1);
    denseColOffsets[0] = 0;
    host::inclusive_scan(numOfDenseColSegmentInEachRowPanel.data(),
                         numOfDenseColSegmentInEachRowPanel.data() + numOfDenseColSegmentInEachRowPanel.size(),
                         denseColOffsets.data() + 1);

    // Initialize the sparseColOffsets
    sparseColOffsets.resize(numRowPanels + 1);
    sparseColOffsets[0] = 0;
    host::inclusive_scan(numOfSparseColSegmentInEachRowPanel.data(),
                         numOfSparseColSegmentInEachRowPanel.data() + numOfSparseColSegmentInEachRowPanel.size(),
                         sparseColOffsets.data() + 1);

    // Initialize the denseCols,sparseColIndices
    denseCols.resize(denseColOffsets[numRowPanels]);
    sparseCols.resize(sparseColOffsets[numRowPanels]);
#pragma omp parallel for
    for (int rowPanelId = 0; rowPanelId < numRowPanels; ++rowPanelId) {
        UIN *colsCurrentRowPanelPtr = nonZeroColsInEachRowPanel[rowPanelId].data();
        UIN *colsCurrentRowPanelEndPtr =
            nonZeroColsInEachRowPanel[rowPanelId].data() + nonZeroColsInEachRowPanel[rowPanelId].size();

        UIN *denseColsCurrentRowPanelPtr = colsCurrentRowPanelPtr;
        UIN *denseColsCurrentRowPanelEndPtr = colsCurrentRowPanelPtr + numOfDenseColSegmentInEachRowPanel[rowPanelId];
        std::copy(denseColsCurrentRowPanelPtr,
                  denseColsCurrentRowPanelEndPtr,
                  denseCols.begin() + denseColOffsets[rowPanelId]);

        UIN *sparseColsCurrentRowPanelPtr = denseColsCurrentRowPanelEndPtr;
        UIN *sparseColsCurrentRowPanelEndPtr = colsCurrentRowPanelEndPtr;
        std::copy(sparseColsCurrentRowPanelPtr,
                  sparseColsCurrentRowPanelEndPtr,
                  sparseCols.begin() + sparseColOffsets[rowPanelId]);
    }

    timeCalculator.endClock();
    time = timeCalculator.getTime();
}

// Divide rows into row panels and columns reordered in each row panel.
void colReordering(const sparseMatrix::CSR<float> &matrix,
                   const UIN numRowPanels,
                   const std::vector<UIN> &reorderedRows,
                   std::vector<UIN> &reorderedCols,
                   std::vector<UIN> &reorderedColOffsets) {
    std::vector<UIN> numOfNonZeroColSegmentInEachRowPanel(numRowPanels, 0);
    std::vector<std::vector<UIN>>
        colsInEachRowPanel_sparse(numRowPanels, std::vector<UIN>(matrix.col())); // Containing empty columns
#pragma omp parallel for schedule(dynamic)
    for (int rowPanelId = 0; rowPanelId < numRowPanels; ++rowPanelId) {
        const UIN startIdxOfReorderedRowsCurrentRowPanel = rowPanelId * ROW_PANEL_SIZE;
        const UIN endIdxOfReorderedRowsCurrentRowPanel = std::min(
            startIdxOfReorderedRowsCurrentRowPanel + ROW_PANEL_SIZE,
            static_cast<UIN>(reorderedRows.size()));

        // Count the number of non-zero elements for each column segment
        std::vector<UIN> numOfNonZeroInEachColSegment(matrix.col(), 0);
        for (int reorderedRowIndex = startIdxOfReorderedRowsCurrentRowPanel;
             reorderedRowIndex < endIdxOfReorderedRowsCurrentRowPanel;
             ++reorderedRowIndex) { // Loop through the rows in this row panel
            const UIN row = reorderedRows[reorderedRowIndex];
            for (UIN idx = matrix.rowOffsets()[row]; idx < matrix.rowOffsets()[row + 1];
                 ++idx) { // Loop through the columns in this row
                const UIN col = matrix.colIndices()[idx];
                ++numOfNonZeroInEachColSegment[col];
            }
        }

        std::vector<UIN> &colIndicesCurrentRowPanel = colsInEachRowPanel_sparse[rowPanelId];
        std::iota(colIndicesCurrentRowPanel.begin(), colIndicesCurrentRowPanel.end(), 0);
        host::sort_by_key_descending_order(numOfNonZeroInEachColSegment.data(),
                                           numOfNonZeroInEachColSegment.data() + numOfNonZeroInEachColSegment.size(),
                                           colIndicesCurrentRowPanel.data());
        UIN numNonZeroColSegment = 0;
        while (numNonZeroColSegment < matrix.col() && numOfNonZeroInEachColSegment[numNonZeroColSegment] != 0) {
            ++numNonZeroColSegment;
        }
        numOfNonZeroColSegmentInEachRowPanel[rowPanelId] = numNonZeroColSegment;
    }

    reorderedColOffsets.resize(numRowPanels + 1);
    reorderedColOffsets[0] = 0;
    host::inclusive_scan(numOfNonZeroColSegmentInEachRowPanel.data(),
                         numOfNonZeroColSegmentInEachRowPanel.data() + numOfNonZeroColSegmentInEachRowPanel.size(),
                         reorderedColOffsets.data() + 1);

    reorderedCols.resize(reorderedColOffsets[numRowPanels]);
#pragma omp parallel for
    for (int rowPanelId = 0; rowPanelId < numRowPanels; ++rowPanelId) {
        std::copy(colsInEachRowPanel_sparse[rowPanelId].begin(),
                  colsInEachRowPanel_sparse[rowPanelId].begin() + numOfNonZeroColSegmentInEachRowPanel[rowPanelId],
                  reorderedCols.begin() + reorderedColOffsets[rowPanelId]);
    }
}