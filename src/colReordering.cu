#include <numeric>
#include <cmath>
#include <unordered_map>
#include <omp.h>
#include <cuda_runtime.h>

#include "CudaTimeCalculator.cuh"
#include "BSMR.hpp"
#include "TensorCoreConfig.cuh"
#include "parallelAlgorithm.cuh"
#include "sddmmKernel.cuh"

namespace kernel{
template <typename T>
static __inline__ __device__ T warp_reduce_sum(T value){
    /* aggregate all value that each thread within a warp holding.*/
    T ret = value;

    for (int w = 1; w < warpSize; w = w << 1){
        T tmp = __shfl_xor_sync(0xffffffff, ret, w);
        ret += tmp;
    }
    return ret;
}

template <typename T>
static __inline__ __device__ T reduce_sum(T value, T* shm){
    unsigned int stride;
    unsigned int tid = threadIdx.x;
    T tmp = warp_reduce_sum(value); // perform warp shuffle first for less utilized shared memory

    unsigned int block_warp_id = tid / warpSize;
    unsigned int lane = tid % warpSize;
    if (lane == 0)
        shm[block_warp_id] = tmp;
    __syncthreads();
    for (stride = blockDim.x / (2 * warpSize); stride >= 1; stride = stride >> 1){
        if (block_warp_id < stride && lane == 0){
            shm[block_warp_id] += shm[block_warp_id + stride];
        }

        __syncthreads();
    }
    return shm[0];
}

// blockDim:[512,1,1]
__global__ void calculateNumNonZeroInColSegmentsPerRowPanel(const UIN numCols,
                                                            const UIN* __restrict__ rowOffsets,
                                                            const UIN* __restrict__ colIndices,
                                                            const UIN numNonZeroRow,
                                                            const UIN* __restrict__ reorderedRows,
                                                            UIN* numNonZeroColSegmentsPerRowPanel){
    const UIN rowPanelId = blockIdx.x;

    const UIN warpId = threadIdx.x >> 5;
    const UIN laneId = threadIdx.x & 31;

    const UIN indexOfReorderedRows = rowPanelId * ROW_PANEL_SIZE + warpId;
    if (indexOfReorderedRows >= numNonZeroRow){
        return;
    }
    const UIN row = reorderedRows[indexOfReorderedRows];

    for (int idx = rowOffsets[row] + laneId; idx < rowOffsets[row + 1]; idx += WARP_SIZE){
        const UIN col = colIndices[idx];
        atomicAdd(numNonZeroColSegmentsPerRowPanel + rowPanelId * numCols + col, 1);
    }
}

// blockDim:[512,1,1]
__global__ void calculateNumNonZeroColSegmentsPerRowPanel(const UIN numCols,
                                                          const UIN* __restrict__ rowOffsets,
                                                          const UIN* __restrict__ colIndices,
                                                          const UIN numNonZeroRow,
                                                          const UIN* __restrict__ reorderedRows,
                                                          UIN* __restrict__ numNonZeroColSegmentsPerRowPanel){
    const UIN rowPanelId = blockIdx.x;

    const UIN warpId = threadIdx.x >> 5;

    constexpr UIN numColumnsRecordedPerThreadBlock = 8192;
    __shared__ UIN nnnzPerColSegment[numColumnsRecordedPerThreadBlock];

    for (int i = threadIdx.x; i < numColumnsRecordedPerThreadBlock; i += blockDim.x){
        nnnzPerColSegment[i] = 0;
    }


    const UIN indexOfReorderedRows = rowPanelId * ROW_PANEL_SIZE + warpId;
    if (indexOfReorderedRows >= numNonZeroRow){
        return;
    }

    const UIN row = reorderedRows[indexOfReorderedRows + warpId];
    const UIN endIdx = rowOffsets[row + 1];

    const UIN laneId = threadIdx.x & 31;

    for (int idx = rowOffsets[row] + laneId; idx < endIdx; idx += WARP_SIZE){
        const UIN col = colIndices[idx];
    }
}

// blockDim:[512,1,1]
__global__ void calculateNNZPerColSegmentPerPanel(const UIN numCols,
                                                  const UIN* __restrict__ rowOffsets,
                                                  const UIN* __restrict__ colIndices,
                                                  const UIN numNonZeroRow,
                                                  const UIN* __restrict__ reorderedRows,
                                                  const UIN* __restrict__ rowPanelColSegmentOffsets,
                                                  UIN* __restrict__ nnzPerColSegmentPerPanel,
                                                  UIN* __restrict__ colIndicesPerPanel_dev){
    const UIN rowPanelId = blockIdx.x;

    const UIN warpId = threadIdx.x >> 5;

    const UIN indexOfReorderedRows = rowPanelId * ROW_PANEL_SIZE + warpId;
    if (indexOfReorderedRows >= numNonZeroRow){
        return;
    }

    const UIN startIdxOfNumColSegments = rowPanelColSegmentOffsets[rowPanelId];
    const UIN endIdxOfNumColSegments = rowPanelColSegmentOffsets[rowPanelId + 1];
    const UIN row = reorderedRows[indexOfReorderedRows];
    const UIN endIdx = rowOffsets[row + 1];

    const UIN laneId = threadIdx.x & 31;

    for (int idx = rowOffsets[row] + laneId; idx < endIdx; idx += WARP_SIZE){
        const UIN col = colIndices[idx];
    }
}

__global__ void analysisDescendingOrderColSegment(const UIN dense_column_segment_threshold,
                                                  const UIN* __restrict__ rowPanelColSegmentOffsets,
                                                  const UIN* __restrict__ nnzPerColSegmentPerRowPanel,
                                                  UIN* numDenseColsPerRowPanel,
                                                  UIN* numSparseColsPerRowPanel){
}

__global__ void calculateNNZPerSparseColSegmentPerRowPanel(){
}
} // namespace kernel

void colReordering_gpu(const sparseMatrix::CSR<float>& matrix,
                       const UIN numRowPanels,
                       const std::vector<UIN>& reorderedRows,
                       const UIN dense_column_segment_threshold,
                       std::vector<UIN>& denseCols,
                       std::vector<UIN>& denseColOffsets,
                       std::vector<UIN>& sparseCols,
                       std::vector<UIN>& sparseColOffsets,
                       std::vector<UIN>& sparseDataOffsets,
                       float& time){
    dev::vector<UIN> rowOffsets_dev(matrix.rowOffsets());
    dev::vector<UIN> colIndices_dev(matrix.colIndices());
    dev::vector<UIN> reorderedRows_dev(reorderedRows);

    dev::vector<UIN> numNonZeroPerColSegmentPerRowPanel_dev(numRowPanels * matrix.col(), 0);

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();
    kernel::calculateNumNonZeroInColSegmentsPerRowPanel<<<numRowPanels,numRowPanels * WARP_SIZE>>>(matrix.col(),
        rowOffsets_dev.data(),
        colIndices_dev.data(),
        reorderedRows.size(),
        reorderedRows_dev.data(),
        numNonZeroPerColSegmentPerRowPanel_dev.data());
    timeCalculator.endClock();
    const float calculateNumNonZeroInColSegmentsPerRowPanel_time = timeCalculator.getTime();
    printf("calculateNumNonZeroInColSegmentsPerRowPanel_time: %f ms\n",
           calculateNumNonZeroInColSegmentsPerRowPanel_time);

    std::vector<UIN> colIndices_sparse(matrix.col() * numRowPanels); // Containing empty columns
    std::vector<UIN> numNonZeroColSegmentsPerRowPanel(numRowPanels, 0);

    timeCalculator.startClock();
#pragma omp parallel for
    for (int rowPanelId = 0; rowPanelId < numRowPanels; ++rowPanelId){
        const UIN startIdx = rowPanelId * matrix.col();
        const UIN endIdx = startIdx + matrix.col();
        host::sequence(colIndices_sparse.data() + startIdx,
                       colIndices_sparse.data() + endIdx,
                       0);

        // 计算具有非零元素的列的数量
        numNonZeroColSegmentsPerRowPanel[rowPanelId] = dev::count_if_positive(
            numNonZeroPerColSegmentPerRowPanel_dev.data() + startIdx,
            numNonZeroPerColSegmentPerRowPanel_dev.data() + endIdx);
    }
    timeCalculator.endClock();
    const float countNumOfNonZeroColsInEachRowPanel_time = timeCalculator.getTime();
    printf("countNumOfNonZeroColsInEachRowPanel_time: %f ms\n", countNumOfNonZeroColsInEachRowPanel_time);

    std::vector<UIN> rowPanelColSegmentOffsets(numRowPanels + 1);
    rowPanelColSegmentOffsets[0] = 0;
    timeCalculator.startClock();
    host::inclusive_scan(numNonZeroColSegmentsPerRowPanel.data(),
                         numNonZeroColSegmentsPerRowPanel.data() + numNonZeroColSegmentsPerRowPanel.size(),
                         rowPanelColSegmentOffsets.data() + 1);
    timeCalculator.endClock();
    const float initRowPanelColOffsets_time = timeCalculator.getTime();
    printf("initRowPanelColOffsets_time: %f ms\n", initRowPanelColOffsets_time);

    const UIN numNonZeroCols = rowPanelColSegmentOffsets.back();
    dev::vector<UIN> colSegmentsPerRowPanel_dev(numNonZeroCols, 0);

    std::vector<UIN> numOfNonZeroInEachColSegment_dense(numNonZeroCols);
    std::vector<UIN> colIndices_dense(numNonZeroCols);

    // for (int rowPanelId = 0; rowPanelId < numRowPanels; ++rowPanelId){
    //     const UIN startIdx = rowPanelId * matrix.col();
    //     const UIN endIdx = startIdx + matrix.col();
    //     // 计算具有非零元素的列的数量
    //     size_t numNonZeroCols = host::count_if_positive(numOfNonZeroInEachColSegment.data(),
    //                                                     numOfNonZeroInEachColSegment.data()
    //                                                     + numOfNonZeroInEachColSegment.size());
    //     std::vector<UIN> numOfNonZeroInEachColSegment_dense(numNonZeroCols);
    //     std::vector<UIN> colIndices_dense(numNonZeroCols);
    //
    //     host::copy_if_positive(numOfNonZeroInEachColSegment.data(),
    //                            numOfNonZeroInEachColSegment.data() + numOfNonZeroInEachColSegment.size(),
    //                            numOfNonZeroInEachColSegment.data(),
    //                            numOfNonZeroInEachColSegment_dense.data());
    //     host::copy_if_positive(colIndices_sparse.data(),
    //                            colIndices_sparse.data() + colIndices_sparse.size(),
    //                            numOfNonZeroInEachColSegment.data(),
    //                            colIndices_dense.data());
    //
    //     host::sort_by_key_descending_order(numOfNonZeroInEachColSegment_dense.data(),
    //                                        numOfNonZeroInEachColSegment_dense.data()
    //                                        + numOfNonZeroInEachColSegment_dense.size(),
    //                                        colIndices_dense.data());
    // }


    time = calculateNumNonZeroInColSegmentsPerRowPanel_time +
        countNumOfNonZeroColsInEachRowPanel_time +
        initRowPanelColOffsets_time;
}

// return the number of dense column segments and the number of sparse column segments
std::pair<UIN, UIN> analysisDescendingOrderColSegment(const float blockDensityThreshold,
                                                      const std::vector<UIN>& numOfNonZeroInEachColSegment){
    const UIN numNonZeroThreshold = static_cast<UIN>(std::ceil(blockDensityThreshold * BLOCK_SIZE));
    UIN numNonZeroColSegment = 0;
    UIN numDenseColSegment = 0;

    UIN numNonZeroInBlock = 0;
    while (numNonZeroColSegment < numOfNonZeroInEachColSegment.size()
        && numOfNonZeroInEachColSegment[numNonZeroColSegment] > 0){
        if (numNonZeroColSegment % BLOCK_COL_SIZE == 0){
            if (numNonZeroInBlock >= numNonZeroThreshold){
                // If the number of non-zero elements in the current block is greater than the threshold, it is a dense column segment
                numDenseColSegment += BLOCK_COL_SIZE;
                numDenseColSegment = std::min(numDenseColSegment,
                                              static_cast<UIN>(numOfNonZeroInEachColSegment.size()));
            }
            numNonZeroInBlock = 0;
        }
        numNonZeroInBlock += numOfNonZeroInEachColSegment[numNonZeroColSegment];

        ++numNonZeroColSegment;
    }

    const UIN numSparseColSegment = numNonZeroColSegment - numDenseColSegment;
    return std::make_pair(numDenseColSegment, numSparseColSegment);
}

// Divide rows into row panels and columns reordered in each row panel. After the columns reordered, the columns are divided into dense and sparse residual columns.
void colReordering_cpu(const sparseMatrix::CSR<float>& matrix,
                       const UIN numRowPanels,
                       const std::vector<UIN>& reorderedRows,
                       const float blockDensityThreshold,
                       std::vector<UIN>& denseCols,
                       std::vector<UIN>& denseColOffsets,
                       std::vector<UIN>& sparseCols,
                       std::vector<UIN>& sparseColOffsets,
                       std::vector<UIN>& sparseDataOffsets,
                       float& time){
    std::vector<UIN> numOfDenseColSegmentInEachRowPanel(numRowPanels, 0);
    std::vector<UIN> numOfSparseColSegmentInEachRowPanel(numRowPanels, 0);
    std::vector<std::vector<UIN>> nonZeroColsInEachRowPanel(numRowPanels);
    std::vector<UIN> numOfSparsePartDataInEachRowPanel(numRowPanels, 0);

    CudaTimeCalculator timeCalculator;
    timeCalculator.startClock();

#pragma omp parallel for schedule(dynamic)
    for (int rowPanelId = 0; rowPanelId < numRowPanels; ++rowPanelId){
        const UIN startIdxOfReorderedRowsCurrentRowPanel = rowPanelId * ROW_PANEL_SIZE;
        const UIN endIdxOfReorderedRowsCurrentRowPanel = std::min(
            startIdxOfReorderedRowsCurrentRowPanel + ROW_PANEL_SIZE,
            static_cast<UIN>(reorderedRows.size()));

        // Count the number of non-zero elements for each column segment
        std::vector<UIN> numOfNonZeroInEachColSegment(matrix.col(), 0);
        for (int reorderedRowIndex = startIdxOfReorderedRowsCurrentRowPanel;
             reorderedRowIndex < endIdxOfReorderedRowsCurrentRowPanel;
             ++reorderedRowIndex){
            // Loop through the rows in this row panel
            const UIN row = reorderedRows[reorderedRowIndex];
            for (UIN idx = matrix.rowOffsets()[row]; idx < matrix.rowOffsets()[row + 1];
                 ++idx){
                // Loop through the columns in this row
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
            analysisDescendingOrderColSegment(blockDensityThreshold, numOfNonZeroInEachColSegment_dense);

        UIN numSparsePartData = 0;
        for (int i = numDenseColSegment; i < numDenseColSegment + numSparseColSegment; ++i){
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
    for (int rowPanelId = 0; rowPanelId < numRowPanels; ++rowPanelId){
        UIN* colsCurrentRowPanelPtr = nonZeroColsInEachRowPanel[rowPanelId].data();
        UIN* colsCurrentRowPanelEndPtr =
            nonZeroColsInEachRowPanel[rowPanelId].data() + nonZeroColsInEachRowPanel[rowPanelId].size();

        UIN* denseColsCurrentRowPanelPtr = colsCurrentRowPanelPtr;
        UIN* denseColsCurrentRowPanelEndPtr = colsCurrentRowPanelPtr + numOfDenseColSegmentInEachRowPanel[rowPanelId];
        std::copy(denseColsCurrentRowPanelPtr,
                  denseColsCurrentRowPanelEndPtr,
                  denseCols.begin() + denseColOffsets[rowPanelId]);

        UIN* sparseColsCurrentRowPanelPtr = denseColsCurrentRowPanelEndPtr;
        UIN* sparseColsCurrentRowPanelEndPtr = colsCurrentRowPanelEndPtr;
        std::copy(sparseColsCurrentRowPanelPtr,
                  sparseColsCurrentRowPanelEndPtr,
                  sparseCols.begin() + sparseColOffsets[rowPanelId]);
    }

    timeCalculator.endClock();
    time = timeCalculator.getTime();
}

// Divide rows into row panels and columns reordered in each row panel.
void colReordering(const sparseMatrix::CSR<float>& matrix,
                   const UIN numRowPanels,
                   const std::vector<UIN>& reorderedRows,
                   std::vector<UIN>& reorderedCols,
                   std::vector<UIN>& reorderedColOffsets){
    std::vector<UIN> numOfNonZeroColSegmentInEachRowPanel(numRowPanels, 0);
    std::vector<std::vector<UIN>>
        colsInEachRowPanel_sparse(numRowPanels, std::vector<UIN>(matrix.col())); // Containing empty columns
#pragma omp parallel for schedule(dynamic)
    for (int rowPanelId = 0; rowPanelId < numRowPanels; ++rowPanelId){
        const UIN startIdxOfReorderedRowsCurrentRowPanel = rowPanelId * ROW_PANEL_SIZE;
        const UIN endIdxOfReorderedRowsCurrentRowPanel = std::min(
            startIdxOfReorderedRowsCurrentRowPanel + ROW_PANEL_SIZE,
            static_cast<UIN>(reorderedRows.size()));

        // Count the number of non-zero elements for each column segment
        std::vector<UIN> numOfNonZeroInEachColSegment(matrix.col(), 0);
        for (int reorderedRowIndex = startIdxOfReorderedRowsCurrentRowPanel;
             reorderedRowIndex < endIdxOfReorderedRowsCurrentRowPanel;
             ++reorderedRowIndex){
            // Loop through the rows in this row panel
            const UIN row = reorderedRows[reorderedRowIndex];
            for (UIN idx = matrix.rowOffsets()[row]; idx < matrix.rowOffsets()[row + 1];
                 ++idx){
                // Loop through the columns in this row
                const UIN col = matrix.colIndices()[idx];
                ++numOfNonZeroInEachColSegment[col];
            }
        }

        std::vector<UIN>& colIndicesCurrentRowPanel = colsInEachRowPanel_sparse[rowPanelId];
        std::iota(colIndicesCurrentRowPanel.begin(), colIndicesCurrentRowPanel.end(), 0);
        host::sort_by_key_descending_order(numOfNonZeroInEachColSegment.data(),
                                           numOfNonZeroInEachColSegment.data() + numOfNonZeroInEachColSegment.size(),
                                           colIndicesCurrentRowPanel.data());
        UIN numNonZeroColSegment = 0;
        while (numNonZeroColSegment < matrix.col() && numOfNonZeroInEachColSegment[numNonZeroColSegment] != 0){
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
    for (int rowPanelId = 0; rowPanelId < numRowPanels; ++rowPanelId){
        std::copy(colsInEachRowPanel_sparse[rowPanelId].begin(),
                  colsInEachRowPanel_sparse[rowPanelId].begin() + numOfNonZeroColSegmentInEachRowPanel[rowPanelId],
                  reorderedCols.begin() + reorderedColOffsets[rowPanelId]);
    }
}
