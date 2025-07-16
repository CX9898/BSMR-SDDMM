#include "config.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <fstream>
#include <mma.h>
// #include <sputnik/spmm/cuda_spmm.h>
// #include <sputnik/sputnik.h>
#include <sstream>
#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <torch/extension.h>
#include <vector>
#define WPB 8
#define EXE_TIME 10
#define NUM_SM_GPU 128 // 4090
#define USE_SPUTNIK
using namespace nvcuda;

struct GpuTimer {
  cudaEvent_t start;
  cudaEvent_t stop;
  GpuTimer() {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start() { cudaEventRecord(start); }

  void Stop() { cudaEventRecord(stop); }

  float Elapsed() {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

// From (https://github.com/xxcclong/GNN-Computing)
typedef uint64_t clocktype;
struct Dur {
  clocktype begin;
  clocktype end;
  int smid = -1;
  Dur(clocktype x, clocktype y, int outsm) {
    begin = x;
    end = y;
    smid = outsm;
  }
};

bool cmp(Dur x, Dur y) { return (x.end > y.end); }
static __device__ inline uint64_t GlobalTimer64(void) {
  volatile uint64_t first_reading;
  volatile uint32_t second_reading;
  uint32_t high_bits_first;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(first_reading));
  high_bits_first = first_reading >> 32;
  asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(second_reading));
  if (high_bits_first == second_reading) {
    return first_reading;
  }
  // Return the value with the updated high bits, but the low bits set to 0.
  return ((uint64_t)second_reading) << 32;
}
__device__ inline uint getSMId() {
  uint smid;
  asm("mov.u32 %0, %smid;" : "=r"(smid));
  return smid;
}

//////////////////////////////////////////////////////////////////////
/// Preprocessing
//////////////////////////////////////////////////////////////////////
__global__ void roundup_to_multiple_of_eight(int *input, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid < size) {
    int rounded_value = ((input[tid] + 7) / 8) * 8;
    input[tid] = rounded_value;
  }
}

__global__ void get_padding_tileid_kernel(int *ori_offset, uint8_t *ori_tileid,
                                          int *padded_offset,
                                          uint8_t *padded_tileid, int size) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < size) {
    int s = ori_offset[tid];
    int e = ori_offset[tid + 1];
    int s1 = padded_offset[tid];
    for (int i = 0; i < e - s; i++) {
      padded_tileid[s1 + i] = ori_tileid[s + i];
    }
  }
}


__global__ void fill_edgeToRow(int *edgeToRow, int *nodePointer,
                               int num_nodes) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  int nid = tid / 32;
  int laneid = tid % 32;
  // check a valid node range.
  if (nid < num_nodes) {
#pragma unroll
    for (int eid = nodePointer[nid] + laneid; eid < nodePointer[nid + 1];
         eid += 32) {
      edgeToRow[eid] = nid;
    }
  }
}

void fill_edgeToRow_cuda(int *edgeToRow, int *nodePointer, int num_nodes) {
  int wrap_size = 32;
  int block_size = 1024;
  int grid_size = (num_nodes * wrap_size + block_size - 1) / block_size;
  fill_edgeToRow<<<grid_size, block_size>>>(edgeToRow, nodePointer, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    // print the CUDA error message and exit
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/*Generate segment*/
__global__ void fill_segment(int *nodePointer, int *seg_out, int blockSize_h,
                             int blockSize_w, int num_nodes) {
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each block one window
  //Window开始的行
  unsigned block_start = nodePointer[winId * blockSize_h];
  //Window结束的行
  unsigned block_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  //window内非零元个数
  unsigned num_window_edges = block_end - block_start;
//   if(winId==0 && threadIdx.x==0){
// 	printf("%d\n", num_window_edges);
//   }
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  for (unsigned idx = tid; idx < num_window_edges; idx += threadPerBlock) {
    seg_out[block_start + idx] = winId;
  }
}

void fill_segment_cuda(int *nodePointer, int *seg_out, int blockSize_h,
                       int blockSize_w, int num_nodes) {
  // 每个window由512个线程负责
  int block_size = 512;
  int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
  fill_segment<<<window_count, block_size>>>(nodePointer, seg_out, blockSize_h,
                                             blockSize_w, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/*Generate TCblock_rowid*/
__global__ void generate_tcblock_rowid(int *rowwindow_offset,
                                       int *tcblock_rowid,
                                       int num_row_windows) {
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = rowwindow_offset[winId];
  unsigned block_end = rowwindow_offset[min(winId + 1, num_row_windows)];
  unsigned num_blocks = block_end - block_start;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  for (unsigned idx = tid; idx < num_blocks; idx += threadPerBlock) {
    tcblock_rowid[block_start + idx] = winId;
  }
}
void generate_tcblock_rowid_cuda(int *rowwindow_offset, int *tcblock_rowid,
                                 int num_row_windows) {
  int block_size = 512;
  int window_count = num_row_windows;
  generate_tcblock_rowid<<<window_count, block_size>>>(
      rowwindow_offset, tcblock_rowid, num_row_windows);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/* Generate edge2column*/
__device__ __forceinline__ int binarysearch(int *arr, int size, int target) {
  int left = 0;
  int right = size - 1;
  while (left <= right) {
    int mid = left + (right - left) / 2;
    if (arr[mid] == target) {
      while (mid > 0 && arr[mid - 1] == target) {
        mid--;
      }
      return mid;
    } else if (arr[mid] < target) {
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }
  return -1;
}
__device__ __forceinline__ void inplace_deduplication(int *array, int length,
                                                      int *loc) {
  int cur = 1;
  while (cur < length) {
    if (array[cur] != array[cur - 1]) {
      (*loc)++;
      array[(*loc)] = array[cur];
    }
    cur++;
  }

  (*loc)++;
}

__device__ __forceinline__ void inplace_deduplication_libra_spmm(int *array, int *counts, int length, int *loc) {
  int count = 1; // 记录当前元素的计数
  for (int cur = 1; cur < length; cur++) {
    if (array[cur] != array[cur - 1]) {
      counts[*loc] = count; // 保存上一个元素的计数
      (*loc)++;             // 更新位置
      array[*loc] = array[cur]; // 将当前元素写入去重数组
      count = 1;            // 重置计数器
    } else {
      count++; // 若相同则增加当前元素的计数
    }
  }

  counts[*loc] = count; // 保存最后一个元素的计数
  (*loc)++;             // 更新位置，表示最终去重后元素个数
}

//去重，以及求vector_num
__device__ __forceinline__ void distribute_libra_spmm(int *array, int *counts, int length, int *loc,
			int threshold, int *vector_num, int *vector_nnz) {
	int count = 1; // 记录当前元素的计数
	for (int cur = 1; cur < length; cur++) {
		if (array[cur] != array[cur - 1]) {
			counts[*loc] = count; // 保存上一个元素的计数
			(*loc)++;             // 更新位置
			array[*loc] = array[cur]; // 将当前元素写入去重数组
			//判断是否超过阈值
			if(count>=threshold){
				(*vector_num)++;
				(*vector_nnz)+=count;
			}
			count = 1;            // 重置计数器
		} else {
			count++; // 若相同则增加当前元素的计数
		}
	}
	counts[*loc] = count; // 保存最后一个元素的计数
	(*loc)++;             // 更新位置，表示最终去重后元素个数
}

__device__ __forceinline__ void distribute_cuda_tile_libra_spmm(
	int *counts_cur, int *edgetocol, int start_row, int num_nodes,
	int *nodePointer, int threshold, int Short_len, int c_s, int *cuda_long, int* cuda_short,
	int *cuda_long_group, int * cuda_short_group) {

	int cur = 0;
	//遍历每一行,统计每行cuda tile的元素个数
	for (int cur_row = start_row; cur_row < min(start_row+8, num_nodes); cur_row++) {
		//遍历当前行的所有元素
		for(int m=nodePointer[cur_row]; m<nodePointer[cur_row+1]; m++){
			//如果当前元素的newcol的值小于threshold,则交由CUDA tile
			int col_density = counts_cur[edgetocol[m]];
			if(col_density < threshold){
				cuda_long[cur]++;
			}
		}
		cur++;
	}

	//拆分cuda_long
	for(int i=0; i<8; i++){
		//如果是短行
		if(cuda_long[i]<= Short_len)
		{
			cuda_short[i] = cuda_long[i];
			cuda_long[i] = 0;
			(*cuda_short_group)++;
		}else{
			//如果是长行, 是否需要差分
			if(cuda_long[i]<=c_s){
				//不需要拆分
				(*cuda_long_group)++;
			}else{
				//需要拆分
				(*cuda_long_group) += cuda_long[i]/c_s;
				//判断residue是否存在
				int residue = (cuda_long[i]%c_s);
				if(residue> 0)
				{
					//residue是短行
					if(residue<= Short_len)
					{				
						cuda_short[i] = residue;
						cuda_long[i] -= residue;
						(*cuda_short_group)++;
					}else{
						(*cuda_short_group)++;
					}
				}
			}
		}
	}


}



__global__ void generate_edgetocolumn(int *nodePointer, int *edgelist,
                                      int *edgelist_sort, int *edgetocol,
                                      int *blockpartition, int *blocknum,
                                      int blockSize_h, int blockSize_w,
                                      int num_nodes) {
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = nodePointer[winId * blockSize_h];
  unsigned block_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = block_end - block_start;
  if (num_window_edges == 0)
    return;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  int *start = edgelist_sort + block_start;
  int size = 0;
  //去重
  inplace_deduplication(start, num_window_edges, &size);
  //num是每个窗口有多少个block
  int num = (size + blockSize_w) / blockSize_w;
  atomicAdd(blocknum, num);
  blockpartition[winId] = num;
  for (unsigned idx = block_start; idx < block_end; idx += 1) {
    int index = binarysearch(start, size + 1, edgelist[idx]);
    edgetocol[idx] = index;
  }
}
void generate_edgetocolumn_cuda(int *nodePointer, int *edgelist,
                                int *edgelist_sort, int *edgetocol,
                                int *blockpartition, int *blocknum,
                                int blockSize_h, int blockSize_w,
                                int num_nodes) {
  int block_size = 1;
  int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
  //每个block负责一个window, 每个block中只有一个线程
//   int block_size1 = 128;
//   int block_count1 = (window_count + 127) / 128;
  generate_edgetocolumn<<<window_count, block_size>>>(
      nodePointer, edgelist, edgelist_sort, edgetocol, blockpartition, blocknum,
      blockSize_h, blockSize_w, num_nodes);
  // generate_edgetocolumn_v1<<< window_count, block_size >>> (nodePointer,
  // edgelist, edgelist_sort, edgetocol, blockpartition, blocknum, blockSize_h,
  // blockSize_w, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

__global__ void generate_edgetocolumn_fs(int *nodePointer, int *edgelist,
                                      int *edgelist_sort, int *edgetocol,
                                      int *blockpartition, int *blocknum, int *vectornum,
                                      int blockSize_h, int blockSize_w,
                                      int num_nodes) {
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = nodePointer[winId * blockSize_h];
  unsigned block_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = block_end - block_start;
  if (num_window_edges == 0)
    return;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  int *start = edgelist_sort + block_start;
  int size = 0;
  int num = 0;
  //去重
  inplace_deduplication(start, num_window_edges, &size);

  //num是每个窗口有多少个block
  if(size>0)
  num = (size + blockSize_w - 1) / blockSize_w;
  atomicAdd(blocknum, num);
  atomicAdd(vectornum, size);
  //vector个数
  blockpartition[winId] = size;
  for (unsigned idx = block_start; idx < block_end; idx += 1) {
    int index = binarysearch(start, size + 1, edgelist[idx]);
    edgetocol[idx] = index;
  }
}
void generate_edgetocolumn_cuda_fs(int *nodePointer, int *edgelist,
                                int *edgelist_sort, int *edgetocol,
                                int *blockpartition, int *blocknum, int * vectornum,
                                int blockSize_h, int blockSize_w,
                                int num_nodes) {
  int block_size = 1;
  int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
  //每个block负责一个window, 每个block中只有一个线程
//   int block_size1 = 128;
//   int block_count1 = (window_count + 127) / 128;
  generate_edgetocolumn_fs<<<window_count, block_size>>>(
      nodePointer, edgelist, edgelist_sort, edgetocol, blockpartition, blocknum, vectornum,
      blockSize_h, blockSize_w, num_nodes);
  // generate_edgetocolumn_v1<<< window_count, block_size >>> (nodePointer,
  // edgelist, edgelist_sort, edgetocol, blockpartition, blocknum, blockSize_h,
  // blockSize_w, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}


__global__ void generate_edgetocolumn_fs_ori(int *nodePointer, int *edgelist,
                                      int *edgelist_sort, int *edgetocol,
                                      int *blockpartition, int *blocknum, int *vectornum,
                                      int blockSize_h, int blockSize_w,
                                      int num_nodes) {
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = nodePointer[winId * blockSize_h];
  unsigned block_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = block_end - block_start;
  if (num_window_edges == 0)
    return;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  int *start = edgelist_sort + block_start;
  int size = 0;
  int num = 0;
  //去重
  inplace_deduplication(start, num_window_edges, &size);

  //num是每个窗口有多少个block
  if(size>0)
  num = (size + blockSize_w - 1) / blockSize_w;
  atomicAdd(blocknum, num);
  atomicAdd(vectornum, num*blockSize_w);
  //vector个数
  blockpartition[winId] = num*blockSize_w;
  for (unsigned idx = block_start; idx < block_end; idx += 1) {
    int index = binarysearch(start, size + 1, edgelist[idx]);
    edgetocol[idx] = index;
  }
}
void generate_edgetocolumn_cuda_fs_ori(int *nodePointer, int *edgelist,
                                int *edgelist_sort, int *edgetocol,
                                int *blockpartition, int *blocknum, int * vectornum,
                                int blockSize_h, int blockSize_w,
                                int num_nodes) {
  int block_size = 1;
  int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
  generate_edgetocolumn_fs_ori<<<window_count, block_size>>>(
      nodePointer, edgelist, edgelist_sort, edgetocol, blockpartition, blocknum, vectornum,
      blockSize_h, blockSize_w, num_nodes);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

__global__ void generate_edgetocolumn_fs_balance(int *nodePointer, int *edgelist,
                                      int *edgelist_sort, int *edgetocol,
                                      int *blockpartition, int *vectorpartition, int *blocknum, int *vectornum,
                                      int blockSize_h, int blockSize_w,
                                      int num_nodes, int part) {
  int winId = blockIdx.x; // each warp one window
  unsigned block_start = nodePointer[winId * blockSize_h];
  unsigned block_end =
      nodePointer[min(winId * blockSize_h + blockSize_h, num_nodes)];
  unsigned num_window_edges = block_end - block_start;
  if (num_window_edges == 0)
    return;
  const unsigned threadPerBlock = blockDim.x * blockDim.y;
  int *start = edgelist_sort + block_start;
  int size = 0;
  int num = 0;
  //去重
  inplace_deduplication(start, num_window_edges, &size);

  //num是每个窗口有多少个block
  if(size>0)
  num = (size + blockSize_w - 1) / blockSize_w;
  //group 个数
  int group = (num + part - 1) / part;
  atomicAdd(blocknum, group);
  atomicAdd(vectornum, size);
  blockpartition[winId] = group;
  vectorpartition[winId] = size;
  for (unsigned idx = block_start; idx < block_end; idx += 1) {
    int index = binarysearch(start, size + 1, edgelist[idx]);
    edgetocol[idx] = index;
  }
}
void generate_edgetocolumn_cuda_fs_balance(int *nodePointer, int *edgelist,
                                int *edgelist_sort, int *edgetocol,
                                int *blockpartition, int *vectorpartition, int *blocknum, int * vectornum,
                                int blockSize_h, int blockSize_w,
                                int num_nodes, int part) {
  int block_size = 1;
  int window_count = (num_nodes + blockSize_h - 1) / blockSize_h;
  generate_edgetocolumn_fs_balance<<<window_count, block_size>>>(
      nodePointer, edgelist, edgelist_sort, edgetocol, blockpartition, vectorpartition, blocknum, vectornum,
      blockSize_h, blockSize_w, num_nodes, part);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

/*Generate TC offset, tileid and AtoB*/
__global__ void generate_tcoffset_id_atob_fs(
    int *nodePointer, int *rowwindow_offset, int *edgeToColumn, int *edgeToRow,
    int *edgeList, half *values,
    int *sparseatob, int max_block, int num_nodes, long blockSize_h,
    int blockSize_w, int num_row_windows) {
//   extern __shared__ int pos_ptr[];
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each warp one window
  unsigned vector_start = rowwindow_offset[winId];
  unsigned vector_end = rowwindow_offset[min(winId + 1, num_row_windows)];
  unsigned num_vector = vector_end - vector_start;
  if (num_vector == 0) {
    return;
  }
  unsigned element_start = nodePointer[winId * blockSize_h];
  unsigned element_end =
      nodePointer[min(int(winId * blockSize_h + blockSize_h), num_nodes)];
  unsigned num_window_edges = element_end - element_start;
  if (num_window_edges == 0) {
    return;
  }
  //开始看每个非零元在block内的偏移了
//   auto tileid = tcblocktile_id + element_start;
  auto values_ = values + vector_start*blockSize_h;
  auto sparse_AToB = sparseatob + vector_start;
  for (unsigned e_index = element_start; e_index < element_end; e_index++) {
    unsigned col = edgeToColumn[e_index]; // new col
    unsigned tcblock_id = col / blockSize_w;
    unsigned row_local = edgeToRow[e_index] % blockSize_h;
    unsigned col_local = col % blockSize_w;

	//如果存在， 且元素在residue里，需要按每行residue偏移
	int residue = num_vector % blockSize_w;
	if(residue>0 & col>=(num_vector-residue)){
		values_[tcblock_id*blockSize_h*blockSize_w + row_local*residue + col_local] = __float2half(1.0);
	}else{
		values_[tcblock_id*blockSize_h*blockSize_w + row_local*blockSize_w + col_local] = __float2half(1.0);
	}
	sparse_AToB[tcblock_id * blockSize_w + col_local] = edgeList[e_index];
    // pos_ptr[tcblock_id]++;
  }
}

void generate_tcoffset_id_atob_cuda_fs(int *nodePointer, int *rowwindow_offset,
                                    int *edgeToColumn, int *edgeToRow,
                                    int *edgeList, half *values, int *sparseatob,
                                    int max_block, int num_nodes,
                                    int blockSize_h, int blockSize_w,
                                    int num_row_windows) {
  int block_size = 1;
  int window_count = num_row_windows;
  generate_tcoffset_id_atob_fs<<<window_count, block_size>>>(
      nodePointer, rowwindow_offset, edgeToColumn, edgeToRow, edgeList,
    values, sparseatob, max_block, num_nodes,
      blockSize_h, blockSize_w, num_row_windows);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}


__global__ void generate_tcoffset_id_atob_fs_ori(
    int *nodePointer, int *rowwindow_offset, int *edgeToColumn, int *edgeToRow,
    int *edgeList, half *values,
    int *sparseatob, int max_block, int num_nodes, long blockSize_h,
    int blockSize_w, int num_row_windows) {
//   extern __shared__ int pos_ptr[];
  int tid = threadIdx.x;
  int winId = blockIdx.x; // each warp one window
  unsigned vector_start = rowwindow_offset[winId];
  unsigned vector_end = rowwindow_offset[min(winId + 1, num_row_windows)];
  unsigned num_vector = vector_end - vector_start;
  if (num_vector == 0) {
    return;
  }
  unsigned element_start = nodePointer[winId * blockSize_h];
  unsigned element_end =
      nodePointer[min(int(winId * blockSize_h + blockSize_h), num_nodes)];
  unsigned num_window_edges = element_end - element_start;
  if (num_window_edges == 0) {
    return;
  }
  //开始看每个非零元在block内的偏移了
//   auto tileid = tcblocktile_id + element_start;
  auto values_ = values + vector_start*blockSize_h;
  auto sparse_AToB = sparseatob + vector_start;
  for (unsigned e_index = element_start; e_index < element_end; e_index++) {
    unsigned col = edgeToColumn[e_index]; // new col
    unsigned tcblock_id = col / blockSize_w;
    unsigned row_local = edgeToRow[e_index] % blockSize_h;
    unsigned col_local = col % blockSize_w;

	//如果存在， 且元素在residue里，需要按每行residue偏移
	values_[tcblock_id*blockSize_h*blockSize_w + row_local*blockSize_w + col_local] = __float2half(1.0);
	
	sparse_AToB[tcblock_id * blockSize_w + col_local] = edgeList[e_index];
    // pos_ptr[tcblock_id]++;
  }
}

void generate_tcoffset_id_atob_cuda_fs_ori(int *nodePointer, int *rowwindow_offset,
                                    int *edgeToColumn, int *edgeToRow,
                                    int *edgeList, half *values, int *sparseatob,
                                    int max_block, int num_nodes,
                                    int blockSize_h, int blockSize_w,
                                    int num_row_windows) {
  int block_size = 1;
  int window_count = num_row_windows;
  generate_tcoffset_id_atob_fs_ori<<<window_count, block_size>>>(
      nodePointer, rowwindow_offset, edgeToColumn, edgeToRow, edgeList,
    values, sparseatob, max_block, num_nodes,
      blockSize_h, blockSize_w, num_row_windows);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}


__global__ void generate_tcoffset_id_atob_fs_balance(
    int *nodePointer, int *rowwindow_offset, int *vectorwindow_offset,
    int *edgeToColumn, int *edgeToRow,
    int *edgeList, half *values,
    int *sparseatob, int max_block, int num_nodes, long blockSize_h,
    int blockSize_w, int num_row_windows,
    int *b_rowwindow_offset_d, int *b_window_row_d, int *b_atomic_d, int part) {

  int tid = threadIdx.x;
  int winId = blockIdx.x; // each warp one window
  int group_offset = rowwindow_offset[winId];

  b_rowwindow_offset_d += group_offset;
  b_window_row_d += group_offset;
  b_atomic_d += group_offset;
  unsigned vector_start = vectorwindow_offset[winId];
  unsigned vector_end = vectorwindow_offset[min(winId + 1, num_row_windows)];
  unsigned num_vector = vector_end - vector_start;
  if (num_vector == 0) {
    return;
  }
  //根据part划分
  int block_num = (num_vector + blockSize_w - 1) / blockSize_w;
  int group_num = (block_num + part - 1) / part;
  if(group_num==1){
    b_rowwindow_offset_d[0] = num_vector;
    b_window_row_d[0] = winId;
    b_atomic_d[0] = 0;
  }
  else{
    for(int i=0; i<(group_num-1); i++)
    {      
      b_rowwindow_offset_d[0] = part*blockSize_w;
      b_window_row_d[0] = winId;
      b_atomic_d[0] = 1;

      b_rowwindow_offset_d++;
      b_window_row_d++;
      b_atomic_d++;
    }

    b_rowwindow_offset_d[0] = num_vector % (part*blockSize_w);
    b_window_row_d[0] = winId;
    b_atomic_d[0] = 1;
  }

  unsigned element_start = nodePointer[winId * blockSize_h];
  unsigned element_end =
      nodePointer[min(int(winId * blockSize_h + blockSize_h), num_nodes)];
  unsigned num_window_edges = element_end - element_start;
  if (num_window_edges == 0) {
    return;
  }
  //开始看每个非零元在block内的偏移了
//   auto tileid = tcblocktile_id + element_start;
  auto values_ = values + vector_start*blockSize_h;
  auto sparse_AToB = sparseatob + vector_start;
  for (unsigned e_index = element_start; e_index < element_end; e_index++) {
    unsigned col = edgeToColumn[e_index]; // new col
    unsigned tcblock_id = col / blockSize_w;
    unsigned row_local = edgeToRow[e_index] % blockSize_h;
    unsigned col_local = col % blockSize_w;

	//如果存在， 且元素在residue里，需要按每行residue偏移
	int residue = num_vector % blockSize_w;
	if(residue>0 & col>=(num_vector-residue)){
		values_[tcblock_id*blockSize_h*blockSize_w + row_local*residue + col_local] = __float2half(1.0);
	}else{
		values_[tcblock_id*blockSize_h*blockSize_w + row_local*blockSize_w + col_local] = __float2half(1.0);
	}
	
	sparse_AToB[tcblock_id * blockSize_w + col_local] = edgeList[e_index];
    // pos_ptr[tcblock_id]++;
  }
}

void generate_tcoffset_id_atob_cuda_fs_balance(int *nodePointer, int *rowwindow_offset, int *vectorwindow_offset,
                                    int *edgeToColumn, int *edgeToRow,
                                    int *edgeList, half *values, int *sparseatob,
                                    int max_block, int num_nodes,
                                    int blockSize_h, int blockSize_w,
                                    int num_row_windows,
                                    int *b_rowwindow_offset_d, int *b_window_row_d, int *b_atomic_d, int part) {
  int block_size = 1;
  int window_count = num_row_windows;
  generate_tcoffset_id_atob_fs_balance<<<window_count, block_size>>>(
      nodePointer, rowwindow_offset, vectorwindow_offset,
      edgeToColumn, edgeToRow, edgeList,
    values, sparseatob, max_block, num_nodes,
      blockSize_h, blockSize_w, num_row_windows,
    b_rowwindow_offset_d, b_window_row_d, b_atomic_d, part);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(error));
    exit(-1);
  }
}

void padding_up_8(int *input, int size) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  roundup_to_multiple_of_eight<<<blocksPerGrid, threadsPerBlock>>>(input, size);
}
void get_padding_tileid(int *ori_offset, uint8_t *ori_tileid,
                        int *padded_offset, uint8_t *padded_tileid, int size) {
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  get_padding_tileid_kernel<<<blocksPerGrid, threadsPerBlock>>>(
      ori_offset, ori_tileid, padded_offset, padded_tileid, size);
}


void print_first_20(const thrust::device_vector<int>& seg, const thrust::device_vector<int>& el, const std::string& label) {
    std::cout << label << "前 20 个值:" << std::endl;
    std::cout << "Seg: ";
    thrust::host_vector<int> host_seg(seg.begin(), seg.begin() + 176);
    for (int i = 0; i < 176; i++) {
        std::cout << host_seg[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "EL: ";
    thrust::host_vector<int> host_el(el.begin(), el.begin() + 176);
    for (int i = 0; i < 176; i++) {
        std::cout << host_el[i] << " ";
    }
    std::cout << std::endl;
}

/*main function*/
std::tuple<torch::Tensor, int, torch::Tensor>
seg_sort_dequ_fs(int *seg, int *edgeLists, int *nodepointer, int *edgetocol,
              int *edgetorow, int *blockpartition, int *block_num, int *vector_num,
              int *rowwindow_offset, int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num) {
	thrust::device_ptr<int> Seg = thrust::device_pointer_cast(seg);
	thrust::device_vector<int> deviceSeg(Seg, Seg + num_edges);
  cudaFree(seg);

	thrust::device_ptr<int> EL = thrust::device_pointer_cast(edgeLists);
	thrust::device_vector<int> deviceEL(EL, EL + num_edges);
	auto begin = thrust::make_zip_iterator(
		thrust::make_tuple(deviceSeg.begin(), deviceEL.begin()));
	auto end = thrust::make_zip_iterator(
		thrust::make_tuple(deviceSeg.end(), deviceEL.end()));

	thrust::sort(thrust::device, begin, end);

	// thrust::device_ptr<int> Counts = thrust::device_pointer_cast(edgeLists);
	// thrust::device_vector<int> deviceCounts(Counts, Counts + num_edges);
	generate_edgetocolumn_cuda_fs(
		nodepointer, edgeLists, thrust::raw_pointer_cast(&deviceEL[0]),
		edgetocol,
		blockpartition, block_num, vector_num, blockSize_h, blockSize_w, num_nodes);

	thrust::device_ptr<int> blockpartition_ptr =
		thrust::device_pointer_cast(blockpartition);
	thrust::device_ptr<int> rowwindow_offset_ptr =
		thrust::device_pointer_cast(rowwindow_offset + 1);
	thrust::device_vector<int> blockpartition_vector(
		blockpartition_ptr, blockpartition_ptr + rowwindow_num);
  cudaFree(blockpartition);
	thrust::inclusive_scan(blockpartition_vector.begin(),
							blockpartition_vector.end(), rowwindow_offset_ptr);
  
	auto options_gpu =
		torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	auto options_gpu_unit8 =
		torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
	thrust::device_ptr<int> bnum_ptr = thrust::device_pointer_cast(block_num);
	thrust::host_vector<int> bnum_vector(bnum_ptr, bnum_ptr + 1);
	int block_counter = bnum_vector[0];

	thrust::device_ptr<int> vnum_ptr = thrust::device_pointer_cast(vector_num);
	thrust::host_vector<int> vnum_vector(vnum_ptr, vnum_ptr + 1);
	long vector_counter = vnum_vector[0];

  	//声明最终的数据结构
  auto values_tensor = torch::zeros({vector_counter*blockSize_h}, torch::kFloat16).to(torch::kCPU);
  auto sparse_AToX_index_tensor = torch::zeros({vector_counter}, torch::kInt32).to(torch::kCPU);

	auto values = reinterpret_cast<half *>(values_tensor.data<at::Half>());
	auto sparse_AToX_index = sparse_AToX_index_tensor.data<int>();

  half *values_d;
  int *sparse_AToX_index_d;

  cudaMalloc(&values_d, (values_tensor.size(0)) * sizeof(half));
  cudaMalloc(&sparse_AToX_index_d, (sparse_AToX_index_tensor.size(0)) * sizeof(int));

  cudaMemcpy(values_d, values , (values_tensor.size(0)) * sizeof(half), cudaMemcpyHostToDevice);
  // cudaMemcpy(sparse_AToX_index_d, sparse_AToX_index , (sparse_AToX_index_tensor.size(0)) * sizeof(int), cudaMemcpyHostToDevice);

	generate_tcoffset_id_atob_cuda_fs(
		nodepointer, rowwindow_offset, edgetocol, edgetorow, edgeLists,
		values_d, sparse_AToX_index_d, 1,
		num_nodes, blockSize_h, blockSize_w, rowwindow_num);

  cudaMemcpy(values, values_d, vector_counter*blockSize_h * sizeof(half), cudaMemcpyDeviceToHost);
  cudaMemcpy(sparse_AToX_index, sparse_AToX_index_d, vector_counter * sizeof(int), cudaMemcpyDeviceToHost);


    cudaFree(values_d);
    cudaFree(sparse_AToX_index_d);

	return std::make_tuple(
						sparse_AToX_index_tensor,
							block_counter,values_tensor);
}




/*main function*/
std::tuple<torch::Tensor, int, torch::Tensor>
seg_sort_dequ_fs_ori(int *seg, int *edgeLists, int *nodepointer, int *edgetocol,
              int *edgetorow, int *blockpartition, int *block_num, int *vector_num,
              int *rowwindow_offset, int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num) {
	thrust::device_ptr<int> Seg = thrust::device_pointer_cast(seg);
	thrust::device_vector<int> deviceSeg(Seg, Seg + num_edges);
  cudaFree(seg);

	thrust::device_ptr<int> EL = thrust::device_pointer_cast(edgeLists);
	thrust::device_vector<int> deviceEL(EL, EL + num_edges);
	auto begin = thrust::make_zip_iterator(
		thrust::make_tuple(deviceSeg.begin(), deviceEL.begin()));
	auto end = thrust::make_zip_iterator(
		thrust::make_tuple(deviceSeg.end(), deviceEL.end()));

	thrust::sort(thrust::device, begin, end);

	// thrust::device_ptr<int> Counts = thrust::device_pointer_cast(edgeLists);
	// thrust::device_vector<int> deviceCounts(Counts, Counts + num_edges);
	generate_edgetocolumn_cuda_fs_ori(
		nodepointer, edgeLists, thrust::raw_pointer_cast(&deviceEL[0]),
		edgetocol,
		blockpartition, block_num, vector_num, blockSize_h, blockSize_w, num_nodes);

	thrust::device_ptr<int> blockpartition_ptr =
		thrust::device_pointer_cast(blockpartition);
	thrust::device_ptr<int> rowwindow_offset_ptr =
		thrust::device_pointer_cast(rowwindow_offset + 1);
	thrust::device_vector<int> blockpartition_vector(
		blockpartition_ptr, blockpartition_ptr + rowwindow_num);
  cudaFree(blockpartition);
	thrust::inclusive_scan(blockpartition_vector.begin(),
							blockpartition_vector.end(), rowwindow_offset_ptr);
  
	auto options_gpu =
		torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	auto options_gpu_unit8 =
		torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
	thrust::device_ptr<int> bnum_ptr = thrust::device_pointer_cast(block_num);
	thrust::host_vector<int> bnum_vector(bnum_ptr, bnum_ptr + 1);
	long block_counter = bnum_vector[0];

	thrust::device_ptr<int> vnum_ptr = thrust::device_pointer_cast(vector_num);
	thrust::host_vector<int> vnum_vector(vnum_ptr, vnum_ptr + 1);
	int vector_counter = vnum_vector[0];

  	//声明最终的数据结构
  auto values_tensor = torch::zeros({block_counter*blockSize_h*blockSize_w}, torch::kFloat16).to(torch::kCPU);
  auto sparse_AToX_index_tensor = torch::full({block_counter * blockSize_w}, -1, torch::kInt32).to(torch::kCPU);

	auto values = reinterpret_cast<half *>(values_tensor.data<at::Half>());
	auto sparse_AToX_index = sparse_AToX_index_tensor.data<int>();

  half *values_d;
  int *sparse_AToX_index_d;

  cudaMalloc(&values_d, (values_tensor.size(0)) * sizeof(half));
  cudaMalloc(&sparse_AToX_index_d, (sparse_AToX_index_tensor.size(0)) * sizeof(int));

  cudaMemcpy(values_d, values , (values_tensor.size(0)) * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(sparse_AToX_index_d, sparse_AToX_index , (sparse_AToX_index_tensor.size(0)) * sizeof(int), cudaMemcpyHostToDevice);


	generate_tcoffset_id_atob_cuda_fs_ori(
		nodepointer, rowwindow_offset, edgetocol, edgetorow, edgeLists,
		values_d, sparse_AToX_index_d, 1,
		num_nodes, blockSize_h, blockSize_w, rowwindow_num);

  cudaMemcpy(values, values_d, vector_counter*blockSize_h * sizeof(half), cudaMemcpyDeviceToHost);
  cudaMemcpy(sparse_AToX_index, sparse_AToX_index_d, vector_counter * sizeof(int), cudaMemcpyDeviceToHost);


    cudaFree(values_d);
    cudaFree(sparse_AToX_index_d);

	return std::make_tuple(
						sparse_AToX_index_tensor,
							block_counter,values_tensor);
}



std::tuple<torch::Tensor, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
seg_sort_dequ_fs_balance(int *seg, int *edgeLists, int *nodepointer, int *edgetocol,
              int *edgetorow, int *blockpartition, int *vectorPartition, int *block_num, int *vector_num,
              int *rowwindow_offset, int * vectorwindow_offset, 
              int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num, int part) {
	thrust::device_ptr<int> Seg = thrust::device_pointer_cast(seg);
	thrust::device_vector<int> deviceSeg(Seg, Seg + num_edges);
  cudaFree(seg);

	thrust::device_ptr<int> EL = thrust::device_pointer_cast(edgeLists);
	thrust::device_vector<int> deviceEL(EL, EL + num_edges);
	auto begin = thrust::make_zip_iterator(
		thrust::make_tuple(deviceSeg.begin(), deviceEL.begin()));
	auto end = thrust::make_zip_iterator(
		thrust::make_tuple(deviceSeg.end(), deviceEL.end()));

	thrust::sort(thrust::device, begin, end);

  //确定每个window需要几个group,且返回后需要累加
  //确定每个window中的vector个数，且返回后不需要累加
	generate_edgetocolumn_cuda_fs_balance(
		nodepointer, edgeLists, thrust::raw_pointer_cast(&deviceEL[0]),
		edgetocol,
		blockpartition, vectorPartition, block_num, vector_num, blockSize_h, blockSize_w, num_nodes, part);

	thrust::device_ptr<int> blockpartition_ptr =
		thrust::device_pointer_cast(blockpartition);
	thrust::device_ptr<int> rowwindow_offset_ptr =
		thrust::device_pointer_cast(rowwindow_offset + 1);
	thrust::device_vector<int> blockpartition_vector(
		blockpartition_ptr, blockpartition_ptr + rowwindow_num);
  cudaFree(blockpartition);
	thrust::inclusive_scan(blockpartition_vector.begin(),
							blockpartition_vector.end(), rowwindow_offset_ptr);

	thrust::device_ptr<int> vectorpartition_ptr =
		thrust::device_pointer_cast(vectorPartition);
	thrust::device_ptr<int> vectorwindow_offset_ptr =
		thrust::device_pointer_cast(vectorwindow_offset + 1);
	thrust::device_vector<int> vectorpartition_vector(
		vectorpartition_ptr, vectorpartition_ptr + rowwindow_num);
  cudaFree(vectorPartition);
	thrust::inclusive_scan(vectorpartition_vector.begin(),
							vectorpartition_vector.end(), vectorwindow_offset_ptr);
  
	auto options_gpu =
		torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
	auto options_gpu_unit8 =
		torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
	thrust::device_ptr<int> bnum_ptr = thrust::device_pointer_cast(block_num);
	thrust::host_vector<int> bnum_vector(bnum_ptr, bnum_ptr + 1);
	int block_counter = bnum_vector[0];
  // printf("%d\n", block_counter);

	thrust::device_ptr<int> vnum_ptr = thrust::device_pointer_cast(vector_num);
	thrust::host_vector<int> vnum_vector(vnum_ptr, vnum_ptr + 1);
	long vector_counter = vnum_vector[0];

  	//声明最终的数据结构
  auto values_tensor = torch::zeros({vector_counter*blockSize_h}, torch::kFloat16).to(torch::kCPU);
  auto sparse_AToX_index_tensor = torch::zeros({vector_counter}, torch::kInt32).to(torch::kCPU);
  //根据block_counter确定b_rowwindow_offset_tensor， b_window_rowTensor， b_atomicTensor
  // auto b_rowwindow_offsetTensor = torch::zeros({block_counter+1}, torch::kInt32).to(torch::kCPU);
  auto b_window_rowTensor = torch::zeros({block_counter}, torch::kInt32).to(torch::kCPU);
  auto b_atomicTensor = torch::zeros({block_counter}, torch::kInt32).to(torch::kCPU);

	auto values = reinterpret_cast<half *>(values_tensor.data<at::Half>());
	auto sparse_AToX_index = sparse_AToX_index_tensor.data<int>();
	// auto b_rowwindow_offset = b_rowwindow_offsetTensor.data<int>();
	auto b_window_row = b_window_rowTensor.data<int>();
	auto b_atomic = b_atomicTensor.data<int>();

  half *values_d;
  int *sparse_AToX_index_d, *b_rowwindow_offset_d, *b_window_row_d, *b_atomic_d;

  cudaMalloc(&values_d, (values_tensor.size(0)) * sizeof(half));
  cudaMalloc(&sparse_AToX_index_d, (sparse_AToX_index_tensor.size(0)) * sizeof(int));
  cudaMalloc(&b_rowwindow_offset_d, (block_counter) * sizeof(int));
  cudaMalloc(&b_window_row_d, block_counter * sizeof(int));
  cudaMalloc(&b_atomic_d, block_counter * sizeof(int));

  cudaMemcpy(values_d, values , (values_tensor.size(0)) * sizeof(half), cudaMemcpyHostToDevice);


	generate_tcoffset_id_atob_cuda_fs_balance(
		nodepointer, rowwindow_offset, vectorwindow_offset, edgetocol, edgetorow, edgeLists,
		values_d, sparse_AToX_index_d, 1,
		num_nodes, blockSize_h, blockSize_w, rowwindow_num,
    b_rowwindow_offset_d, b_window_row_d, b_atomic_d, part);

  cudaMemcpy(values, values_d, vector_counter*blockSize_h * sizeof(half), cudaMemcpyDeviceToHost);
  cudaMemcpy(sparse_AToX_index, sparse_AToX_index_d, vector_counter * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(b_window_row, b_window_row_d, block_counter * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(b_atomic, b_atomic_d, block_counter * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(values_d);
  cudaFree(sparse_AToX_index_d);
  cudaFree(b_window_row_d);
  cudaFree(b_atomic_d);

  //累加
  auto b_rowwindow_offset_outTensor = torch::zeros({block_counter+1}, torch::kInt32).to(torch::kCPU);
  auto b_rowwindow_offset_out = b_rowwindow_offset_outTensor.data<int>();
  int *b_rowwindow_offset_out_d;
  cudaMalloc(&b_rowwindow_offset_out_d, (block_counter+1) * sizeof(int));
  cudaMemset(b_rowwindow_offset_out_d, 0, (block_counter + 1) * sizeof(int));

	thrust::device_ptr<int> b_rowwindow_offset_d_ptr =
		thrust::device_pointer_cast(b_rowwindow_offset_d);
	thrust::device_ptr<int> b_rowwindow_offset_out_d_ptr =
		thrust::device_pointer_cast(b_rowwindow_offset_out_d + 1);
	thrust::device_vector<int> b_rowwindow_offset_d_vector(
		b_rowwindow_offset_d_ptr, b_rowwindow_offset_d_ptr + block_counter);
	thrust::inclusive_scan(b_rowwindow_offset_d_vector.begin(),
							b_rowwindow_offset_d_vector.end(), b_rowwindow_offset_out_d_ptr);
              
  cudaMemcpy(b_rowwindow_offset_out, b_rowwindow_offset_out_d, (block_counter+1) * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(b_rowwindow_offset_d);
  cudaFree(b_rowwindow_offset_out_d);


	return std::make_tuple(
						sparse_AToX_index_tensor,
							block_counter,values_tensor, b_rowwindow_offset_outTensor, b_window_rowTensor, b_atomicTensor);
}