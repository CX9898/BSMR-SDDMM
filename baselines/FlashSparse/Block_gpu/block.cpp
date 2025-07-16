#include <torch/extension.h>
#include <vector>
#include <string.h>
#include <cstdlib>
#include <map>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <chrono>
#include <thrust/device_vector.h>
#include <thrust/unique.h>
#define min(x, y) (((x) < (y))? (x) : (y))

#include <cuda_fp16.h>
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

void fill_edgeToRow_cuda(int *edgeToRow, int *nodePointer, int num_nodes);

void fill_window_cuda(int *edgeToColumn, int *blockPartition, int *nodePointer,
                      int *edgeList, int blockSize_h, int blockSize_w,
                      int num_nodes);

void fill_segment_cuda(int *nodePointer, int *seg_out, int blockSize_h,
                       int blockSize_w, int num_nodes);


std::tuple<torch::Tensor, int, torch::Tensor>
seg_sort_dequ_fs(int *seg, int *edgeLists, int *nodepointer, int *edgetocol,
              int *edgetorow, int *blockPartition, int *blocknum, int * vectornum,
              int *row_window_offset, int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num);

std::tuple<torch::Tensor, int, torch::Tensor>
seg_sort_dequ_fs_ori(int *seg, int *edgeLists, int *nodepointer, int *edgetocol,
              int *edgetorow, int *blockPartition, int *blocknum, int * vectornum,
              int *row_window_offset, int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num);

std::tuple<torch::Tensor, int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
seg_sort_dequ_fs_balance(int *seg, int *edgeLists, int *nodepointer, int *edgetocol,
              int *edgetorow, int *blockPartition, int *vectorPartition, int *blocknum, int * vectornum,
              int *row_window_offset, int *vector_window_offset, int blockSize_h, int blockSize_w,
              int num_nodes, int num_edges, int rowwindow_num, int part);

#define CHECK_CUDA(x)                                                          \
  TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, float>
preprocess_gpu_fs( torch::Tensor nodePointer_tensor, torch::Tensor edgeList_tensor,
               int num_nodes, int num_edges, int blockSize_h, int blockSize_w) {
  // input tensors.
  auto options_gpu =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto options_gpu_unit8 =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  int window_num = num_nodes/blockSize_h;
  // auto blockPartition_tensor = torch::zeros({window_num}, torch::kInt32).to(torch::kCPU);
  auto row_window_offset_tensor = torch::zeros({window_num+1}, torch::kInt32).to(torch::kCPU);
  // auto edgeToColumn_tensor = torch::zeros({num_edges}, torch::kInt32).to(torch::kCPU);
  // auto edgeToRow_tensor = torch::zeros({num_edges}, torch::kInt32).to(torch::kCPU);
  auto seg_out_tensor = torch::zeros({edgeList_tensor.size(0)}, torch::kInt32).to(torch::kCPU);

  auto edgeList = edgeList_tensor.data<int>();
  // auto blockPartition = blockPartition_tensor.data<int>();
  auto row_window_offset = row_window_offset_tensor.data<int>();
  // auto edgeToColumn = edgeToColumn_tensor.data<int>();
  // auto edgeToRow = edgeToRow_tensor.data<int>();
  auto nodePointer = nodePointer_tensor.data<int>();
  auto seg_out = seg_out_tensor.data<int>();

int *edgeList_d, *blockPartition_d, *row_window_offset_d, *edgeToColumn_d, *edgeToRow_d, *nodePointer_d, *seg_out_d;

// 分配 GPU 内存
cudaMalloc(&edgeList_d, (edgeList_tensor.size(0)) * sizeof(int));
cudaMalloc(&blockPartition_d, (window_num) * sizeof(int));
cudaMalloc(&row_window_offset_d, (window_num+1) * sizeof(int));
cudaMalloc(&edgeToColumn_d, (num_edges) * sizeof(int));
cudaMalloc(&edgeToRow_d, (num_edges) * sizeof(int));
cudaMalloc(&nodePointer_d, (nodePointer_tensor.size(0)) * sizeof(int));
cudaMalloc(&seg_out_d, (seg_out_tensor.size(0)) * sizeof(int));

  cudaMemcpy(edgeList_d, edgeList , (edgeList_tensor.size(0)) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(nodePointer_d, nodePointer , (nodePointer_tensor.size(0)) * sizeof(int), cudaMemcpyHostToDevice);

  //#NNZ
  auto blocknum = torch::zeros({1}, options_gpu);
  auto vectornum = torch::zeros({1}, options_gpu);
  auto block_num = blocknum.data<int>();
  auto vector_num = vectornum.data<int>();
  auto start = std::chrono::high_resolution_clock::now();
  fill_edgeToRow_cuda(edgeToRow_d, nodePointer_d, num_nodes);
  int block_counter = 0;

  fill_segment_cuda(nodePointer_d, seg_out_d, blockSize_h, blockSize_w, num_nodes);

  auto tuple_tensor_blockcnt = seg_sort_dequ_fs(
      seg_out_d, edgeList_d, nodePointer_d, edgeToColumn_d, edgeToRow_d, blockPartition_d,
      block_num, vector_num, row_window_offset_d, blockSize_h, blockSize_w, num_nodes,
      edgeList_tensor.size(0), window_num);
      
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "\t GPU Preprocess time: " << elapsed_seconds.count()
            << " seconds\n";

  auto sparse_AToX_index_tensor = std::get<0>(tuple_tensor_blockcnt);
  block_counter = std::get<1>(tuple_tensor_blockcnt);
  auto tcvalues_tensor = std::get<2>(tuple_tensor_blockcnt);
  // printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter,
  //        block_counter * 8 * 16);

float elapsed_time = static_cast<float>(elapsed_seconds.count());

  cudaMemcpy(row_window_offset, row_window_offset_d, (window_num+1)* sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(edgeList_d);
    // cudaFree(blockPartition_d);
    cudaFree(row_window_offset_d);
    cudaFree(edgeToColumn_d);
    cudaFree(edgeToRow_d);
    cudaFree(nodePointer_d);
    // cudaFree(seg_out_d);

  return std::make_tuple(row_window_offset_tensor,
                         sparse_AToX_index_tensor, tcvalues_tensor, elapsed_time*1000);
}

//ori

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, float>
preprocess_gpu_fs_ori( torch::Tensor nodePointer_tensor, torch::Tensor edgeList_tensor,
               int num_nodes, int num_edges, int blockSize_h, int blockSize_w) {
  // input tensors.
  auto options_gpu =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto options_gpu_unit8 =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  int window_num = num_nodes/blockSize_h;
  // auto blockPartition_tensor = torch::zeros({window_num}, torch::kInt32).to(torch::kCPU);
  auto row_window_offset_tensor = torch::zeros({window_num+1}, torch::kInt32).to(torch::kCPU);
  // auto edgeToColumn_tensor = torch::zeros({num_edges}, torch::kInt32).to(torch::kCPU);
  // auto edgeToRow_tensor = torch::zeros({num_edges}, torch::kInt32).to(torch::kCPU);
  auto seg_out_tensor = torch::zeros({edgeList_tensor.size(0)}, torch::kInt32).to(torch::kCPU);

  auto edgeList = edgeList_tensor.data<int>();
  // auto blockPartition = blockPartition_tensor.data<int>();
  auto row_window_offset = row_window_offset_tensor.data<int>();
  // auto edgeToColumn = edgeToColumn_tensor.data<int>();
  // auto edgeToRow = edgeToRow_tensor.data<int>();
  auto nodePointer = nodePointer_tensor.data<int>();
  auto seg_out = seg_out_tensor.data<int>();

int *edgeList_d, *blockPartition_d, *row_window_offset_d, *edgeToColumn_d, *edgeToRow_d, *nodePointer_d, *seg_out_d;

// 分配 GPU 内存
cudaMalloc(&edgeList_d, (edgeList_tensor.size(0)) * sizeof(int));
cudaMalloc(&blockPartition_d, (window_num) * sizeof(int));
cudaMalloc(&row_window_offset_d, (window_num+1) * sizeof(int));
cudaMalloc(&edgeToColumn_d, (num_edges) * sizeof(int));
cudaMalloc(&edgeToRow_d, (num_edges) * sizeof(int));
cudaMalloc(&nodePointer_d, (nodePointer_tensor.size(0)) * sizeof(int));
cudaMalloc(&seg_out_d, (seg_out_tensor.size(0)) * sizeof(int));

  cudaMemcpy(edgeList_d, edgeList , (edgeList_tensor.size(0)) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(nodePointer_d, nodePointer , (nodePointer_tensor.size(0)) * sizeof(int), cudaMemcpyHostToDevice);

  //#NNZ
  auto blocknum = torch::zeros({1}, options_gpu);
  auto vectornum = torch::zeros({1}, options_gpu);
  auto block_num = blocknum.data<int>();
  auto vector_num = vectornum.data<int>();
  auto start = std::chrono::high_resolution_clock::now();
  fill_edgeToRow_cuda(edgeToRow_d, nodePointer_d, num_nodes);
  int block_counter = 0;

  fill_segment_cuda(nodePointer_d, seg_out_d, blockSize_h, blockSize_w, num_nodes);

  auto tuple_tensor_blockcnt = seg_sort_dequ_fs_ori(
      seg_out_d, edgeList_d, nodePointer_d, edgeToColumn_d, edgeToRow_d, blockPartition_d,
      block_num, vector_num, row_window_offset_d, blockSize_h, blockSize_w, num_nodes,
      edgeList_tensor.size(0), window_num);
      
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::cout << "\t GPU Preprocess time: " << elapsed_seconds.count()
            << " seconds\n";

  auto sparse_AToX_index_tensor = std::get<0>(tuple_tensor_blockcnt);
  block_counter = std::get<1>(tuple_tensor_blockcnt);
  auto tcvalues_tensor = std::get<2>(tuple_tensor_blockcnt);
  // printf("TC_Blocks:\t%d\nExp_Edges:\t%d\n", block_counter,
  //        block_counter * 8 * 16);

float elapsed_time = static_cast<float>(elapsed_seconds.count());

  cudaMemcpy(row_window_offset, row_window_offset_d, (window_num+1)* sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(edgeList_d);
    // cudaFree(blockPartition_d);
    cudaFree(row_window_offset_d);
    cudaFree(edgeToColumn_d);
    cudaFree(edgeToRow_d);
    cudaFree(nodePointer_d);
    // cudaFree(seg_out_d);

  return std::make_tuple(row_window_offset_tensor,
                         sparse_AToX_index_tensor, tcvalues_tensor, elapsed_time*1000);
}

//balance

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,float>
preprocess_gpu_fs_balance( torch::Tensor nodePointer_tensor, torch::Tensor edgeList_tensor,
               int num_nodes, int num_edges, int blockSize_h, int blockSize_w, int part) {
  // input tensors.
  auto options_gpu =
      torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto options_gpu_unit8 =
      torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  int window_num = num_nodes/blockSize_h;
  // auto blockPartition_tensor = torch::zeros({window_num}, torch::kInt32).to(torch::kCPU);
  // auto vectorPartition_tensor = torch::zeros({window_num}, torch::kInt32).to(torch::kCPU);
  // auto row_window_offset_tensor = torch::zeros({window_num}, torch::kInt32).to(torch::kCPU);
  // auto edgeToColumn_tensor = torch::zeros({num_edges}, torch::kInt32).to(torch::kCPU);
  // auto edgeToRow_tensor = torch::zeros({num_edges}, torch::kInt32).to(torch::kCPU);
  auto seg_out_tensor = torch::zeros({edgeList_tensor.size(0)}, torch::kInt32).to(torch::kCPU);

  auto edgeList = edgeList_tensor.data<int>();
  // auto blockPartition = blockPartition_tensor.data<int>();
  // auto vectorPartition = vectorPartition_tensor.data<int>();
  // auto row_window_offset = row_window_offset_tensor.data<int>();
  // auto edgeToColumn = edgeToColumn_tensor.data<int>();
  // auto edgeToRow = edgeToRow_tensor.data<int>();
  auto nodePointer = nodePointer_tensor.data<int>();
  auto seg_out = seg_out_tensor.data<int>();

int *edgeList_d, *blockPartition_d, *vectorPartition_d, 
*row_window_offset_d, *vector_window_offset_d,
*edgeToColumn_d, *edgeToRow_d, *nodePointer_d, *seg_out_d;

// 分配 GPU 内存
cudaMalloc(&edgeList_d, (edgeList_tensor.size(0)) * sizeof(int));
cudaMalloc(&blockPartition_d, (window_num) * sizeof(int));
cudaMalloc(&vectorPartition_d, (window_num) * sizeof(int));
cudaMalloc(&row_window_offset_d, (window_num+1) * sizeof(int));
cudaMalloc(&vector_window_offset_d, (window_num+1) * sizeof(int));
cudaMalloc(&edgeToColumn_d, num_edges * sizeof(int));
cudaMalloc(&edgeToRow_d, num_edges * sizeof(int));
cudaMalloc(&nodePointer_d, (nodePointer_tensor.size(0)) * sizeof(int));
cudaMalloc(&seg_out_d, (seg_out_tensor.size(0)) * sizeof(int));

cudaMemcpy(edgeList_d, edgeList , (edgeList_tensor.size(0)) * sizeof(int), cudaMemcpyHostToDevice);
cudaMemcpy(nodePointer_d, nodePointer , (nodePointer_tensor.size(0)) * sizeof(int), cudaMemcpyHostToDevice);

//#NNZ
auto blocknum = torch::zeros({1}, options_gpu);
auto vectornum = torch::zeros({1}, options_gpu);
auto block_num = blocknum.data<int>();
auto vector_num = vectornum.data<int>();
auto start = std::chrono::high_resolution_clock::now();
fill_edgeToRow_cuda(edgeToRow_d, nodePointer_d, num_nodes);
int block_counter = 0;

fill_segment_cuda(nodePointer_d, seg_out_d, blockSize_h, blockSize_w, num_nodes);

auto tuple_tensor_blockcnt = seg_sort_dequ_fs_balance(
    seg_out_d, edgeList_d, nodePointer_d, edgeToColumn_d, edgeToRow_d, blockPartition_d, vectorPartition_d,
    block_num, vector_num, row_window_offset_d, vector_window_offset_d, blockSize_h, blockSize_w, num_nodes,
    edgeList_tensor.size(0), window_num, part);
    
auto end = std::chrono::high_resolution_clock::now();
std::chrono::duration<double> elapsed_seconds = end - start;
std::cout << "\t GPU Preprocess time: " << elapsed_seconds.count()
          << " seconds\n";

auto sparse_AToX_index_tensor = std::get<0>(tuple_tensor_blockcnt);
block_counter = std::get<1>(tuple_tensor_blockcnt);
auto tcvalues_tensor = std::get<2>(tuple_tensor_blockcnt);
auto b_rowwindow_offsetTensor = std::get<3>(tuple_tensor_blockcnt);
auto b_window_rowTensor = std::get<4>(tuple_tensor_blockcnt);
auto b_atomicTensor = std::get<5>(tuple_tensor_blockcnt);

float elapsed_time = static_cast<float>(elapsed_seconds.count());


    cudaFree(edgeList_d);
    cudaFree(row_window_offset_d);
    cudaFree(vector_window_offset_d);
    cudaFree(edgeToColumn_d);
    cudaFree(edgeToRow_d);
    cudaFree(nodePointer_d);
    // cudaFree(seg_out_d);

  return std::make_tuple(b_rowwindow_offsetTensor,
                         sparse_AToX_index_tensor, tcvalues_tensor, 
                         b_window_rowTensor, b_atomicTensor, elapsed_time*1000);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("preprocess_gpu_fs", &preprocess_gpu_fs, "Preprocess Step on (CUDA)");

  m.def("preprocess_gpu_fs_ori", &preprocess_gpu_fs_ori, "Preprocess Step on (CUDA)");

  m.def("preprocess_gpu_fs_balance", &preprocess_gpu_fs_balance, "Preprocess Step on (CUDA)");
}