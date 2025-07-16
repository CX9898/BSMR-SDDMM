#include <torch/extension.h>

std::vector<torch::Tensor> blockProcess8_16_csr(torch::Tensor row1, torch::Tensor column1);



//new
std::vector<torch::Tensor> blockProcess_fp16(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1,  int window1, int wide1);

std::vector<torch::Tensor> blockProcess_fp16_ori(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1,  int window1, int wide1);

std::vector<torch::Tensor> blockProcess_tf32(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1,  int window1, int wide1);

std::vector<torch::Tensor> blockProcess_sddmm(torch::Tensor row1, torch::Tensor column1, int window1, int wide1, int partSize_t);

std::vector<torch::Tensor> blockProcess_sddmm_gnn(torch::Tensor row1, torch::Tensor column1, int window1, int wide1, int partSize_t);

std::vector<torch::Tensor> blockProcess_fp16_balance(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1, int window1, int wide1, int partSize_t);

std::vector<torch::Tensor> blockProcess_tf32_balance(torch::Tensor row1, torch::Tensor column1, torch::Tensor degree1, int window1, int wide1, int partSize_t);