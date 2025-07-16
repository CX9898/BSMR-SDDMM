#include "../sddmm_utils/compute_utils.h"
#include "../sddmm_utils/output_tile.h"
#include <stdio.h>
#include <mma.h>
#include <cstdint>
#include <iostream>
#include <torch/torch.h>

/*
FP16
*/

__global__ void sddmm_gen_forward_cuda_kernel_gat(
    const long dimW,
    const int* __restrict__ row_offsets, 
    const int* __restrict__ col_indices,
    const int* __restrict__ values,
    const int* t_window_row,
    const half* __restrict__ l_matrix,
    const half* __restrict__ r_matrix,
    half* __restrict__ output_matrix,
    const int parts_t,
    int warps_block,
    int dimMori,
    int splitk)
{
    long m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=parts_t)
    return;    

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;
    int nonzeros_ori = nonzeros;
    if(nonzeros%16>0) nonzeros += 16 - nonzeros%16;
    int tcu_blocks=nonzeros/16;

    if(tcu_blocks==0) return;

    int id=blockIdx.x*(warps_block)+(threadIdx.x/32);
    if(id>= tcu_blocks) return;

    uint32_t output_fragment[2] = {0,0};
    long col =0;
    long col1 = 0;
    if((id+1)*16 <=  nonzeros_ori){
        col=*(col_indices + row_offset_vec + (id*16) + ((threadIdx.x%32)/4));
        col1=*(col_indices + row_offset_vec + (id*16) + ((threadIdx.x%32)/4) +8);
    }else{
        int temp_col = (id*16) + ((threadIdx.x%32)/4);
        if(temp_col < nonzeros_ori)
        col=*(col_indices + row_offset_vec + (id*16) + ((threadIdx.x%32)/4));
        else col = -1;
        if((temp_col + 8) < nonzeros_ori)
        col1=*(col_indices + row_offset_vec + (id*16) + ((threadIdx.x%32)/4) +8);
        else col1 = -1;
    }
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    long row=cur_m_index_vec*8 + ((threadIdx.x%32)/4);
    if(col>=0) r_matrix+=col*dimW;
    l_matrix+=row*dimW;
    mmaComputeUtils_fp16_gen computer(
        reinterpret_cast<const half2 *>(l_matrix),
        reinterpret_cast<const half2 *>(r_matrix), 
        output_fragment, 
        threadIdx.x);
    int steps = dimW/16;
    // int residue = dimW%16;

    if(steps>0)
        for(int i = 0; i < steps; i++)
            computer.TileMAC(dimW, col, col1, dimMori,row);
    // if(residue>0)
    //     computer.TileMACResidue(residue,dimW, col, col1, dimMori, row);
        // __syncthreads();
        // if(threadIdx.x==4&m_index_vec==1&id==0)
        //     {
        //         half * p=reinterpret_cast<half *>(output_fragment);
        //         printf("thread_id:%d, comoute, value is:%f, %f, %f, %f\n", threadIdx.x,__half2float(*(p)), __half2float(*(p+1)), __half2float(*(p+2)), __half2float(*(p+3)));
        //     }
        //     __syncthreads();

    //需要进行将output_fragment0与output_fragment0进行累加
    //然后再根据values中的-1的地方置为0
    mmaOutputTile_fp16_gen_gnn output_tile_storer(threadIdx.x);
    output_tile_storer.Store(row_offset_vec, output_matrix,
    values,
    reinterpret_cast<half *>(output_fragment), id, col, col1, nonzeros_ori);
}

float sddmm_gen_forward_cuda_gat(
    long dimN, 
    // long dimM, 
    int * row_offsets, 
    int * col_indices,
    int * values,
    int* t_window_row,
    const int parts_t,
    half * lhs_matrix,
    // half * rhs_matrix,
    half * output_matrix,
    // int max_vectors,
    int dimMori,
    int epoches,
    int maxPart)
{
    // auto output_matrix = torch::zeros({8*row_offsets[row_offsets.size(0)-1].item<int>()}, torch::kCUDA).to(torch::kF16);
    int warps = 4;
    int splitk = 0;
    if(parts_t<500000) splitk=8;
    else splitk=((parts_t/1250000)+1)*20;
    dim3 grid_dim(maxPart/warps, splitk ,(parts_t/splitk+1));
    dim3 block_dim(warps*32, 1, 1);
    for(int iter=0; iter<10; ++iter){
        sddmm_gen_forward_cuda_kernel_gat<<<grid_dim, block_dim>>>
        (dimN, 
        row_offsets, 
        col_indices, 
        values, 
        t_window_row,
        lhs_matrix, 
        lhs_matrix, 
        output_matrix, 
        parts_t, warps, dimMori, splitk);
    }
     //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        sddmm_gen_forward_cuda_kernel_gat<<<grid_dim, block_dim>>>
        (dimN, 
        row_offsets, 
        col_indices, 
        values, 
        t_window_row,
        lhs_matrix, 
        lhs_matrix, 
        output_matrix, 
        parts_t, warps, dimMori, splitk);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    return spmm_ms_avg;
}

// 16x1

__global__ void sddmm_gen_forward_cuda_kernel_gat_16(
    const long dimW,
    const int* __restrict__ row_offsets, 
    const int* __restrict__ col_indices,
    const int* __restrict__ values,
    const int* t_window_row,
    const at::Half* __restrict__ l_matrix,
    const at::Half* __restrict__ r_matrix,
    at::Half* __restrict__ output_matrix,
    const int parts_t,
    int warps_block,
    int dimMori,
    int splitk)
{
    long m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=parts_t)
    return;    

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;
    int nonzeros_ori = nonzeros;
    if(nonzeros%8>0) nonzeros += 8 - nonzeros%8;
    int warps=nonzeros/8;
    int blocks=warps/(warps_block)+1;
    if(warps%(warps_block)==0) blocks-=1;
    //根据blockIdx.y以及该当前warp所在的block，当前的warp是否超过稀疏矩阵有效的block数
    if(blockIdx.x >= blocks)
    return;
    int id=blockIdx.x*(warps_block)+(threadIdx.x/32);
    if(id>= warps) return;

    uint32_t output_fragment[2] = {0,0};
    long col_mma = -1;
    if((id*8 + ((threadIdx.x%32)/4)) <  nonzeros_ori){
        col_mma=*(col_indices + row_offset_vec + (id*8) + ((threadIdx.x%32)/4));
    }
    long col =0;
    long col1 = 0;
    if((id+1)*8 <=  nonzeros_ori){
        col=*(col_indices + row_offset_vec + (id*8) + ((threadIdx.x%32)%4)*2);
        col1=*(col_indices + row_offset_vec + (id*8) + ((threadIdx.x%32)%4)*2 +1);
    }else{
        int temp_col = (id*8) + ((threadIdx.x%32)%4)*2;
        if(temp_col < nonzeros_ori)
        col=*(col_indices + row_offset_vec + (id*8) + ((threadIdx.x%32)%4)*2);
        else col = -1;
        if((temp_col + 1) < nonzeros_ori)
        col1=*(col_indices + row_offset_vec + (id*8) + ((threadIdx.x%32)%4)*2 +1);
        else col1 = -1;
    }
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    long row=cur_m_index_vec*16 + ((threadIdx.x%32)/4);
    if(col_mma>=0) r_matrix+=col_mma*dimW;
    l_matrix+=row*dimW;
    mmaComputeUtils_fp16_gen_16 computer(
        l_matrix,
        r_matrix, 
        output_fragment, 
        threadIdx.x);
    int steps = dimW/16;
    // int residue = dimW%16;

    if(steps>0)
        for(int i = 0; i < steps; i++)
            computer.TileMAC(dimW, col_mma, dimMori,row);
    // if(residue>0)
    //     computer.TileMACResidue(residue,dimW, col_mma, dimMori, row);
        // __syncthreads();
        // if(threadIdx.x==4&m_index_vec==1&id==0)
        //     {
        //         half * p=reinterpret_cast<half *>(output_fragment);
        //         printf("thread_id:%d, comoute, value is:%f, %f, %f, %f\n", threadIdx.x,__half2float(*(p)), __half2float(*(p+1)), __half2float(*(p+2)), __half2float(*(p+3)));
        //     }
        //     __syncthreads();

    //需要进行将output_fragment0与output_fragment0进行累加
    //然后再根据values中的-1的地方置为0
    mmaOutputTile_fp16_gen_16 output_tile_storer(threadIdx.x);
    output_tile_storer.Store(row_offset_vec, output_matrix,
    values,
    reinterpret_cast<at::Half *>(output_fragment), id, col, col1, nonzeros_ori);
}

float sddmm_gen_forward_cuda_gat_16(
    long dimN, 
    // long dimM, 
    int * row_offsets, 
    int * col_indices,
    int * values,
    int* t_window_row,
    const int parts_t,
    half * lhs_matrix,
    // half * rhs_matrix,
    half * output_matrix,
    // int max_vectors,
    int dimMori,
    int epoches,
    int maxPart)
{
    // auto output_matrix = torch::zeros({8*row_offsets[row_offsets.size(0)-1].item<int>()}, torch::kCUDA).to(torch::kF16);
    int warps = 4;
    int splitk = 0;
    if(parts_t<500000) splitk=8;
    else splitk=((parts_t/1250000)+1)*20;
    dim3 grid_dim(maxPart/warps, splitk ,(parts_t/splitk+1));
    dim3 block_dim(warps*32, 1, 1);
    for(int iter=0; iter<10; ++iter){
        sddmm_gen_forward_cuda_kernel_gat_16<<<grid_dim, block_dim>>>
        (dimN, 
        row_offsets, 
        col_indices, 
        values, 
        t_window_row,
        reinterpret_cast<const at::Half*>(lhs_matrix), 
        reinterpret_cast<const at::Half*>(lhs_matrix), 
        reinterpret_cast< at::Half*>(output_matrix), 
        parts_t, warps, dimMori, splitk);
    }
     //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        sddmm_gen_forward_cuda_kernel_gat_16<<<grid_dim, block_dim>>>
        (dimN, 
        row_offsets, 
        col_indices, 
        values, 
        t_window_row,
        reinterpret_cast<const at::Half*>(lhs_matrix), 
        reinterpret_cast<const at::Half*>(lhs_matrix), 
        reinterpret_cast< at::Half*>(output_matrix), 
        parts_t, warps, dimMori, splitk);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    return spmm_ms_avg;
}



/*
TF32
*/

__global__ void sddmm_gen_forward_cuda_kernel_gat_tf32(
    const long dimW,
    const int* __restrict__ row_offsets, 
    const int* __restrict__ col_indices,
    const int* __restrict__ values,
    const int* t_window_row,
    const float* __restrict__ l_matrix,
    const float* __restrict__ r_matrix,
    float* __restrict__ output_matrix,
    const int parts_t,
    int warps_block,
    int dimMori,
    int splitk)
{
    long m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=parts_t)
    return;    

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;
    int nonzeros_ori = nonzeros;
    if(nonzeros%16>0) nonzeros += 16 - nonzeros%16;
    int tcu_blocks=nonzeros/16;

    if(tcu_blocks==0) return;

    int id=blockIdx.x*(warps_block)+(threadIdx.x/32);
    if(id>= tcu_blocks) return;   



    float output_fragment[4] = {0,0,0,0};
    long col =0;
    long col1 = 0;
    if((id+1)*16 <=  nonzeros_ori){
        col=*(col_indices + row_offset_vec + (id*16) + ((threadIdx.x%32)/4));
        col1=*(col_indices + row_offset_vec + (id*16) + ((threadIdx.x%32)/4) +8);
    }else{
        int temp_col = (id*16) + ((threadIdx.x%32)/4);
        if(temp_col < nonzeros_ori)
        col=*(col_indices + row_offset_vec + (id*16) + ((threadIdx.x%32)/4));
        else col = -1;
        if((temp_col + 8) < nonzeros_ori)
        col1=*(col_indices + row_offset_vec + (id*16) + ((threadIdx.x%32)/4) +8);
        else col1 = -1;
    }
    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    long row=cur_m_index_vec*8 + ((threadIdx.x%32)/4);
    if(col>=0) r_matrix+=col*dimW;
    l_matrix+=row*dimW;
    mmaComputeUtils_tf32_gen computer(
        l_matrix,
        r_matrix, 
        output_fragment, 
        threadIdx.x);
    int steps = dimW/8;
    // int residue = dimW%8;

    if(steps>0)
        for(int i = 0; i < steps; i++){
            computer.TileMAC(dimW, col, col1, dimMori,row);

            // if(blockIdx.x==0 and m_index_vec==(dimM-3) and threadIdx.x==0)
            // printf("%d\n", i);
        }
    
    // if(residue>0)
    //     computer.TileMACResidue(residue,dimW, col, col1, dimMori, row);
        // __syncthreads();
        // if(threadIdx.x==4&m_index_vec==1&id==0)
        //     {
        //         half * p=reinterpret_cast<half *>(output_fragment);
        //         printf("thread_id:%d, comoute, value is:%f, %f, %f, %f\n", threadIdx.x,__half2float(*(p)), __half2float(*(p+1)), __half2float(*(p+2)), __half2float(*(p+3)));
        //     }
        //     __syncthreads();

    //需要进行将output_fragment0与output_fragment0进行累加
    //然后再根据values中的-1的地方置为0
    mmaOutputTile_tf32_gen_gnn output_tile_storer(threadIdx.x);
    output_tile_storer.Store(row_offset_vec, output_matrix,
    values,
    output_fragment, id, col, col1, nonzeros_ori);
}

float sddmm_gen_forward_cuda_gat_tf32(
    long dimN, 
    // long dimM, 
    int * row_offsets, 
    int * col_indices,
    int *  values,
    int* t_window_row,
    const int parts_t,
    float *  lhs_matrix,
    // float *  rhs_matrix,
    float * output_matrix,
    // int max_vectors,
    int dimMori,
    int epoches,
    int maxPart)
{
    // auto output_matrix = torch::zeros({8*row_offsets[row_offsets.size(0)-1].item<int>()}, torch::kCUDA).to(torch::kF32);
    int warps = 4;
    int splitk = 0;
    if(parts_t<500000) splitk=8;
    else splitk=((parts_t/1250000)+1)*20;
    dim3 grid_dim(maxPart/warps, splitk ,(parts_t/splitk+1));
    dim3 block_dim(warps*32, 1, 1);
    for(int iter=0; iter<10; ++iter){
        sddmm_gen_forward_cuda_kernel_gat_tf32<<<grid_dim, block_dim>>>
        (dimN, 
        row_offsets, 
        col_indices, 
        values, 
        t_window_row,
        lhs_matrix, 
        lhs_matrix, 
        output_matrix, 
        parts_t, warps, dimMori, splitk);
    }
         //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        sddmm_gen_forward_cuda_kernel_gat_tf32<<<grid_dim, block_dim>>>
        (dimN, 
        row_offsets, 
        col_indices, 
        values, 
        t_window_row,
        lhs_matrix, 
        lhs_matrix, 
        output_matrix, 
        parts_t, warps, dimMori, splitk);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    return spmm_ms_avg;
}

// 16x1 tf 32 m16n16
__global__ void sddmm_gen_forward_cuda_kernel_gat_tf32_16(
    const long dimW,
    const int* __restrict__ row_offsets, 
    const int* __restrict__ col_indices,
    const int* __restrict__ values,
    const int* t_window_row,
    const float* __restrict__ l_matrix,
    const float* __restrict__ r_matrix,
    float* __restrict__ output_matrix,
    const int parts_t,
    int warps_block,
    int dimMori,
    int splitk)
{
    long m_index_vec = (blockIdx.z*splitk)+blockIdx.y;
    if(m_index_vec>=parts_t)
    return;    

    // Load the row offset and calculate the number of nonzeros in the row
    int row_offset_vec = __ldg(row_offsets + m_index_vec);
    int nonzeros = __ldg(row_offsets + m_index_vec + 1) - row_offset_vec;
    int nonzeros_ori = nonzeros;
    if(nonzeros%8>0) nonzeros += 8 - nonzeros%8;
    int warps=nonzeros/8;
    int blocks=warps/(warps_block)+1;
    if(warps%(warps_block)==0) blocks-=1;
    //根据blockIdx.y以及该当前warp所在的block，当前的warp是否超过稀疏矩阵有效的block数
    if(blockIdx.x >= blocks)
    return;
    int id=blockIdx.x*(warps_block)+(threadIdx.x/32);
    if(id>= warps) return;

    float output_fragment[4] = {0,0,0,0};
    long col_mma = -1;
    if((id*8 + ((threadIdx.x%32)/4)) <  nonzeros_ori){
        col_mma=*(col_indices + row_offset_vec + (id*8) + ((threadIdx.x%32)/4));
    }
    long col =0;
    long col1 = 0;
    if((id+1)*8 <=  nonzeros_ori){
        col=*(col_indices + row_offset_vec + (id*8) + ((threadIdx.x%32)%4)*2);
        col1=*(col_indices + row_offset_vec + (id*8) + ((threadIdx.x%32)%4)*2 +1);
    }else{
        int temp_col = (id*8) + ((threadIdx.x%32)%4)*2;
        if(temp_col < nonzeros_ori)
        col=*(col_indices + row_offset_vec + (id*8) + ((threadIdx.x%32)%4)*2);
        else col = -1;
        if((temp_col + 1) < nonzeros_ori)
        col1=*(col_indices + row_offset_vec + (id*8) + ((threadIdx.x%32)%4)*2 +1);
        else col1 = -1;
    }

    int cur_m_index_vec = __ldg(t_window_row + m_index_vec);
    long row=cur_m_index_vec*16 + ((threadIdx.x%32)/4);
    if(col_mma>=0) r_matrix+=col_mma*dimW;
    l_matrix+=row*dimW;
    mmaComputeUtils_tf32_gen_16 computer(
        l_matrix,
        r_matrix, 
        output_fragment, 
        threadIdx.x);
    int steps = dimW/8;

    if(steps>0)
        for(int i = 0; i < steps; i++)
            computer.TileMAC(dimW, col_mma, dimMori,row);


    //需要进行将output_fragment0与output_fragment0进行累加
    //然后再根据values中的-1的地方置为0
    mmaOutputTile_tf32_gen_16 output_tile_storer(threadIdx.x);
    output_tile_storer.Store(row_offset_vec, output_matrix,
    values,
    output_fragment, id);
}

float sddmm_gen_forward_cuda_gat_tf32_16(
    const long dimN, 
    // const long dimM,
    int * row_offsets, 
    int * col_indices,
    int *  values,
    int* t_window_row,
    const int parts_t,
    float *  lhs_matrix,
    // float *  rhs_matrix,
    float * output_matrix,
    // int max_vectors,
    int dimMori,
    int epoches,
    int maxPart)
{
    int warps = 4;
    int splitk = 0;
    if(parts_t<500000) splitk=8;
    else splitk=((parts_t/1250000)+1)*20;
    dim3 grid_dim(maxPart/warps, splitk ,(parts_t/splitk+1));
    dim3 block_dim(warps*32, 1, 1);
    for(int iter=0; iter<10; ++iter){
        sddmm_gen_forward_cuda_kernel_gat_tf32_16<<<grid_dim, block_dim>>>
        (dimN, 
        row_offsets, 
        col_indices, 
        values, 
        t_window_row,
        lhs_matrix, 
        lhs_matrix, 
        output_matrix, 
        parts_t, warps, dimMori, splitk);
    }
         //测试kernel
    float spmm_ms_avg = 0.0f;
    float spmm_ms = 0.0f;
    cudaEvent_t spmm_start;
    cudaEvent_t spmm_end;
    cudaEventCreate(&spmm_start);
    cudaEventCreate(&spmm_end);
    cudaEventRecord(spmm_start);
    for(int iter=0; iter<epoches; ++iter){
        sddmm_gen_forward_cuda_kernel_gat_tf32_16<<<grid_dim, block_dim>>>
        (dimN, 
        row_offsets, 
        col_indices, 
        values, 
        t_window_row,
        lhs_matrix, 
        lhs_matrix, 
        output_matrix, 
        parts_t, warps, dimMori, splitk);
    }
    cudaEventRecord(spmm_end);
    cudaEventSynchronize(spmm_end);
    cudaEventElapsedTime(&spmm_ms, spmm_start, spmm_end);
    cudaEventDestroy(spmm_start);
    cudaEventDestroy(spmm_end);

    //计算时间 ms
    spmm_ms_avg = spmm_ms/(float)epoches;
    return spmm_ms_avg;
}

//gnn
void sddmm_gen_forward_cuda_gat_gnn(
    long dimN, 
    // long dimM, 
    int * row_offsets, 
    int * col_indices,
    int * values,
    int* t_window_row,
    const int parts_t,
    half * lhs_matrix,
    half * rhs_matrix,
    half * output_matrix,
    // int max_vectors,
    int dimMori,
    int maxPart)
{
    // auto output_matrix = torch::zeros({8*row_offsets[row_offsets.size(0)-1].item<int>()}, torch::kCUDA).to(torch::kF16);
    int warps = 4;
    int splitk = 0;
    if(parts_t<500000) splitk=8;
    else splitk=((parts_t/1250000)+1)*20;
    dim3 grid_dim(maxPart/warps, splitk ,(parts_t/splitk+1));
    dim3 block_dim(warps*32, 1, 1);

    sddmm_gen_forward_cuda_kernel_gat<<<grid_dim, block_dim>>>
    (dimN, 
    row_offsets, 
    col_indices, 
    values, 
    t_window_row,
    lhs_matrix, 
    rhs_matrix, 
    output_matrix,
    parts_t, warps, dimMori, splitk);

}


void sddmm_gen_forward_cuda_gat_tf32_gnn(
    long dimN, 
    // long dimM, 
    int * row_offsets, 
    int * col_indices,
    int *  values,
    int* t_window_row,
    const int parts_t,
    float *  lhs_matrix,
    float *  rhs_matrix,
    float * output_matrix,
    // int max_vectors,
    int dimMori,
    int maxPart)
{
    // auto output_matrix = torch::zeros({8*row_offsets[row_offsets.size(0)-1].item<int>()}, torch::kCUDA).to(torch::kF32);
    int warps = 4;
    int splitk = 0;
    if(parts_t<500000) splitk=8;
    else splitk=((parts_t/1250000)+1)*20;
    dim3 grid_dim(maxPart/warps, splitk ,(parts_t/splitk+1));
    dim3 block_dim(warps*32, 1, 1);

        sddmm_gen_forward_cuda_kernel_gat_tf32<<<grid_dim, block_dim>>>
        (dimN, 
        row_offsets, 
        col_indices, 
        values, 
        t_window_row,
        lhs_matrix, 
        rhs_matrix, 
        output_matrix, 
        parts_t, warps, dimMori, splitk);

}