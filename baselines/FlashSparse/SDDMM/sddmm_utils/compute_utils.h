#include <mma.h>
#include <cstdint>
#include <cuda_fp16.h>

#include <torch/torch.h>

/*
FP16
*/

// struct mmaComputeUtils_fp16_gen{
//     const at::Half* lhs_tile_;
//     const at::Half* rhs_tile_;
//     uint32_t* output_fragment_;
//     const int lane_id_;
//     const int warpin_id;
//     const int warp_id;
//     int dimN_index_;
//     int col_;
//     int row_;
//     int dimW_;

//     // Constructor
//     __device__ __forceinline__ mmaComputeUtils_fp16_gen(
//         const at::Half *lhs_tile,
//         const at::Half *rhs_tile,
//         uint32_t* output_fragment,
//         int lane_id):
//             lhs_tile_(lhs_tile),
//             rhs_tile_(rhs_tile),
//             output_fragment_(output_fragment),
//             lane_id_(lane_id),
//             warpin_id(lane_id & 31),
//             warp_id(lane_id>>5){}
    
//     // Compute
//     __device__ __forceinline__ void TileMAC(int dimW, long col, long col1,int dimMori, long row){
//         //load lhs_matrix
//         at::Half lhs_fragment_half[4]={0.0,0.0,0.0,0.0};
//         uint32_t *lhs_fragment=reinterpret_cast<uint32_t*>(lhs_fragment_half);
//         if(row < dimMori){
//         lhs_fragment_half[0]=*(lhs_tile_+((warpin_id%4)*2));
//         lhs_fragment_half[1]=*(lhs_tile_+((warpin_id%4)*2)+1);
//         lhs_fragment_half[2]=*(lhs_tile_+((warpin_id%4)*2)+8);
//         lhs_fragment_half[3]=*(lhs_tile_+((warpin_id%4)*2)+9);
//         lhs_tile_=lhs_tile_+16;}

//         //load rhs_matrix
//         at::Half rhs_fragment_half[8]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
//         uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_half);
//         if(col>=0){
//         rhs_fragment_half[0]=*(rhs_tile_ + ((warpin_id%4)*2));
//         rhs_fragment_half[1]=*(rhs_tile_ + ((warpin_id%4)*2) + 1);
//         rhs_fragment_half[4]=*(rhs_tile_ + ((warpin_id%4)*2) + 8);
//         rhs_fragment_half[5]=*(rhs_tile_ + ((warpin_id%4)*2) + 9);
//         }
//         if((col1)>=0){
//         rhs_fragment_half[2]=*(rhs_tile_ + ((warpin_id%4)*2) + ((col1-col)*dimW));
//         rhs_fragment_half[3]=*(rhs_tile_ + ((warpin_id%4)*2) + 1 + ((col1-col)*dimW));
//         rhs_fragment_half[6]=*(rhs_tile_ + ((warpin_id%4)*2) + 8 + ((col1-col)*dimW));
//         rhs_fragment_half[7]=*(rhs_tile_ + ((warpin_id%4)*2) + 9 + ((col1-col)*dimW));
//         }
//         //更新rhs_tile
//         rhs_tile_=rhs_tile_+16;
//             // if(warpin_id==4&warp_id==1)
//             // {
//             //     half *p=reinterpret_cast<half *>(rhs_fragment);
//             //     for(int i=0;i<8;i++)
//             //     printf("thread_id:%d, value is:%f\n", warpin_id,__half2float(*(p+i)));
//             // }
     
//         //MMA 
//         asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \t"
//             "{%0,%1}, \t"
//             "{%2,%3,%4,%5}, \t"
//             "{%6,%7}, \t"
//             "{%0,%1}; ":
//             "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
//             "r"(rhs_fragment[0]),  "r"(rhs_fragment[1]),  "r"(rhs_fragment[2]),  "r"(rhs_fragment[3]),
//             "r"(lhs_fragment[0]), "r"(lhs_fragment[1])
//         ); 

//     }
    

//     __device__ __forceinline__ void TileMACResidue(int residue,int dimW, long col, long col1, int dimMori,  long row){
//         //load lhs_matrix
//         at::Half lhs_fragment_half[4]={0.0,0.0,0.0,0.0};
//         uint32_t *lhs_fragment=reinterpret_cast<uint32_t*>(lhs_fragment_half);
//         if(row < dimMori){
//          if((((warpin_id%4)*2))<residue)
//         lhs_fragment_half[0]=*(lhs_tile_+((warpin_id%4)*2));
//         if((((warpin_id%4)*2) + 1)<residue)
//         lhs_fragment_half[1]=*(lhs_tile_+((warpin_id%4)*2) +1);
//         if((((warpin_id%4)*2) + 8)<residue)
//         lhs_fragment_half[2]=*(lhs_tile_+((warpin_id%4)*2) +8);
//         if((((warpin_id%4)*2) + 9)<residue)
//         lhs_fragment_half[3]=*(lhs_tile_+((warpin_id%4)*2) +9);}

//         //load rhs_matrix
//          at::Half rhs_fragment_half[8]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
//         uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_half);
//         if(col>=0){
//             if(((warpin_id%4)*2)<residue)
//             rhs_fragment_half[0]=*(rhs_tile_ +((warpin_id%4)*2));
//             if(((warpin_id%4)*2 +1)<residue)
//             rhs_fragment_half[1]=*(rhs_tile_ +((warpin_id%4)*2) +1);
//             if(((warpin_id%4)*2 +8)<residue)
//             rhs_fragment_half[4]=*(rhs_tile_ +((warpin_id%4)*2) +8);
//             if(((warpin_id%4)*2 +9)<residue)
//             rhs_fragment_half[5]=*(rhs_tile_ +((warpin_id%4)*2) +9);
//         }
//         if((col1)>=0){
//             if(((warpin_id%4)*2)<residue)
//             rhs_fragment_half[2]=*(rhs_tile_ + ((warpin_id%4)*2) + ((col1-col)*dimW));
//             if(((warpin_id%4)*2 +1)<residue)
//             rhs_fragment_half[3]=*(rhs_tile_ + ((warpin_id%4)*2) + 1 + ((col1-col)*dimW));
//             if(((warpin_id%4)*2 +8)<residue)
//             rhs_fragment_half[6]=*(rhs_tile_ + ((warpin_id%4)*2) + 8 + ((col1-col)*dimW));
//             if(((warpin_id%4)*2 +9)<residue)
//             rhs_fragment_half[7]=*(rhs_tile_ + ((warpin_id%4)*2) + 9 + ((col1-col)*dimW));
//         }
//         //MMA 
//         asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \t"
//             "{%0,%1}, \t"
//             "{%2,%3,%4,%5}, \t"
//             "{%6,%7}, \t"
//             "{%0,%1}; ":
//             "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
//             "r"(rhs_fragment[0]),  "r"(rhs_fragment[1]),  "r"(rhs_fragment[2]),  "r"(rhs_fragment[3]),
//             "r"(lhs_fragment[0]), "r"(lhs_fragment[1])
//         );  
//     }

    
// };

/*
FP16
*/

struct mmaComputeUtils_fp16_gen{
    const half2* lhs_tile_;
    const half2* rhs_tile_;
    uint32_t* output_fragment_;
    const int lane_id_;
    const int warpin_id;
    const int warp_id;
    int dimN_index_;
    int col_;
    int row_;
    int dimW_;

    // Constructor
    __device__ __forceinline__ mmaComputeUtils_fp16_gen(
        const half2 *lhs_tile,
        const half2*rhs_tile,
        uint32_t* output_fragment,
        int lane_id):
            lhs_tile_(lhs_tile),
            rhs_tile_(rhs_tile),
            output_fragment_(output_fragment),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5){}
    
    // Compute
    __device__ __forceinline__ void TileMAC(int dimW, long col, long col1,int dimMori, long row){
        //load lhs_matrix
        at::Half lhs_fragment_half[4]={0.0,0.0,0.0,0.0};
        half2 *lhs_fragment_half2 = reinterpret_cast<half2*>(lhs_fragment_half);
        uint32_t *lhs_fragment=reinterpret_cast<uint32_t*>(lhs_fragment_half);
        if(row < dimMori){
        lhs_fragment_half2[0] = *(lhs_tile_+(warpin_id%4));
        lhs_fragment_half2[1] = *(lhs_tile_+(warpin_id%4)+4);

        // lhs_fragment_half[0]=*(lhs_tile_+((warpin_id%4)*2));
        // lhs_fragment_half[1]=*(lhs_tile_+((warpin_id%4)*2)+1);
        // lhs_fragment_half[2]=*(lhs_tile_+((warpin_id%4)*2)+8);
        // lhs_fragment_half[3]=*(lhs_tile_+((warpin_id%4)*2)+9);
        lhs_tile_=lhs_tile_+8;}

        //load rhs_matrix
        at::Half rhs_fragment_half[8]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        half2 *rhs_fragment_half2 = reinterpret_cast<half2*>(rhs_fragment_half);
        uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_half);
        if(col>=0){
        // rhs_fragment_half[0]=*(rhs_tile_ + ((warpin_id%4)*2));
        // rhs_fragment_half[1]=*(rhs_tile_ + ((warpin_id%4)*2) + 1);
        // rhs_fragment_half[4]=*(rhs_tile_ + ((warpin_id%4)*2) + 8);
        // rhs_fragment_half[5]=*(rhs_tile_ + ((warpin_id%4)*2) + 9);
        rhs_fragment_half2[0] = *(rhs_tile_+(warpin_id%4));
        rhs_fragment_half2[2] = *(rhs_tile_+(warpin_id%4)+4);
        }
        if((col1)>=0){
        int temp = ((col1-col)*dimW)/2;
        rhs_fragment_half2[1] = *(rhs_tile_+(warpin_id%4) + temp);
        rhs_fragment_half2[3] = *(rhs_tile_+(warpin_id%4)+4 + temp);
        // rhs_fragment_half[2]=*(rhs_tile_ + ((warpin_id%4)*2) + ((col1-col)*dimW));
        // rhs_fragment_half[3]=*(rhs_tile_ + ((warpin_id%4)*2) + 1 + ((col1-col)*dimW));
        // rhs_fragment_half[6]=*(rhs_tile_ + ((warpin_id%4)*2) + 8 + ((col1-col)*dimW));
        // rhs_fragment_half[7]=*(rhs_tile_ + ((warpin_id%4)*2) + 9 + ((col1-col)*dimW));
        }
        //更新rhs_tile
        rhs_tile_=rhs_tile_+8;
            // if(warpin_id==4&warp_id==1)
            // {
            //     half *p=reinterpret_cast<half *>(rhs_fragment);
            //     for(int i=0;i<8;i++)
            //     printf("thread_id:%d, value is:%f\n", warpin_id,__half2float(*(p+i)));
            // }
     
        //MMA 
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \t"
            "{%0,%1}, \t"
            "{%2,%3,%4,%5}, \t"
            "{%6,%7}, \t"
            "{%0,%1}; ":
            "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
            "r"(rhs_fragment[0]),  "r"(rhs_fragment[1]),  "r"(rhs_fragment[2]),  "r"(rhs_fragment[3]),
            "r"(lhs_fragment[0]), "r"(lhs_fragment[1])
        ); 

    }
    

    __device__ __forceinline__ void TileMACResidue(int residue,int dimW, long col, long col1, int dimMori,  long row){
        const half * temp_lhs_tile_ = reinterpret_cast<const half*>(lhs_tile_);
        const half * temp_rhs_tile_ = reinterpret_cast<const half*>(rhs_tile_);
        //load lhs_matrix
        at::Half lhs_fragment_half[4]={0.0,0.0,0.0,0.0};
        uint32_t *lhs_fragment=reinterpret_cast<uint32_t*>(lhs_fragment_half);
        if(row < dimMori){
         if((((warpin_id%4)*2))<residue)
        lhs_fragment_half[0]=*(temp_lhs_tile_+((warpin_id%4)*2));
        if((((warpin_id%4)*2) + 1)<residue)
        lhs_fragment_half[1]=*(temp_lhs_tile_+((warpin_id%4)*2) +1);
        if((((warpin_id%4)*2) + 8)<residue)
        lhs_fragment_half[2]=*(temp_lhs_tile_+((warpin_id%4)*2) +8);
        if((((warpin_id%4)*2) + 9)<residue)
        lhs_fragment_half[3]=*(temp_lhs_tile_+((warpin_id%4)*2) +9);}

        //load rhs_matrix
         at::Half rhs_fragment_half[8]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_half);
        if(col>=0){
            if(((warpin_id%4)*2)<residue)
            rhs_fragment_half[0]=*(temp_rhs_tile_ +((warpin_id%4)*2));
            if(((warpin_id%4)*2 +1)<residue)
            rhs_fragment_half[1]=*(temp_rhs_tile_ +((warpin_id%4)*2) +1);
            if(((warpin_id%4)*2 +8)<residue)
            rhs_fragment_half[4]=*(temp_rhs_tile_ +((warpin_id%4)*2) +8);
            if(((warpin_id%4)*2 +9)<residue)
            rhs_fragment_half[5]=*(temp_rhs_tile_ +((warpin_id%4)*2) +9);
        }
        if((col1)>=0){
            if(((warpin_id%4)*2)<residue)
            rhs_fragment_half[2]=*(temp_rhs_tile_ + ((warpin_id%4)*2) + ((col1-col)*dimW));
            if(((warpin_id%4)*2 +1)<residue)
            rhs_fragment_half[3]=*(temp_rhs_tile_ + ((warpin_id%4)*2) + 1 + ((col1-col)*dimW));
            if(((warpin_id%4)*2 +8)<residue)
            rhs_fragment_half[6]=*(temp_rhs_tile_ + ((warpin_id%4)*2) + 8 + ((col1-col)*dimW));
            if(((warpin_id%4)*2 +9)<residue)
            rhs_fragment_half[7]=*(temp_rhs_tile_ + ((warpin_id%4)*2) + 9 + ((col1-col)*dimW));
        }
        //MMA 
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \t"
            "{%0,%1}, \t"
            "{%2,%3,%4,%5}, \t"
            "{%6,%7}, \t"
            "{%0,%1}; ":
            "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
            "r"(rhs_fragment[0]),  "r"(rhs_fragment[1]),  "r"(rhs_fragment[2]),  "r"(rhs_fragment[3]),
            "r"(lhs_fragment[0]), "r"(lhs_fragment[1])
        );  
    }

    
};

// 16x1 fp16
struct mmaComputeUtils_fp16_gen_16{
    const at::Half* lhs_tile_;
    const at::Half* rhs_tile_;
    uint32_t* output_fragment_;
    const int lane_id_;
    const int warpin_id;
    const int warp_id;
    int dimN_index_;
    int col_;
    int row_;
    int dimW_;

    // Constructor
    __device__ __forceinline__ mmaComputeUtils_fp16_gen_16(
        const at::Half *lhs_tile,
        const at::Half *rhs_tile,
        uint32_t* output_fragment,
        int lane_id):
            lhs_tile_(lhs_tile),
            rhs_tile_(rhs_tile),
            output_fragment_(output_fragment),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5){}
    
    // Compute
    __device__ __forceinline__ void TileMAC(int dimW, long col, int dimMori, long row){

        //load lhs_matrix
        at::Half lhs_fragment_half[8]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        uint32_t *lhs_fragment = reinterpret_cast<uint32_t *>(lhs_fragment_half);
        if(row < dimMori){
        lhs_fragment_half[0]=*(lhs_tile_ + ((warpin_id%4)*2));
        lhs_fragment_half[1]=*(lhs_tile_ + ((warpin_id%4)*2) + 1);
        lhs_fragment_half[4]=*(lhs_tile_ + ((warpin_id%4)*2) + 8);
        lhs_fragment_half[5]=*(lhs_tile_ + ((warpin_id%4)*2) + 9);
        }
        if((row+8) < dimMori){
        lhs_fragment_half[2]=*(lhs_tile_ + ((warpin_id%4)*2) + 8*dimW);
        lhs_fragment_half[3]=*(lhs_tile_ + ((warpin_id%4)*2) + 1 + 8*dimW);
        lhs_fragment_half[6]=*(lhs_tile_ + ((warpin_id%4)*2) + 8 + 8*dimW);
        lhs_fragment_half[7]=*(lhs_tile_ + ((warpin_id%4)*2) + 9 + 8*dimW);
        }
        lhs_tile_=lhs_tile_+16;

        //load rhs_matrix
        at::Half rhs_fragment_half[4]={0.0,0.0,0.0,0.0};
        uint32_t *rhs_fragment=reinterpret_cast<uint32_t*>(rhs_fragment_half);
        if(col >= 0){
        rhs_fragment_half[0]=*(rhs_tile_+((warpin_id%4)*2));
        rhs_fragment_half[1]=*(rhs_tile_+((warpin_id%4)*2)+1);
        rhs_fragment_half[2]=*(rhs_tile_+((warpin_id%4)*2)+8);
        rhs_fragment_half[3]=*(rhs_tile_+((warpin_id%4)*2)+9);
        }

        //更新rhs_tile
        rhs_tile_=rhs_tile_+16;
            // if(warpin_id==4&warp_id==1)
            // {
            //     half *p=reinterpret_cast<half *>(rhs_fragment);
            //     for(int i=0;i<8;i++)
            //     printf("thread_id:%d, value is:%f\n", warpin_id,__half2float(*(p+i)));
            // }
     
        //MMA 
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \t"
            "{%0,%1}, \t"
            "{%2,%3,%4,%5}, \t"
            "{%6,%7}, \t"
            "{%0,%1}; ":
            "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
            "r"(lhs_fragment[0]),  "r"(lhs_fragment[1]),  "r"(lhs_fragment[2]),  "r"(lhs_fragment[3]),
            "r"(rhs_fragment[0]), "r"(rhs_fragment[1])
        ); 

    }
    

    __device__ __forceinline__ void TileMACResidue(int residue,int dimW, long col, int dimMori,  long row){
        //load lhs_matrix
        at::Half lhs_fragment_half[8]={0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        uint32_t *lhs_fragment = reinterpret_cast<uint32_t *>(lhs_fragment_half);

        if(row < dimMori){
        if((((warpin_id%4)*2))<residue)
            lhs_fragment_half[0]=*(lhs_tile_ + ((warpin_id%4)*2));
        if((((warpin_id%4)*2) + 1)<residue)
            lhs_fragment_half[1]=*(lhs_tile_ + ((warpin_id%4)*2) + 1);
        if((((warpin_id%4)*2) + 8)<residue)
            lhs_fragment_half[4]=*(lhs_tile_ + ((warpin_id%4)*2) + 8);
        if((((warpin_id%4)*2) + 9)<residue)
            lhs_fragment_half[5]=*(lhs_tile_ + ((warpin_id%4)*2) + 9);
        }
        if((row+8) < dimMori){
        if((((warpin_id%4)*2))<residue)
            lhs_fragment_half[2]=*(lhs_tile_ + ((warpin_id%4)*2) + 8*dimW);
        if((((warpin_id%4)*2) + 1)<residue)
            lhs_fragment_half[3]=*(lhs_tile_ + ((warpin_id%4)*2) + 1 + 8*dimW);
        if((((warpin_id%4)*2) + 8)<residue)
            lhs_fragment_half[6]=*(lhs_tile_ + ((warpin_id%4)*2) + 8 + 8*dimW);
        if((((warpin_id%4)*2) + 9)<residue)
            lhs_fragment_half[7]=*(lhs_tile_ + ((warpin_id%4)*2) + 9 + 8*dimW);
        }

        //load rhs_matrix
        at::Half rhs_fragment_half[4]={0.0,0.0,0.0,0.0};
        uint32_t *rhs_fragment=reinterpret_cast<uint32_t*>(rhs_fragment_half);
        if(col >= 0){
        if((((warpin_id%4)*2))<residue)
            rhs_fragment_half[0]=*(rhs_tile_+((warpin_id%4)*2));
        if((((warpin_id%4)*2) + 1)<residue)
            rhs_fragment_half[1]=*(rhs_tile_+((warpin_id%4)*2)+1);
        if((((warpin_id%4)*2) + 8)<residue)
            rhs_fragment_half[2]=*(rhs_tile_+((warpin_id%4)*2)+8);
        if((((warpin_id%4)*2) + 9)<residue)
            rhs_fragment_half[3]=*(rhs_tile_+((warpin_id%4)*2)+9);
        }

            // if(warpin_id==4&warp_id==1)
            // {
            //     half *p=reinterpret_cast<half *>(rhs_fragment);
            //     for(int i=0;i<8;i++)
            //     printf("thread_id:%d, value is:%f\n", warpin_id,__half2float(*(p+i)));
            // }
     
        //MMA 
        asm("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 \t"
            "{%0,%1}, \t"
            "{%2,%3,%4,%5}, \t"
            "{%6,%7}, \t"
            "{%0,%1}; ":
            "+r"(output_fragment_[0]), "+r"(output_fragment_[1]):
            "r"(lhs_fragment[0]),  "r"(lhs_fragment[1]),  "r"(lhs_fragment[2]),  "r"(lhs_fragment[3]),
            "r"(rhs_fragment[0]), "r"(rhs_fragment[1])
        ); 
    }

    
};


/*
TF32
*/

struct mmaComputeUtils_tf32_gen{
    const float* lhs_tile_;
    const float* rhs_tile_;
    float* output_fragment_;
    const int lane_id_;
    const int warpin_id;
    const int warp_id;
    int dimN_index_;
    int col_;
    int row_;
    int dimW_;

    // Constructor
    __device__ __forceinline__ mmaComputeUtils_tf32_gen(
        const float *lhs_tile,
        const float *rhs_tile,
        float* output_fragment,
        int lane_id):
            lhs_tile_(lhs_tile),
            rhs_tile_(rhs_tile),
            output_fragment_(output_fragment),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5){}
    
    // Compute
    __device__ __forceinline__ void TileMAC(int dimW, long col, long col1,int dimMori, long row){
        //load lhs_matrix
        float lhs_fragment_tf32[2]={0.0, 0.0};
        uint32_t *lhs_fragment = reinterpret_cast<uint32_t *>(lhs_fragment_tf32);
        if(row < dimMori){
        lhs_fragment_tf32[0]=*(lhs_tile_+((warpin_id%4)));
        lhs_fragment_tf32[1]=*(lhs_tile_+((warpin_id%4))+4);
        lhs_tile_=lhs_tile_+8;}

        //load rhs_matrix
        float rhs_fragment_tf32[4]={0.0, 0.0, 0.0, 0.0};
        uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_tf32);
        if(col>=0){
        rhs_fragment_tf32[0]=*(rhs_tile_ + (warpin_id%4));
        rhs_fragment_tf32[2]=*(rhs_tile_ + (warpin_id%4) + 4);
        }
        if((col1)>=0){
        rhs_fragment_tf32[1]=*(rhs_tile_ + (warpin_id%4) + ((col1-col)*dimW));
        rhs_fragment_tf32[3]=*(rhs_tile_ + (warpin_id%4) + 4 + ((col1-col)*dimW));
        }
        //更新rhs_tile
        rhs_tile_=rhs_tile_+8;
            // if(warpin_id==4&warp_id==1)
            // {
            //     half *p=reinterpret_cast<half *>(rhs_fragment);
            //     for(int i=0;i<8;i++)
            //     printf("thread_id:%d, value is:%f\n", warpin_id,__half2float(*(p+i)));
            // }
     
       
        asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), "r"(rhs_fragment[2]), "r"(rhs_fragment[3]),
             "r"(lhs_fragment[0]), "r"(lhs_fragment[1]), 
             "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));      
    } 

    
    

    __device__ __forceinline__ void TileMACResidue(int residue,int dimW, long col, long col1, int dimMori,  long row){
        //load lhs_matrix
        float lhs_fragment_tf32[2]={0.0, 0.0};
        uint32_t *lhs_fragment = reinterpret_cast<uint32_t *>(lhs_fragment_tf32);
        if((warpin_id%4)<residue)
        lhs_fragment_tf32[0]=*(lhs_tile_+(warpin_id%4));
        if(((warpin_id%4) + 4)<residue)
        lhs_fragment_tf32[1]=*(lhs_tile_+(warpin_id%4) + 4);

        //load rhs_matrix
        float rhs_fragment_tf32[4]={0.0, 0.0, 0.0, 0.0};
        uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_tf32);
        if(col>=0){
            if((warpin_id%4)<residue)
            rhs_fragment_tf32[0]=*(rhs_tile_ +(warpin_id%4));
            if(((warpin_id%4) +4)<residue)
            rhs_fragment_tf32[2]=*(rhs_tile_ +((warpin_id%4)*2) +8);
        }
        if((col1)>=0){
            if((warpin_id%4)<residue)
            rhs_fragment_tf32[1]=*(rhs_tile_ + (warpin_id%4) + ((col1-col)*dimW));
            if(((warpin_id%4) +4)<residue)
            rhs_fragment_tf32[3]=*(rhs_tile_ + (warpin_id%4) + 4 + ((col1-col)*dimW));
        }

        //MMA 
        asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), "r"(rhs_fragment[2]), "r"(rhs_fragment[3]),
             "r"(lhs_fragment[0]), "r"(lhs_fragment[1]), 
             "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));      
        }

};


// 16x1 tf32
struct mmaComputeUtils_tf32_gen_16{
    const float* lhs_tile_;
    const float* rhs_tile_;
    float* output_fragment_;
    const int lane_id_;
    const int warpin_id;
    const int warp_id;
    int dimN_index_;
    int col_;
    int row_;
    int dimW_;

    // Constructor
    __device__ __forceinline__ mmaComputeUtils_tf32_gen_16(
        const float *lhs_tile,
        const float *rhs_tile,
        float* output_fragment,
        int lane_id):
            lhs_tile_(lhs_tile),
            rhs_tile_(rhs_tile),
            output_fragment_(output_fragment),
            lane_id_(lane_id),
            warpin_id(lane_id & 31),
            warp_id(lane_id>>5){}
    
    // Compute
    __device__ __forceinline__ void TileMAC(int dimW, long col, int dimMori, long row){
        //load lhs_matrix
        float lhs_fragment_tf32[4]={0.0, 0.0, 0.0, 0.0};
        uint32_t *lhs_fragment = reinterpret_cast<uint32_t *>(lhs_fragment_tf32);
        if(row < dimMori){
        lhs_fragment_tf32[0]=*(lhs_tile_ + (warpin_id%4));
        lhs_fragment_tf32[2]=*(lhs_tile_ + 4 + (warpin_id%4));
        }
        if((row+8) < dimMori){
        lhs_fragment_tf32[1]=*(lhs_tile_ + 8*dimW + (warpin_id%4));
        lhs_fragment_tf32[3]=*(lhs_tile_ + 4+ 8*dimW + (warpin_id%4));
        }
        lhs_tile_=lhs_tile_+8;

        //load rhs_matrix
        float rhs_fragment_tf32[2]={0.0, 0.0};
        uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_tf32);
        if(col >= 0){
        rhs_fragment_tf32[0]=*(rhs_tile_ + (warpin_id%4));
        rhs_fragment_tf32[1]=*(rhs_tile_+4 + (warpin_id%4));
        }
        //更新rhs_tile
        rhs_tile_=rhs_tile_+8;
            // if(warpin_id==4&warp_id==1)
            // {
            //     half *p=reinterpret_cast<half *>(rhs_fragment);
            //     for(int i=0;i<8;i++)
            //     printf("thread_id:%d, value is:%f\n", warpin_id,__half2float(*(p+i)));
            // }
     
       
        asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(lhs_fragment[0]), "r"(lhs_fragment[1]), "r"(lhs_fragment[2]), "r"(lhs_fragment[3]),
             "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), 
             "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3]));      
    } 

    
    

    __device__ __forceinline__ void TileMACResidue(int residue,int dimW, long col, int dimMori,  long row){
         //load lhs_matrix
        float lhs_fragment_tf32[4]={0.0, 0.0, 0.0, 0.0};
        uint32_t *lhs_fragment = reinterpret_cast<uint32_t *>(lhs_fragment_tf32);
        if(row < dimMori){
        if((warpin_id%4)<residue)
            lhs_fragment_tf32[0]=*(lhs_tile_ + (warpin_id%4));
        if(((warpin_id%4)+4)<residue)
            lhs_fragment_tf32[2]=*(lhs_tile_ + 4 + (warpin_id%4));
        }
        if((row+8) < dimMori){
        if((warpin_id%4)<residue)
            lhs_fragment_tf32[1]=*(lhs_tile_ + 8*dimW + (warpin_id%4));
        if(((warpin_id%4)+4)<residue)
            lhs_fragment_tf32[3]=*(lhs_tile_ + 4+ 8*dimW + (warpin_id%4));
        }

        //load rhs_matrix
        float rhs_fragment_tf32[2]={0.0, 0.0};
        uint32_t *rhs_fragment = reinterpret_cast<uint32_t *>(rhs_fragment_tf32);
        if(col >= 0){
        if((warpin_id%4)<residue)
            rhs_fragment_tf32[0]=*(rhs_tile_+ (warpin_id%4));
        if(((warpin_id%4)+4)<residue)
            rhs_fragment_tf32[1]=*(rhs_tile_+4+(warpin_id%4));
        }
        //更新rhs_tile
            // if(warpin_id==4&warp_id==1)
            // {
            //     half *p=reinterpret_cast<half *>(rhs_fragment);
            //     for(int i=0;i<8;i++)
            //     printf("thread_id:%d, value is:%f\n", warpin_id,__half2float(*(p+i)));
            // }
     
       
        asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
            : "=f"(output_fragment_[0]), "=f"(output_fragment_[1]), "=f"(output_fragment_[2]), "=f"(output_fragment_[3])
            : "r"(lhs_fragment[0]), "r"(lhs_fragment[1]), "r"(lhs_fragment[2]), "r"(lhs_fragment[3]),
             "r"(rhs_fragment[0]), "r"(rhs_fragment[1]), 
             "f"(output_fragment_[0]), "f"(output_fragment_[1]), "f"(output_fragment_[2]), "f"(output_fragment_[3])); 
    }
};
