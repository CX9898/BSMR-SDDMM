#pragma once
#include <omp.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include <utility>
#include <bits/stdc++.h>
#include <time.h>
#include <sys/time.h>
#include "Matrix.hpp"

using namespace std;

inline double seconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

const int actv_row_size = 180;
const int SM_CAPACITY = 6144;
const int BLOCKSIZE = 512;

namespace isratnisa {
class Matrix {
 public:
  void copyFromMatrix(const ::SparseMatrix<float> &matrix);

  long n_rows, n_cols;
  long nnz;

  vector<int> rows;
  vector<int> cols;
  vector<float> vals;

};
}

namespace isratnisa {

class TiledMatrix {
 public:
  int ntile_c;
  int ntile_r;
  int max_active_block;
  int max_active_row;
  long nnz;

  vector<int> rows;
  vector<int> cols;
  vector<float> vals;
  vector<int> row_holder;
  vector<int> active_row;
  vector<int> lastIdx_block_tile;
  vector<int> n_actv_row;
  vector<int> lastIdx_tile;
  vector<int> tiled_ind;

  TiledMatrix(isratnisa::Matrix S, int tile_sizeX, int tile_sizeY) {
      ntile_c = S.n_cols / tile_sizeX + 1;
      ntile_r = S.n_rows / tile_sizeY + 1;
      max_active_block = (S.n_rows / actv_row_size + 1);
      lastIdx_block_tile.resize((ntile_c + 1) * (S.n_rows / actv_row_size + 1));
      lastIdx_tile.resize(ntile_c + 1);
      rows.resize(S.nnz + ntile_c * BLOCKSIZE - 1);
      cols.resize(S.nnz + ntile_c * BLOCKSIZE - 1);
      vals.resize(S.nnz + ntile_c * BLOCKSIZE - 1);
      tiled_ind.resize(S.nnz + ntile_c * BLOCKSIZE - 1);
      active_row.resize(S.n_rows * ntile_c);
      row_holder.resize(S.n_rows);
      n_actv_row.resize(ntile_c);
  }
};

}
void make_HTasH(const vector<float> H, vector<float> &H_t, int n_cols, int k);

void initial(vector<float> &X, long n, int k);
void unsorted_make_CSR(int *rows, int *cols, float *vals, long nnz, long n_rows, long n_cols, int *row_ptr);
// void make_tile(smat_t &R, mat_int &tiled_bin, const int TS)

void make_CSR(vector<int> rows, vector<int> cols, vector<float> vals, long nnz,
              long n_rows, int *row_ptr, int *row_holder);

// void make_tile(smat_t &R, mat_int &tiled_bin, const int TS)


void make_2DBlocks(int *row_ptr, int *row_ind, int *col_ind, float *val_ind, long nnz, long n_rows, long n_cols);

void rewrite_matrix_2D(int *row_ptr, int *row_ind, int *col_ind, float *val_ind,
                       int *new_rows, int *new_cols, float *new_vals, long nnz, long n_rows, long n_cols,
                       int TS, int *tiled_ind, int *lastIdx_tile);
void rewrite_col_sorted_matrix(int *row_ptr, int *row_ind, int *col_ind, float *val_ind,
                               int *new_rows, int *new_cols, float *new_vals, long nnz, long n_rows, long n_cols,
                               int TS, int *tiled_ind, int *lastIdx_tile, int block, long &new_nnz);

int rewrite_matrix_1D(const isratnisa::Matrix S, isratnisa::TiledMatrix &tS, int *row_ptr, int TS, int *row_holder);


//perfect
// void rewrite_matrix_1D(int * row_ptr, int * row_ind, int *col_ind, float * val_ind, 
//  int *new_rows, int *new_cols, float * new_vals, long nnz, long n_rows, long n_cols,
//  int TS, int *tiled_ind, int * lastIdx_tile,  int block, long &new_nnz){

//  long new_idx = 0, idx =0;
//  int n_tile = n_cols/TS + 1, tile_no=0;
//     int *row_lim = new int[ n_rows];
//     int *count = new int[ n_tile];
//     int *active_row = new int[n_tile * n_rows];
//     lastIdx_tile[0] = 0; 

//     // #pragma omp parallel for 
//     for(int tile_lim = TS; tile_lim <= (n_cols+TS-1); tile_lim+=TS){ 
//      tile_no = tile_lim/TS;  

//      for(int r = 0; r <n_rows; ++r){ 
//          if(tile_lim == TS){
//              idx = row_ptr[r]; row_lim[r] = idx;}
//          else 
//                 idx = row_lim[r];

//              while(col_ind[idx] < tile_lim && idx < row_ptr[r+1]){
//                  // if(col_ind[idx] == 1)
//           //           cout << " inside " <<col_ind[idx]<<" " << idx <<" "<< new_idx <<" "<<tile_no << endl;                        
//                 tiled_ind[new_idx] = idx;
//                 new_rows[new_idx] = row_ind[idx];
//                 new_cols[new_idx] = col_ind[idx];
//                 new_vals[new_idx] = val_ind[idx];
//                 new_idx++;
//                 idx++;
//             }   
//             if(idx != row_lim[r])  
//                 active_row[tile_no * n_rows + count[tile_no]++]=r;       
//             row_lim[r] = idx;

//         }
//         lastIdx_tile[tile_no] = new_idx; 
//         // if(tile_no < 5){
//         //     cout << tile_no << " tile : " << count[tile_no] << endl;
//         //     for (int i = 0; i < count[tile_no]; ++i)
//         //         cout << active_row[tile_no*n_rows+i] << endl;

//         // }
//     }
//     for (int i = lastIdx_tile[2]; i < lastIdx_tile[3]; ++i)
//     {
//         if(new_cols[i]<180 )
//             cout << "lastIdx_tile " <<i<< " " <<row_ind[i]<< " "<<col_ind[i]<<" " <<new_cols[i] << endl;
//     }

//     new_nnz = nnz;

//     // for (int i = 0; i < nnz; ++i)
//     //   cout << "old nnz " <<i<<": " <<row_ind[i] <<" "<<col_ind[i] <<" "<<val_ind[i]<< endl;

//     // for (int i = 0; i < nnz; ++i)
//     //   cout << "new nnz " << i<<": " << new_rows[i] <<" "<<new_cols[i] <<" "<<new_vals[i] << endl;
//     // for (int i = 0; i <10; ++i)
//     //       cout  << i <<" : "<<row_ind[i] <<" " << col_ind[i] << " new: " << tiled_ind[i] 
//     //    <<" , "<< new_rows[i] <<" "<< new_cols[i]<< endl;
//      delete(row_lim);
// }
// fixing last tile
// long rewrite_matrix_1D(int * row_ptr, int * row_ind, int *col_ind, float * val_ind, 
//  int *new_rows, int *new_cols, float * new_vals, long nnz, long n_rows, long n_cols,
//  int TS, int *tiled_ind, int * lastIdx_tile, int blocksize, long &new_nnz){

//  long new_idx = 0, idx =0;
//  int n_tile = n_cols/TS + 1, tile_no=0;
//     int *row_lim = new int[ n_rows];
//     lastIdx_tile[0] = 0; 
//     long tot =0 ;

//     // #pragma omp parallel for 
//     for(int tile_lim = TS; tile_lim <= (n_cols+TS-1); tile_lim+=TS){ 
//      tile_no = tile_lim/TS; 
//         long tile_nnz = 0; 
//      for(int r = 0; r <n_rows; ++r){ 
//          if(tile_lim == TS){
//              idx = row_ptr[r]; row_lim[r] = idx;}
//          else {idx = row_lim[r];
//              // cout << " real " << r <<" "<<idx << endl;
//          } 

//              while(col_ind[idx] < tile_lim && idx < row_ptr[r+1]){
//                  // cout << " inside " <<r<<" " << new_idx <<" "<< idx << endl;                        
//                 tiled_ind[new_idx] = idx;
//                 new_rows[new_idx] = row_ind[idx];
//                 new_cols[new_idx] = col_ind[idx];
//                 new_vals[new_idx] = val_ind[idx];
//                 new_idx++;
//                 idx++;
//                 tile_nnz++;

//             }            
//             row_lim[r] = idx;
//         }  
//         // tot += tile_nnz;
//         // cout <<tile_nnz << " "<< tot << endl;

//         //lastIdx_tile[tile_no] = new_idx; 
//         int nnz_tile = new_idx - lastIdx_tile[tile_no-1];
//         int remainder_block = blocksize - nnz_tile % blocksize;
//         if(nnz_tile % blocksize == 0)
//          remainder_block = 0;

//         while(remainder_block >0){
//          tiled_ind[new_idx] = idx-1;
//             new_rows[new_idx] = new_rows[new_idx-1];
//             new_cols[new_idx] = new_cols[new_idx-1];
//             new_vals[new_idx] = 0;
//             // cout <<"fill up " <<new_idx <<" "<< idx <<" "<< row_ind[idx] << endl;
//             new_idx++;
//             remainder_block--;
//         }
//         lastIdx_tile[tile_no] = new_idx; 
//     }
//     cout << lastIdx_tile[0] << " "<<lastIdx_tile[1] << " "<<lastIdx_tile[2] << " " << endl;
//     new_nnz = new_idx;

//   //   for (int i = 0; i < nnz; ++i)
//      // cout << "before "<<i <<": "<< row_ind[i] <<" "<< col_ind[i] << " " << val_ind[i] << endl;
//   //   for (int i = 0; i < new_nnz; ++i)
//   //         cout << "after "<<i <<": "<< new_rows[i] <<" "<< new_cols[i] << " " << new_vals[i] << endl;

//   //   //for (int i = 16000; i <1650; ++i)
//     //       cout  << i <<" : " << lastIdx_tile[i+1]-lastIdx_tile[i] << endl;
//     delete(row_lim);

// }


void make_2Dtile(int *row_ptr, int *row_ind, int *col_ind, float *val_ind, long nnz, long n_rows, long n_cols,
                 int TS, int *row_lim);
void comp_bin(int n_bin, int *count, int n_rows, int *row_ptr, int nnz_max);


