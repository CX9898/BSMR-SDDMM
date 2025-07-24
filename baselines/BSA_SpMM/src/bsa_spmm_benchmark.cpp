#include "bsa_spmm.cuh"
#include "option.h"
#include "logger.h"
#include "utilities.h"
#include "matrices.h"
#include "spmm.h"
#include "reorder.h"
#include "validate.h"


// int main(int argc, char* argv[])
// {
//     Option option = Option(argc, argv);
//     CSR lhs = CSR(option);
//     // ARR rhs = ARR(option, lhs);
//     ARR rhs = ARR(lhs.original_cols, lhs.cols, option.n_cols, true);
//     ARR result_mat = ARR(lhs.original_rows, lhs.rows, option.n_cols, false);
//     rhs.fill_random(option.zero_padding);
//     LOGGER logger = LOGGER(option);
//     Major b_major = row, c_major = row;
//     // BSA
//     if (option.spmm == 0)
//     {
//         vector<intT> permutation = reorder(lhs, option.method, option.alpha, option.block_size, option.repetitions,
//                                            logger);
//         BSA_HYBRID bsa_lhs = BSA_HYBRID(lhs, logger, option.block_size, option.delta, permutation);
//         bsa_spmm(bsa_lhs, rhs, result_mat, logger, option.repetitions, option.compress_rows, option.valid);
//     }
//     // cuSPARSE
//     else if (option.spmm == 1)
//     {
//         cusparse_spmm(lhs, rhs, result_mat, logger, option.repetitions, option.pattern_only);
//     }
//     // cuBLAS
//     else if (option.spmm == 2)
//     {
//         b_major = col;
//         c_major = col;
//         cublas_gemm(lhs, rhs, result_mat, logger, option.repetitions, option.pattern_only);
//     }
//
//     if (option.valid)
//     {
//         bool result = validate_spmm(lhs, rhs, result_mat, b_major, c_major);
//         if (result)
//         {
//             printf("Validation Succeeded\n");
//         }
//         else
//         {
//             printf("Validation Failed\n");
//         }
//     }
//     if (option.output_filename.length())
//     {
//         logger.save_logfile();
//     }
//     // logger.print_logfile();
//
//     return 0;
// }
