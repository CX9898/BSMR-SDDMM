#include "bsa_spmm.cuh"
#include "option.h"
#include "logger.h"
#include "utilities.h"
// #include "matrices.h"
#include "spmm.h"
#include "reorder.h"

// int main(int argc, char *argv[])
// {
//     Option option = Option(argc, argv);
//     CSR lhs = CSR(option);
//     ARR rhs = ARR(lhs.original_cols, lhs.cols, option.n_cols, true);
//     ARR result_mat = ARR(lhs.original_rows, lhs.rows, option.n_cols, false);
//     rhs.fill_random(option.zero_padding);
//     LOGGER logger = LOGGER(option);

//     vector<intT> permutation = reorder(lhs, option.method, option.alpha, option.block_size, option.repetitions, logger);
//     BSA_HYBRID bsa_lhs = BSA_HYBRID(lhs, logger, option.block_size, option.delta, permutation);

//     if (option.output_filename.length())
//         logger.save_logfile();
// }