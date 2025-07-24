#include <fstream>
#include <random>
#include <iomanip>

#include "bsa_spmm.cuh"
#include "option.h"
#include "logger.h"
#include "utilities.h"
// #include "matrices.h"
#include "spmm.h"
#include "reorder.h"

template <typename T>
std::string to_trimmed_string(T value, const int precision = 6){
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;

    std::string s = oss.str();

    if (s.find('.') != std::string::npos){
        s.erase(s.find_last_not_of('0') + 1);
        if (s.back() == '.') s.pop_back();
    }

    return s;
}

void test_mode_reordering(CSR& lhs, Option& option){
    const std::vector<float> similarityThresholdAlpha = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};
    const std::vector<float> blockDensityThresholdDelta = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};
    for (const auto& alpha : similarityThresholdAlpha){
        option.alpha = alpha;
        LOGGER logger = LOGGER(option);
        vector<intT> permutation = reorder(lhs, option.method, option.alpha, option.block_size, option.repetitions,
                                           logger);
        for (const auto& delta : blockDensityThresholdDelta){
            option.delta = delta;
            logger.delta = delta;
            logger.num_tiles = 0;
            logger.avg_density_of_tiles = 0.0f;
            BSA_HYBRID bsa_lhs = BSA_HYBRID(lhs, logger, option.block_size, option.delta, permutation);

            const std::string logFile = option.output_log_directory + "BSA_" +
                "a_" + to_trimmed_string(alpha) + "_" +
                "d_" + to_trimmed_string(delta) + ".log";
            std::ofstream fout(logFile, std::ios::app);
            if (fout.fail()){
                fprintf(stderr, "Error, failed to open log file: %s\n", logFile.c_str());
                return;
            }
            fout << "\n---New data---\n";
            logger.print_logfile(fout);
        }
    }
}

int main(int argc, char* argv[]){
    Option option = Option(argc, argv);
    CSR lhs = CSR(option);
    ARR rhs = ARR(lhs.original_cols, lhs.cols, option.n_cols, true);
    ARR result_mat = ARR(lhs.original_rows, lhs.rows, option.n_cols, false);
    rhs.fill_random(option.zero_padding);
    LOGGER logger = LOGGER(option);
    printf("file : %s\n", option.input_filename.c_str());

    if (option.test_mode){
        test_mode_reordering(lhs, option);
        return 0;
    }

    vector<intT> permutation = reorder(lhs, option.method, option.alpha, option.block_size, option.repetitions, logger);
    BSA_HYBRID bsa_lhs = BSA_HYBRID(lhs, logger, option.block_size, option.delta, permutation);

    if (option.output_filename.length()){
        logger.save_logfile();
    }
}
