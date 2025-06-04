#include "similarity.h"
#include "logger.h"
#include "utilities.h"
#include <queue>
#include <vector>
#include <set>
#include <chrono>
#include <iostream>

using namespace std;
vector<intT> reorder(CSR &lhs, intT method, float alpha, intT block_size, intT n_repetition, LOGGER &logger);
vector<intT> bsa_reordering_cpu(CSR &lhs, LOGGER &logger, float alpha, intT block_size, float &reordering_time);
vector<intT> bsa_reordering_gpu(CSR &lhs, float alpha, intT block_size, float &reordering_time, intT &cluster_cnt);