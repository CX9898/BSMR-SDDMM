#pragma once

#include <cstdio>
#include <vector>

#include <cuda_runtime.h>

#include "Matrix.hpp"
#include "devMatrix.cuh"
#include "checkData.hpp"
#include "cudaErrorCheck.cuh"

const float epsilon = 1e-4;

/**
 * error checking
 **/
template<typename T>
bool checkDataFunction(const size_t num, const T *data1, const T *data2);

template<typename T>
bool checkData(const std::vector<T> &data1, const std::vector<T> &data2);

template<typename T>
bool checkData(const size_t num, const std::vector<T> &dataHost1, const T *dataDev2);

template<typename T>
bool checkData(const size_t num, const T *dataDev1, const std::vector<T> &dataHost2);

template<typename T>
bool checkDevData(const size_t num, const T *dataDev1, const T *dataDev2);

template<typename T>
inline bool checkOneData(const T data1, const T data2);

template<typename T>
inline bool checkOneData(const T data1, const T data2) {
    return data1 == data2;
}
template<>
inline bool checkOneData<float>(const float data1, const float data2) {
    return abs(data1 - data2) / data1 < epsilon;
}
template<>
inline bool checkOneData<double>(const double data1, const double data2) {
    return abs(data1 - data2) / data1 < epsilon;
}

template<typename T>
bool checkDataFunction(const size_t num, const T *data1, const T *data2) {
    printf("|---------------------------check data---------------------------|\n"
           "| Checking results...\n");

    int errors = 0;
    for (int idx = 0; idx < num; ++idx) {
        const T oneData1 = data1[idx];
        const T oneData2 = data2[idx];
        if (!checkOneData(oneData1, oneData2)) {
            ++errors;
            if (errors < 10) {
                printf("| Error : idx = %d, data1 = %f, data2 = %f\n",
                       idx, static_cast<float>(oneData1), static_cast<float>(oneData2));
            }
        }
    }

    if (errors > 0) {
        printf("| No Pass! Inconsistent data! %d errors! Error rate : %2.2f%%\n",
               errors, static_cast<float>(errors) / static_cast<float>(num) * 100);
        printf("|----------------------------------------------------------------|\n");
        return false;
    }

    printf("| Pass! Result validates successfully.\n");
    printf("|----------------------------------------------------------------|\n");
    return true;
}

template<typename T>
bool checkData(const std::vector<T> &data1, const std::vector<T> &data2) {
    if (data1.size() != data2.size()) {
        return false;
    }
    return checkDataFunction(data1.size(), data1.data(), data2.data());
}

template<typename T>
bool checkDevData(const size_t num, const T *dataDev1, const T *dataDev2) {
    auto dataHost1 = static_cast<T *>(malloc(num * sizeof(float)));
    auto dataHost2 = static_cast<T *>(malloc(num * sizeof(float)));

    cudaErrCheck(cudaMemcpy(dataHost1, dataDev1, num * sizeof(T), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(dataHost2, dataDev2, num * sizeof(T), cudaMemcpyDeviceToHost));

    bool res = checkDataFunction(num, dataHost1, dataHost2);;

    free(dataHost1);
    free(dataHost2);

    return res;
}

template<typename T>
bool checkData(const size_t num, const std::vector<T> &dataHost1, const T *dataDev2) {

    auto dataHost2 = static_cast<T *>(malloc(num * sizeof(T)));
    cudaErrCheck(cudaMemcpy(dataHost2, dataDev2, num * sizeof(T), cudaMemcpyDeviceToHost));

    bool res = checkDataFunction(num, dataHost1.data(), dataHost2);;

    free(dataHost2);

    return res;
}

template<typename T>
bool checkData(const size_t num, const T *dataDev1, const std::vector<T> &dataHost2) {

    auto dataHost1 = static_cast<T *>(malloc(num * sizeof(T)));
    cudaErrCheck(cudaMemcpy(dataHost1, dataDev1, num * sizeof(T), cudaMemcpyDeviceToHost));

    bool res = checkDataFunction(num, dataHost1, dataHost2.data());;

    free(dataHost1);

    return res;
}
//
//template bool checkData<uint32_t>(const std::vector<uint32_t> &, const std::vector<uint32_t> &);
//template bool checkData<uint64_t>(const std::vector<uint64_t> &, const std::vector<uint64_t> &);
//template bool checkData<int>(const std::vector<int> &, const std::vector<int> &);
//template bool checkData<float>(const std::vector<float> &, const std::vector<float> &);
//template bool checkData<double>(const std::vector<double> &, const std::vector<double> &);