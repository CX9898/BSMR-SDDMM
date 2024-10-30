#pragma once

#include <cstdio>
#include <vector>
#include <string>

#include <cuda_runtime.h>

#include "Matrix.hpp"
#include "devMatrix.cuh"
#include "checkData.hpp"
#include "cudaErrorCheck.cuh"

const float epsilon = 1e-4;

/**
 * error checking
 **/
//template<typename T>
//inline bool checkDataFunction(const size_t num, const T *data1, const T *data2, size_t &numError);
//
//template<typename T>
//bool checkData(const std::vector<T> &data1, const std::vector<T> &data2);
//
//template<typename T>
//bool checkData(const std::vector<T> &data1, const dev::vector<T> &devData2);
//
//template<typename T>
//bool checkData(const dev::vector<T> &devData1, const std::vector<T> &hostData2);
//
//template<typename T>
//bool checkData(const size_t num, const std::vector<T> &dataHost1, const T *dataDev2);
//
//template<typename T>
//bool checkData(const size_t num, const T *dataDev1, const std::vector<T> &dataHost2);
//
//template<typename T>
//bool checkDevData(const size_t num, const T *dataDev1, const T *dataDev2);
//
//template<typename T>
//inline bool checkOneData(const T data1, const T data2);

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
inline bool checkDataFunction(const size_t num, const T *data1, const T *data2, size_t &numError) {
    printf("|---------------------------check data---------------------------|\n");
    printf("| Data size : %ld\n", num);
    printf("| Checking results...\n");

    size_t errors = 0;
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
    numError = errors;
    if (errors > 0) {
        printf("| No Pass! Inconsistent data! %zu errors! Error rate : %2.2f%%\n",
               errors, static_cast<float>(errors) / static_cast<float>(num) * 100);
        printf("|----------------------------------------------------------------|\n");
        return false;
    }

    printf("| Pass! Result validates successfully.\n");
    printf("|----------------------------------------------------------------|\n");
    return true;
}

template<typename T>
inline bool checkData(const std::vector<T> &hostData1, const std::vector<T> &hostData2) {
    if (hostData1.size() != hostData2.size()) {
        return false;
    }
    size_t numError;
    return checkDataFunction(hostData1.size(), hostData1.data(), hostData2.data(), numError);
}

template<typename T>
inline bool checkData(const std::vector<T> &hostData1, const std::vector<T> &hostData2, size_t &numError) {
    if (hostData1.size() != hostData2.size()) {
        return false;
    }
    return checkDataFunction(hostData1.size(), hostData1.data(), hostData2.data(), numError);
}

template<typename T>
bool checkData(const std::vector<T> &hostData1, const dev::vector<T> &devData2) {
    std::vector<T> hostData2;
    d2h(hostData2, devData2);
    return checkData(hostData1, hostData2);
}

template<typename T>
bool checkData(const std::vector<T> &hostData1, const dev::vector<T> &devData2, size_t &numError) {
    std::vector<T> hostData2;
    d2h(hostData2, devData2);
    return checkData(hostData1, hostData2, numError);
}

template<typename T>
bool checkData(const dev::vector<T> &devData1, const std::vector<T> &hostData2) {
    std::vector<T> hostData1;
    d2h(hostData1, devData1);
    return checkData(hostData1, hostData2);
}

template<typename T>
bool checkData(const dev::vector<T> &devData1, const std::vector<T> &hostData2, size_t &numError) {
    std::vector<T> hostData1;
    d2h(hostData1, devData1);
    return checkData(hostData1, hostData2, numError);
}