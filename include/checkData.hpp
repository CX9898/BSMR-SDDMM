#pragma once

#include <vector>

const float epsilon = 1e-3;

/**
 * error checking
 **/
template<typename T>
bool checkData(const T data1, const T data2);

template<typename T>
bool checkData(const size_t num, const T *data1, const T *data2);

template<typename T>
bool checkData(const std::vector<float> &data1, const std::vector<float> &data2);

template<typename T>
bool checkData(const size_t num, const std::vector<float> &dataHost1, const float *dataDev2);

template<typename T>
bool checkData(const size_t num, const float *dataDev1, const std::vector<float> &dataHost2);

template<typename T>
bool checkDevData(const size_t num, const float *dataDev1, const float *dataDev2);