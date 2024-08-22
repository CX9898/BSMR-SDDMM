#pragma once

#include <vector>

const float epsilon = 1e-3;

/**
 * error checking
 **/
template<typename T>
bool checkOneData(const T data1, const T data2);

template<typename T>
bool checkData(const size_t num, const T *data1, const T *data2);

template<typename T>
bool checkData(const std::vector<T> &data1, const std::vector<T> &data2);

template<typename T>
bool checkData(const size_t num, const std::vector<T> &dataHost1, const T *dataDev2);

template<typename T>
bool checkData(const size_t num, const T *dataDev1, const std::vector<T> &dataHost2);

template<typename T>
bool checkDevData(const size_t num, const T *dataDev1, const T *dataDev2);