#pragma once

#include <cstdio>
#include <cmath>
#include <iomanip>
#include <limits>
#include <ostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "Matrix.hpp"
#include "cudaErrorCheck.cuh"
#include "devVector.cuh"

const float ERROR_THRESHOLD_EPSILON = 1e-3;

/** CPU 参考与 GPU 结果在 nnz 向量上的对比统计（与 checkOneData<float> 同一套容差） */
struct FloatVectorAccuracyStats {
    bool sizeMismatch = false;
    /** 参与比较的标量个数（通常等于 nnz） */
    size_t numEntriesCompared = 0;
    /** ||ref - gpu||_F（Frobenius / 欧氏长度） */
    double frobeniusNormOfDifference = 0.0;
    /** ||ref - gpu||_F / ||ref||_F；||ref||_F=0 且差非零时为 inf */
    double relativeFrobeniusError = 0.0;
    double maxAbsDifferencePerEntry = 0.0;
    double maxRelDifferencePerEntry = 0.0;
    /** mean absolute error: mean_i |ref_i - gpu_i| */
    double meanAbsoluteError = 0.0;
    /** mean squared error: mean_i (ref_i - gpu_i)^2; equals RMSE^2 */
    double meanSquaredError = 0.0;
    double rootMeanSquareError = 0.0;
    /** 不满足 checkOneData<float> 的条目数 */
    size_t numEntriesOutsideTolerance = 0;
};

template<typename T>
inline bool checkOneData(const T data1, const T data2) {
    return data1 == data2;
}

template<>
inline bool checkOneData<float>(const float data1, const float data2) {
    constexpr float ABS_EPSILON = 1e-5f;

    const float absDiff = std::fabs(data1 - data2);
    if (absDiff < ABS_EPSILON) return true;

    const float maxVal = std::max(std::max(std::fabs(data1), std::fabs(data2)), ERROR_THRESHOLD_EPSILON);
    return (absDiff / maxVal) < ERROR_THRESHOLD_EPSILON;
}

template<>
inline bool checkOneData<double>(const double data1, const double data2) {
    constexpr double ABS_EPSILON = 1e-5;

    const double absDiff = std::fabs(data1 - data2);
    if (absDiff < ABS_EPSILON) return true;

    const double maxVal = std::max(std::max(std::fabs(data1), std::fabs(data2)), static_cast<double>(ERROR_THRESHOLD_EPSILON));
    return (absDiff / maxVal) < ERROR_THRESHOLD_EPSILON;
}

inline FloatVectorAccuracyStats computeFloatVectorAccuracyStats(const std::vector<float>& ref,
                                                                const std::vector<float>& gpu) {
    FloatVectorAccuracyStats s;
    if (ref.size() != gpu.size()) {
        s.sizeMismatch = true;
        return s;
    }
    s.numEntriesCompared = ref.size();
    if (s.numEntriesCompared == 0) {
        return s;
    }

    double sumSq = 0.0;
    double sumAbs = 0.0;
    double sumSqRef = 0.0;
    float maxAbs = 0.f;
    float maxRel = 0.f;
    size_t outsideTol = 0;
    for (size_t i = 0; i < s.numEntriesCompared; ++i) {
        const float r = ref[i];
        const float g = gpu[i];
        const float ad = std::fabs(r - g);
        sumAbs += static_cast<double>(ad);
        sumSq += static_cast<double>(ad) * static_cast<double>(ad);
        sumSqRef += static_cast<double>(r) * static_cast<double>(r);
        if (ad > maxAbs) {
            maxAbs = ad;
        }
        const float denom = std::max(std::max(std::fabs(r), std::fabs(g)), ERROR_THRESHOLD_EPSILON);
        const float rel = ad / denom;
        if (rel > maxRel) {
            maxRel = rel;
        }
        if (!checkOneData<float>(r, g)) {
            ++outsideTol;
        }
    }
    s.maxAbsDifferencePerEntry = static_cast<double>(maxAbs);
    s.maxRelDifferencePerEntry = static_cast<double>(maxRel);
    const double invN = 1.0 / static_cast<double>(s.numEntriesCompared);
    s.meanAbsoluteError = sumAbs * invN;
    s.meanSquaredError = sumSq * invN;
    s.rootMeanSquareError = std::sqrt(s.meanSquaredError);
    s.numEntriesOutsideTolerance = outsideTol;
    s.frobeniusNormOfDifference = std::sqrt(sumSq);
    const double normRef = std::sqrt(sumSqRef);
    if (normRef > 0.0) {
        s.relativeFrobeniusError = s.frobeniusNormOfDifference / normRef;
    } else if (s.frobeniusNormOfDifference > 0.0) {
        s.relativeFrobeniusError = std::numeric_limits<double>::infinity();
    } else {
        s.relativeFrobeniusError = 0.0;
    }
    return s;
}

/** linePrefix 形如 "accuracy_vs_cpu_ref_"，输出键为 [linePrefix + "num_entries_compared" : …] */
inline void printFloatVectorAccuracyStatsLog(std::ostream& out,
                                              const FloatVectorAccuracyStats& st,
                                              const char* linePrefix){
    if (st.sizeMismatch){
        out << "[" << linePrefix << "compare_status : vector_length_mismatch]\n";
        return;
    }
    if (st.numEntriesCompared == 0){
        out << "[" << linePrefix << "compare_status : zero_nonzeros]\n";
        return;
    }
    out << "[" << linePrefix << "num_entries_compared : " << st.numEntriesCompared << "]\n";
    const std::ios::fmtflags oldFlags = out.flags();
    const std::streamsize oldPrec = out.precision();
    out << std::scientific << std::setprecision(8);
    out << "[" << linePrefix << "frobenius_norm_of_difference : " << st.frobeniusNormOfDifference << "]\n";
    out << "[" << linePrefix << "relative_frobenius_error : " << st.relativeFrobeniusError << "]\n";
    out << "[" << linePrefix << "max_absolute_diff_per_entry : " << st.maxAbsDifferencePerEntry << "]\n";
    out << "[" << linePrefix << "max_relative_diff_per_entry : " << st.maxRelDifferencePerEntry << "]\n";
    out << "[" << linePrefix << "mean_absolute_error : " << st.meanAbsoluteError << "]\n";
    out << "[" << linePrefix << "mean_squared_error : " << st.meanSquaredError << "]\n";
    out << "[" << linePrefix << "root_mean_square_error : " << st.rootMeanSquareError << "]\n";
    out.flags(oldFlags);
    out.precision(oldPrec);
    out << "[" << linePrefix << "num_entries_outside_tolerance : " << st.numEntriesOutsideTolerance << "]\n";
    out << "[" << linePrefix << "percent_entries_outside_tolerance : " << std::fixed << std::setprecision(6)
        << (100.0 * static_cast<double>(st.numEntriesOutsideTolerance) /
            static_cast<double>(st.numEntriesCompared))
        << "%]\n";
}

template<typename T>
inline bool checkDataFunction(const size_t num, const T *data1, const T *data2, size_t &numError) {
    bool isCorrect = true;

    printf("|---------------------------check data---------------------------|\n");
    printf("| Data size : %ld\n", num);
    printf("| Error threshold epsilon : %f\n", ERROR_THRESHOLD_EPSILON);
    printf("| Checking results...\n");

    size_t errors = 0;
    for (int idx = 0; idx < num; ++idx) {
        const T oneData1 = data1[idx];
        const T oneData2 = data2[idx];
        if (!checkOneData(oneData1, oneData2)) {
            ++errors;
            if (errors < 10) {
                printf("| Error : idx = %d, data1 = %f, data2 = %f, difference = %f\n",
                       idx,
                       static_cast<float>(oneData1),
                       static_cast<float>(oneData2),
                       static_cast<float>(oneData1 - oneData2));
            }
        }
    }
    numError = errors;
    if (errors > 0) {
        printf("| No Pass! Inconsistent data! %zu errors! Error rate : %2.2f%%\n",
               errors, static_cast<float>(errors) / static_cast<float>(num) * 100);
        isCorrect = false;
    } else {
        printf("| Pass! Result validates successfully.\n");
    }

    printf("|----------------------------------------------------------------|\n");

    return isCorrect;
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
bool checkData(const dev::vector<T> &devData1, const dev::vector<T> &devData2) {
    std::vector<T> hostData1 = d2h(devData1);
    std::vector<T> hostData2 = d2h(devData2);
    return checkData(hostData1, hostData2);
}

template<typename T>
bool checkData(const dev::vector<T> &devData1, const std::vector<T> &hostData2, size_t &numError) {
    std::vector<T> hostData1;
    d2h(hostData1, devData1);
    return checkData(hostData1, hostData2, numError);
}
