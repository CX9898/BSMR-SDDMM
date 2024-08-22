#include <cstdio>

#include "checkData.hpp"
#include "cudaErrorCheck.cuh"

bool checkData(const size_t num, const float *data1, const float *data2) {
    printf("\n---------------------------\n"
           "Checking results...\n");

    int errors = 0;
    for (int idx = 0; idx < num; ++idx) {
        const float oneData1 = data1[idx];
        const float oneData2 = data2[idx];

        const float diffDats = abs(oneData1 - oneData2);

        const float relativeErr = diffDats / oneData1;
        const float eps = 1e-4;
        if (relativeErr >= eps) {
            ++errors;
            if (errors < 10) {
                printf("Error : idx = %d, data1 = %f, data2 = %f\n", idx, oneData1, oneData2);
            }
        }
    }

    if (errors > 0) {
        printf("No Pass! Inconsistent data! %d errors! Error rate : %2.2f%\n",
               errors, static_cast<float>(errors) / num * 100);

        return false;
    }

    printf("Pass! Result validates successfully.\n");

    return true;
}

bool checkData(const std::vector<float> &data1, const std::vector<float> &data2) {
    if (data1.size() != data2.size()) {
        return false;
    }
    return checkData(data1.size(), data1.data(), data2.data());
}

bool checkDevData(const size_t num, const float *dataDev1, const float *dataDev2) {
    auto dataHost1 = static_cast<float *>(malloc(num * sizeof(float)));
    auto dataHost2 = static_cast<float *>(malloc(num * sizeof(float)));

    cudaErrCheck(cudaMemcpy(dataHost1, dataDev1, num * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(dataHost2, dataDev2, num * sizeof(float), cudaMemcpyDeviceToHost));

    bool res = checkData(num, dataHost1, dataHost2);;

    free(dataHost1);
    free(dataHost2);

    return res;
}

bool checkData(const size_t num, const std::vector<float> &dataHost1, const float *dataDev2) {

    auto dataHost2 = static_cast<float *>(malloc(num * sizeof(float)));
    cudaErrCheck(cudaMemcpy(dataHost2, dataDev2, num * sizeof(float), cudaMemcpyDeviceToHost));

    bool res = checkData(num, dataHost1.data(), dataHost2);;

    free(dataHost2);

    return res;
}

bool checkData(const size_t num, const float *dataDev1, const std::vector<float> &dataHost2) {

    auto dataHost1 = static_cast<float *>(malloc(num * sizeof(float)));
    cudaErrCheck(cudaMemcpy(dataHost1, dataDev1, num * sizeof(float), cudaMemcpyDeviceToHost));

    bool res = checkData(num, dataHost1, dataHost2.data());;

    free(dataHost1);

    return res;
}