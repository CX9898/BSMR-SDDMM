#pragma once

#include <vector>

namespace dev {

inline void cudaSync() {
    cudaDeviceSynchronize();
}

inline void printErrorStringSync() {
    printf("%s\n", cudaGetErrorString(cudaDeviceSynchronize()));
}

template<typename T>
inline void H2D(T *dev, const T *host, const size_t size) {
    cudaMemcpy(dev, host, size * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
inline void D2H(T *host, const T *dev, const size_t size) {
    cudaMemcpy(host, dev, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
inline void H2D(T *dev, const std::vector<T> &host) {
    cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
inline void D2H(std::vector<T> &host, const T *dev, const size_t size) {
    host.clear();
    host.resize(size);
    cudaMemcpy(host.data(), dev, size * sizeof(T), cudaMemcpyDeviceToHost);
}
} // namespace dev