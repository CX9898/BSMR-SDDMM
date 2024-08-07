#pragma once

#include <vector>

namespace dev {
template<typename T>
inline void H2D(T *dev, const T *host, const size_t size) {
    cudaMemcpy(dev, host, size, cudaMemcpyHostToDevice);
}

template<typename T>
inline void D2H(T *host, const T *dev, const size_t size) {
    cudaMemcpy(host, dev, size, cudaMemcpyDeviceToHost);
}

template<typename T>
inline void H2D(T *dev, const std::vector<T> &host) {
    cudaMemcpy(dev, host.data(), host.size(), cudaMemcpyHostToDevice);
}

template<typename T>
inline void D2H(std::vector<T> &host, const T *dev, const size_t size) {
    host.clear();
    host.resize(size);
    cudaMemcpy(host.data(), dev, size, cudaMemcpyDeviceToHost);
}
} // namespace dev