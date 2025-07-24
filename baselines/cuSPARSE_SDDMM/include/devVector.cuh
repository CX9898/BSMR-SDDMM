#pragma once

#include <vector>

#include <cuda_runtime.h>

namespace dev {

template<typename T>
class vector {
 public:
  vector() : size_(0), data_(nullptr) {};
  vector(size_t size);
  vector(size_t size, T value);
  vector(const vector<T> &src);
  vector(const std::vector<T> &src);

  ~vector() {
      if (data_) { cudaFree(data_); }
  };

  void resize(size_t size);
  void clear();

  inline __host__ __device__ size_t size() const {
      return size_;
  }
  inline __host__ __device__ const T *data() const {
      return data_;
  }
  inline __host__ __device__ T *data() {
      return data_;
  }
  inline __device__ const T &operator[](size_t idx) const {
      return data_[idx];
  }
  inline __device__ T &operator[](size_t idx) {
      return data_[idx];
  }

  // iterators
  T *cbegin() const;
  T *begin();
  T *cend() const;
  T *end();
  T *back() const;
  T back_data() const;

 private:
  size_t size_;
  T *data_ = nullptr;
};

template<typename T>
inline vector<T>::vector(const size_t size) : vector() {
    size_ = size;
    if(!size_){
        return;
    }
    cudaMalloc(reinterpret_cast<void **> (&data_), size * sizeof(T));
    if (!data_) {
        fprintf(stderr, "dev::vector: Device memory allocation failed\n");
    }
}

template<typename T>
inline vector<T>::vector(size_t size, T value) {
    size_ = size;
    if(!size_){
        return;
    }
    cudaMalloc(reinterpret_cast<void **> (&data_), size * sizeof(T));
    if (!data_) {
        fprintf(stderr, "dev::vector: Device memory allocation failed\n");
    }
    cudaMemset(data_, value, sizeof(T) * size_);
}

template<typename T>
inline vector<T>::vector(const vector<T> &src) {
    size_ = src.size_;
    if(!size_){
        return;
    }
    cudaMalloc(reinterpret_cast<void **> (&data_), src.size_ * sizeof(T));
    if (!data_) {
        fprintf(stderr, "dev::vector: Device memory allocation failed\n");
    }
    cudaMemcpy(data_, src.data_, size_ * sizeof(T), cudaMemcpyDeviceToDevice);
}

template<typename T>
inline vector<T>::vector(const std::vector<T> &src) {
    size_ = src.size();
    cudaMalloc(reinterpret_cast<void **> (&data_), src.size() * sizeof(T));
    if(!size_){
        return;
    }
    if (!data_) {
        fprintf(stderr, "dev::vector: Device memory allocation failed\n");
    }
    cudaMemcpy(data_, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
inline void vector<T>::resize(size_t size) {
    if (data_) {
        cudaFree(data_);
    }
    size_ = size;
    if(!size_){
        return;
    }
    cudaMalloc(reinterpret_cast<void **> (&data_), size * sizeof(T));
    if (!data_) {
        fprintf(stderr, "dev::vector: Device memory allocation failed\n");
    }
}

template<typename T>
inline void vector<T>::clear() {
    size_ = 0;
    if (data_) {
        cudaFree(data_);
    }
}

template<typename T>
inline T *vector<T>::cbegin() const {
    return data_;
}
template<typename T>
inline T *vector<T>::begin() {
    return data_;
}
template<typename T>
inline T *vector<T>::cend() const {
    if (!data_) {
        return nullptr;
    }
    return data_ + size_ - 1;
}
template<typename T>
inline T *vector<T>::end() {
    if (!data_) {
        return nullptr;
    }
    return data_ + size_ - 1;
}
template<typename T>
inline T *vector<T>::back() const {
    if (!data_) {
        return nullptr;
    }
    return data_ + size_ - 1;
}
template<typename T>
inline T vector<T>::back_data() const {
    if (!data_) {
        return 0;
    }
    T *val = (T *) malloc(sizeof(T));
    cudaMemcpy(val, data_ + size_ - 1, sizeof(T), cudaMemcpyDeviceToHost);
    return *val;
}

} // namespace dev


template<typename T>
inline void h2d(T *dev, const T *host, const size_t size) {
    cudaMemcpy(dev, host, size * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
inline void h2d(T *dev, const std::vector<T> &host) {
    cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
inline void h2d(dev::vector<T> &dev, const std::vector<T> &host) {
    dev.resize(host.size());
    cudaMemcpy(dev.data(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
inline void d2h(T *host, const T *dev, const size_t size) {
    cudaMemcpy(host, dev, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
inline void d2h(std::vector<T> &host, const T *dev, const size_t size) {
    host.clear();
    host.resize(size);
    cudaMemcpy(host.data(), dev, size * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
inline void d2h(std::vector<T> &host, const dev::vector<T> &dev) {
    host.clear();
    host.resize(dev.size());
    cudaMemcpy(host.data(), dev.data(), dev.size() * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
inline std::vector<T> d2h(const T *dev, const size_t size) {
    std::vector<T> host(size);
    cudaMemcpy(host.data(), dev, size * sizeof(T), cudaMemcpyDeviceToHost);
    return host;
}

template<typename T>
inline std::vector<T> d2h(const dev::vector<T> &dev) {
    std::vector<T> host(dev.size());
    cudaMemcpy(host.data(), dev.data(), sizeof(T) * dev.size(), cudaMemcpyDeviceToHost);
    return host;
}

template<typename T>
inline void d2d(T *dest, const T *src, const size_t size) {
    cudaMemcpy(dest, src, size * sizeof(T), cudaMemcpyDeviceToDevice);
}

template<typename T>
inline void d2d(dev::vector<T> &dest, const dev::vector<T> &src) {
    dest.clear();
    dest.resize(src.size());
    cudaMemcpy(dest.data(), src.data(), src.size() * sizeof(T), cudaMemcpyDeviceToDevice);
}

namespace pxy {
template<typename T>
class vector {
 public:
  __host__ __device__ vector(const dev::vector<T> &devVec) : size_(devVec.size()), data_((T *) devVec.data()) {}

  __host__ __device__ size_t size() const {
      return size_;
  }
  __host__ __device__ const T *data() const {
      return data_;
  }
  __host__ __device__ T *data() {
      return data_;
  }
  __host__ __device__ const T &operator[](size_t idx) const {
      return data_[idx];
  }
  __host__ __device__ T &operator[](size_t idx) {
      return data_[idx];
  }

 private:
  size_t size_;
  T *data_ = nullptr;

};
} // namespace pxy