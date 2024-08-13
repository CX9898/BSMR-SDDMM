#pragma once

#include <cuda_runtime.h>

namespace dev {

template<typename T>
class vector {
 public:
  vector() : size_(0), data_(nullptr) {};
  vector(size_t size);
  vector(const vector &src);

  ~vector() {
      if (data_) { cudaFree(data_); }
  };

  inline __host__ __device__ size_t size() {
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
  T *begin() const;
  T *begin();
  T *end();
  T *end() const;

 private:
  size_t size_;
  T *data_ = nullptr;
};

template<typename T>
inline vector<T>::vector(const size_t size) : vector() {
    size_ = size;
    cudaMalloc(reinterpret_cast<void **> (&data_), size * sizeof(T));
}
template<typename T>
inline vector<T>::vector(const vector<T> &src) {
    size_ = src.size_;
    cudaMalloc(reinterpret_cast<void **> (&data_), src.size_ * sizeof(T));
    D2D(data_, src.data_, src.size_);
}

template<typename T>
inline T *vector<T>::begin() const {
    return data_;
}
template<typename T>
inline T *vector<T>::begin() {
    return data_;
}
template<typename T>
inline T *vector<T>::end() const {
    return data_ + size_ - 1;
}
template<typename T>
inline T *vector<T>::end() {
    return data_ + size_ - 1;
}

} // namespace dev
