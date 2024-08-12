#pragma once

namespace dev {

template<typename T>
class vector {
 public:
  vector() : _size(0), _data(nullptr) {};
  vector(int size);
  ~vector();

  inline __host__ __device__ size_t size() {
      return _size;
  }
  inline __host__ __device__ const T *data() const {
      return _data;
  }
  inline __host__ __device__ T *data() {
      return _data;
  }
  inline __device__ const T &operator[](size_t idx) const {
      return _data[idx];
  }
  inline __device__ T &operator[](size_t idx) {
      return _data[idx];
  }

 private:
  size_t _size;
  T *_data;
};

template<typename T>
inline vector<T>::vector(const int size) : vector() {
    _size = size;
    cudaMalloc(reinterpret_cast<void **>(_data), size * sizeof(T));
}

template<typename T>
inline vector<T>::~vector() {
    if (_data) { cudaFree(_data); }
}
} // namespace dev
