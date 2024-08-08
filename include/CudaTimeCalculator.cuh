#pragma once

#include <cstdio>

namespace cudaTimeCalculator {
inline void cudaErrCheck_(cudaError_t stat) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), __FILE__, __LINE__);
    }
}
} // namespace cudaTimeCalculator

class CudaTimeCalculator {
 public:
  CudaTimeCalculator() {
      _time = 0.0f;

      cudaTimeCalculator::cudaErrCheck_(cudaEventCreate(&_star));
      cudaTimeCalculator::cudaErrCheck_(cudaEventCreate(&_stop));
  }

  ~CudaTimeCalculator() {
      cudaTimeCalculator::cudaErrCheck_(cudaEventDestroy(_star));
      cudaTimeCalculator::cudaErrCheck_(cudaEventDestroy(_stop));
  }

  inline void reset(){
      cudaTimeCalculator::cudaErrCheck_(cudaEventDestroy(_star));
      cudaTimeCalculator::cudaErrCheck_(cudaEventDestroy(_stop));

      cudaTimeCalculator::cudaErrCheck_(cudaEventCreate(&_star));
      cudaTimeCalculator::cudaErrCheck_(cudaEventCreate(&_stop));
  }

  inline void startClock() {
      cudaTimeCalculator::cudaErrCheck_(cudaEventRecord(_star));
  }

  inline void endClock() {
      cudaTimeCalculator::cudaErrCheck_(cudaEventRecord(_stop));
      cudaTimeCalculator::cudaErrCheck_(cudaEventSynchronize(_stop));
  }

  inline float getTime() {
      cudaTimeCalculator::cudaErrCheck_(cudaEventElapsedTime(&_time, _star, _stop));
      return _time;
  }

 private:
  cudaEvent_t _star;
  cudaEvent_t _stop;

  float _time;
};

