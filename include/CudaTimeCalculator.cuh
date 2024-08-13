#pragma once

#include <cstdio>
#include <cuda_runtime.h>

namespace cudaTimeCalculator {
inline void cudaErrCheck_(cudaError_t stat) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), __FILE__, __LINE__);
    }
}
} // namespace cudaTimeCalculator

class CudaTimeCalculator {
 public:
  inline CudaTimeCalculator() {
      time_ = 0.0f;

      cudaTimeCalculator::cudaErrCheck_(cudaEventCreate(&star_));
      cudaTimeCalculator::cudaErrCheck_(cudaEventCreate(&stop_));
  }

  ~CudaTimeCalculator() {
      cudaTimeCalculator::cudaErrCheck_(cudaEventDestroy(star_));
      cudaTimeCalculator::cudaErrCheck_(cudaEventDestroy(stop_));
  }

  inline void reset(){
      cudaTimeCalculator::cudaErrCheck_(cudaEventDestroy(star_));
      cudaTimeCalculator::cudaErrCheck_(cudaEventDestroy(stop_));

      cudaTimeCalculator::cudaErrCheck_(cudaEventCreate(&star_));
      cudaTimeCalculator::cudaErrCheck_(cudaEventCreate(&stop_));
  }

  inline void startClock() {
      cudaTimeCalculator::cudaErrCheck_(cudaEventRecord(star_));
  }

  inline void endClock() {
      cudaTimeCalculator::cudaErrCheck_(cudaEventRecord(stop_));
      cudaTimeCalculator::cudaErrCheck_(cudaEventSynchronize(stop_));
  }

  inline float getTime() {
      cudaTimeCalculator::cudaErrCheck_(cudaEventElapsedTime(&time_, star_, stop_));
      return time_;
  }

 private:
  cudaEvent_t star_;
  cudaEvent_t stop_;

  float time_;
};

