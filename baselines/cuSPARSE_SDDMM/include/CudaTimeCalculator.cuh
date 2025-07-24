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
      cudaTimeCalculator::cudaErrCheck_(cudaEventCreate(&start_));
      cudaTimeCalculator::cudaErrCheck_(cudaEventCreate(&stop_));
  }

  ~CudaTimeCalculator() {
      cudaTimeCalculator::cudaErrCheck_(cudaEventDestroy(start_));
      cudaTimeCalculator::cudaErrCheck_(cudaEventDestroy(stop_));
  }

  // 复位计时器(不销毁事件)
  inline void reset() {
      time_ = 0.0f;
  }

  // 记录起始时间(可选 Stream)
  inline void startClock(cudaStream_t stream = 0) {
      cudaTimeCalculator::cudaErrCheck_(cudaEventRecord(start_, stream));
  }

  // 记录结束时间(可选 Stream)
  inline void endClock(cudaStream_t stream = 0) {
      cudaTimeCalculator::cudaErrCheck_(cudaEventRecord(stop_, stream));
  }

  // 获取 GPU 执行时间(毫秒)
  inline float getTime() {
      cudaTimeCalculator::cudaErrCheck_(cudaEventSynchronize(stop_));
      cudaTimeCalculator::cudaErrCheck_(cudaEventElapsedTime(&time_, start_, stop_));
      return time_;
  }

 private:
  cudaEvent_t start_;
  cudaEvent_t stop_;

  float time_;
};

