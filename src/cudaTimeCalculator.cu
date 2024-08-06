#include "CudaTimeCalculator.cuh"
#include "cudaErrorCheck.cuh"

CudaTimeCalculator::CudaTimeCalculator(){
    _time = 0.0f;

    cudaErrCheck(cudaEventCreate(&_star));
    cudaErrCheck(cudaEventCreate(&_stop));
}

CudaTimeCalculator::~CudaTimeCalculator() {
    cudaErrCheck(cudaEventDestroy(_star));
    cudaErrCheck(cudaEventDestroy(_stop));
}

void CudaTimeCalculator::startClock() {
    cudaErrCheck(cudaEventRecord(_star));
}

void CudaTimeCalculator::endClock() {
    cudaErrCheck(cudaEventRecord(_stop));
    cudaErrCheck(cudaEventSynchronize(_stop));
}

float CudaTimeCalculator::getTime() {
    cudaErrCheck(cudaEventElapsedTime(&_time, _star, _stop));
    return _time;
}