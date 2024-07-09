#ifndef CUDA_OPS_H
#define CUDA_OPS_H

#include <vector>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <stdexcept>          // for std::runtime_error

template <typename T>
T* cudaDeviceNew(int n)
{
    T* result;
    auto status = cudaMalloc((void**)&result, n*sizeof(T));
    if (status != cudaSuccess)
	throw std::runtime_error("cudaMalloc: failed!");
    return result;
}

template <typename T>
void MemcpyHostToDevice(T* dptr, const T* hptr, int n)
{
    auto status = cudaMemcpy(dptr, hptr, n*sizeof(T), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
	throw std::runtime_error("cudaMemcpy(Host->Device): failed!");
}

template <typename T>
void MemcpyDeviceToDevice(T* dptr, const T* hptr, int n)
{
    auto status = cudaMemcpy(dptr, hptr, n*sizeof(T), cudaMemcpyDeviceToDevice);
    if (status != cudaSuccess)
	throw std::runtime_error("cudaMemcpy(Host->Device): failed!");
}

template <typename T>
void MemcpyDeviceToHost(T* hptr, const T* dptr, int n)
{
    auto status = cudaMemcpy(hptr, dptr, n*sizeof(T), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
	throw std::runtime_error("cudaMemcpy(Device->Host): failed!");
}

template <typename T>
std::vector<T> MemcpyDeviceToHostVector(const T* dptr, int n)
{
    std::vector<T> host_vec(n);
    auto status = cudaMemcpy(host_vec.data(), dptr, n*sizeof(T), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
	throw std::runtime_error("cudaMemcpy(Device->Host): failed!");
    return host_vec;
}

void cuda_log_vector(double* v, int length);
void cuda_log_vector(float* v, int length);

#endif
