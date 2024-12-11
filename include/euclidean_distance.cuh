#ifndef HNSW_EUCLIDEAN_DISTANCE_CUH
#define HNSW_EUCLIDEAN_DISTANCE_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <utils.cuh>

using utils::Data;

// CUDA kernel for computing partial squared differences
template <typename T = float>
__global__ void euclidean_distance_kernel(const T* a, const T* b, T* partialSums, int size) {
    extern __shared__ T sharedData[];

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load squared differences into shared memory
    if (globalIdx < size) {
        T diff = a[globalIdx] - b[globalIdx];
        sharedData[tid] = diff * diff;
    } else {
        sharedData[tid] = 0.0;
    }

    __syncthreads();

    // Perform tree-based reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write the result of this block to the partialSums array
    if (tid == 0) {
        partialSums[blockIdx.x] = sharedData[0];
    }
}

// Host function to compute Euclidean distance
template <typename T>
T euclidean_distance_cuda(const Data<T>& a, const Data<T>& b) {

    int size = a.size(); // TODO: pass size as reference to save memory in each call.
    const T *h_a = a.x.data();
    const T *h_b = b.x.data();
    T *d_a, *d_b, *d_partialSums;
    int blockSize = 32;
    int numBlocks = (size + blockSize - 1) / blockSize;

    // Allocate memory on device
    cudaMalloc(&d_a, size * sizeof(T));
    cudaMalloc(&d_b, size * sizeof(T));
    cudaMalloc(&d_partialSums, numBlocks * sizeof(T));

    // Copy input data to device
    cudaMemcpy(d_a, h_a, size * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size * sizeof(T), cudaMemcpyHostToDevice);

    // Launch kernel
    euclidean_distance_kernel<T><<<numBlocks, blockSize, blockSize * sizeof(T)>>>(d_a, d_b, d_partialSums, size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();

    // Allocate memory for partial sums on host
    T* h_partialSums = new T[numBlocks];

    // Copy partial sums back to host
    cudaMemcpy(h_partialSums, d_partialSums, numBlocks * sizeof(T), cudaMemcpyDeviceToHost);

    // Final reduction on host
    T sum = 0;
    for (int i = 0; i < numBlocks; ++i) {
        sum += h_partialSums[i];
    }

    // Free memory
    delete[] h_partialSums;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_partialSums);

    return std::sqrt(sum);
}

#endif // HNSW_EUCLIDEAN_DISTANCE_CUH
