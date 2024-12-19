#pragma once

#include <cuda_runtime.h>
#include <cmath>

// Helper function for warp-level reduction
template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// Kepler-optimized Euclidean distance kernel
template <typename T>
__inline__ __device__ void euclidean_distance_kepler(const T* vec1, const T* vec2, T* distances, size_t dimensions) {
    static __shared__ T shared[32];


    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    unsigned int warpid = threadIdx.x / 32;
    unsigned int within_warp = threadIdx.x % 32;

    if (idx >= dimensions) return;

    // Partial sum computation
    T sum = 0.0f;
    for (int j = tid; j < dimensions; j += blockDim.x) {
        float diff = vec1[idx * dimensions + j] - vec2[idx * dimensions + j];
        sum += diff * diff;
    }

    // Warp reduction
    sum = warpReduceSum(sum);

    // Block reduction using shared memory
    if ((tid & 31) == 0) sharedData[tid / warpSize] = sum;
    __syncthreads();

    if (tid < blockDim.x / warpSize) sum = warpReduceSum(sharedData[tid]);
    if (tid == 0) distances[blockIdx.x] = sqrtf(sum);
}


// Single-query Euclidean distance kernel using reduction
template <typename T>
__inline__ __device__ void euclidean_distance_gpu(const T* vec1, const T* vec2, T* distance, int num_dims) {
    static __shared__ T shared[32]; 

    int thread_id = threadIdx.x;              // Thread index within the block
    int warp_id = thread_id / 32;      // Warp ID
    int lane = thread_id % 32;         // Lane within the warp

    // Step 1: Compute partial sums strided
    T sum = 0;
    for (int i = thread_id; i < num_dims; i += blockDim.x) {
        T diff = vec1[i] - vec2[i];
        sum += diff * diff;  
    }

    sum = warp_reduce_sum(sum);

    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (thread_id < blockDim.x / WARP_SIZE) {
        sum = (thread_id < blockDim.x / WARP_SIZE) ? shared[lane] : static_cast<T>(0);
        sum = warp_reduce_sum(sum);
    }

    if (thread_id == 0) {
        *distance = sqrtf(sum); 
    }
}