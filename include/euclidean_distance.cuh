#ifndef HNSW_EUCLIDEAN_DISTANCE_CUH
#define HNSW_EUCLIDEAN_DISTANCE_CUH

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template <typename T>
__inline__ __device__ T euclidean_distance_gpu(const T* vec1, const T* vec2, const int& dimensions) {
    if (threadIdx.x >= dimensions) return 0.0f;
    __shared__ T shared[32]; // Shared memory for partial sums

    int warp = threadIdx.x / 32; // Warp index
    int lane = threadIdx.x % 32; // Lane index within the warp

    T val = 0;

    // Correct input indexing to process the correct vector
    for (int i = threadIdx.x; i < dimensions; i += blockDim.x) {
        T diff = vec1[i] - vec2[i];
        val += diff * diff;
    }

    // Warp-level reduction
    val = warpReduceSum(val);

    // Write partial sums to shared memory
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();

    // Block-level reduction
    if (warp == 0) {
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        val = warpReduceSum(val);
        if (threadIdx.x == 0) {
            shared[0] = val; // Store final result in shared memory
        }
    }
    __syncthreads();

    return (threadIdx.x == 0) ? sqrtf(shared[0]) : 0.0f;
}

// example of batch

__global__ void batch_gpu(const float* vec1, const float* vec2, float* distances, int num_vectors, int dimensions) {
    int vec_idx = blockIdx.x; 
    if (vec_idx >= num_vectors) return;

    // Offset the input pointers for the current vector pair
    const float* vec1_offset = vec1 + vec_idx * dimensions;
    const float* vec2_offset = vec2 + vec_idx * dimensions;

    float distance = euclidean_distance_gpu(vec1_offset, vec2_offset, dimensions);

    __syncthreads(); 
    if (threadIdx.x == 0) {
        distances[vec_idx] = distance;
    }
}

#endif  // HNSW_EUCLIDEAN_DISTANCE_CUH
