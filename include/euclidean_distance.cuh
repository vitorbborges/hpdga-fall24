#ifndef HNSW_EUCLIDEAN_DISTANCE_CUH
#define HNSW_EUCLIDEAN_DISTANCE_CUH

#include <cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32

// Helper function for warp-level reduction
template <typename T = float>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
};

// Single-query Euclidean distance kernel using reduction
template <typename T>
__global__ void euclidean_distance_gpu(const T* vec1, const T* vec2, T* distance, int num_dims) {
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

    sum = warpReduceSum(sum);

    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (thread_id < blockDim.x / WARP_SIZE) {
        sum = (thread_id < blockDim.x / WARP_SIZE) ? shared[lane] : static_cast<T>(0);
        sum = warpReduceSum(sum);
    }

    if (thread_id == 0) {
        atomicAdd(distance, sqrtf(sum));    
    }
}

#endif // HNSW_EUCLIDEAN_DISTANCE_CUH