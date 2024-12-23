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

__inline__ __device__ float warp_reduce_sum(float val) {
  #if __CUDACC_VER_MAJOR__ >= 9
  // __shfl_down is deprecated with cuda 9+. use newer variants
  unsigned int active = __activemask();
  #pragma unroll
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
      val = val + __shfl_down_sync(active, val, offset);
  }
  #else
  #pragma unroll
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
      val = val + __shfl_down(val, offset);
  }
  #endif
  return val;
}

__inline__ __device__ float euclidean_distance(const float* vec1, const float* vec2, int dimensions) {
    float sum = 0.0f;
    for (int i = 0; i < dimensions; ++i) {
        float diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    return sqrtf(sum);
}

template <typename T>
T euclidean_distance_cpu(const T* vec1, const T* vec2, int dimensions) {
    T sum = 0.0f;
    for (int i = 0; i < dimensions; i++) {
        T diff = vec1[i] - vec2[i];
        sum += diff*diff;
    }
    return sqrtf(sum);
}

template <typename T>
__inline__ __device__ T euclidean_opt(const T* vec1, const T* vec2, const int dimensions, const int vec_idx) {
    __shared__ T shared[32]; // Shared memory for partial sums
    int warp = threadIdx.x / 32; // Warp index
    int lane = threadIdx.x % 32; // Lane index within the warp
    T val = 0;
    // Correct input indexing to process the correct vector
    for (int i = threadIdx.x; i < dimensions; i += blockDim.x) {
        T diff = vec1[vec_idx * dimensions + i] - vec2[vec_idx * dimensions + i];
        val += diff * diff;
    }
    // Warp-level reduction
    val = warp_reduce_sum(val);
    // Write partial sums to shared memory
    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();
    // Block-level reduction
    if (warp == 0) {
        val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
        if (threadIdx.x == 0) {
            shared[0] = val; // Store final result in shared memory
        }
    }
    __syncthreads();
    return (threadIdx.x == 0) ? sqrtf(shared[0]) : 0.0f;
}

template <typename T>
__inline__ __device__ T euclideanGPUatomic(const T * vec1, const T * vec2, const int dimensions) {
  __syncthreads();
  static __shared__ T shared[32];
  int warp =  threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  T val = 0;
  for (int i = threadIdx.x; i < dimensions; i += blockDim.x) {
    T _val = vec1[i] - vec2[i];
    val = val + _val * _val;
  }
  val = warpReduceSum(val);
  // Write the partial warp sum to shared memory (Only the first thread in each warp writes)
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();
  // If there's only one warp, we're done
  if (blockDim.x <= 32) {
    return shared[0];
  }
  // Final reduction across warps (using atomicAdd)
  if (threadIdx.x < 32) {
    // Use atomicAdd to accumulate the results from all warps into shared[0]
    atomicAdd(&shared[0], shared[threadIdx.x]);
  }
  __syncthreads();
  return shared[0];
}

template <typename T>
__global__ void batch_euclidean_distance_atomic(const T* vec1, const T* vec2, T* distances, int num_vectors, int dimensions) {
    // Each block processes one vector pair
    int vec_idx = blockIdx.x; // Each block handles a single vector pair
    if (vec_idx >= num_vectors) return; // Out-of-bounds check
    // Calculate Euclidean distance using the atomic function
    T distance = euclideanGPUatomic(&vec1[vec_idx * dimensions], &vec2[vec_idx * dimensions], dimensions);
    // Write the result to global memory
    if (threadIdx.x == 0) {
        distances[vec_idx] = sqrtf(distance); // Apply sqrt to the final sum
    }
}

__global__ void batch_euclidean_distance(const float* vec1, const float* vec2, float* distances, int num_vectors, int dimensions) {
    int vec_idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    if (vec_idx >= num_vectors) return; // Out-of-bounds check
    // Use the inline device function to compute distance for the current vector pair
    distances[vec_idx] = euclidean_distance(&vec1[vec_idx * dimensions], &vec2[vec_idx * dimensions], dimensions);
}

__global__ void batch_gpu(const float* vec1, const float* vec2, float* distances, int num_vectors, int dimensions) {
    int vec_idx = blockIdx.x; // Each block processes one vector pair
    if (vec_idx >= num_vectors) return; // Out-of-bounds check
    // Compute distance using the optimized `euclidean_opt` function
    float distance = euclidean_opt(vec1, vec2, dimensions, vec_idx);
    // Write the result to global memory
    if (threadIdx.x == 0) {
        distances[vec_idx] = distance;
    }
}
#endif  // HNSW_EUCLIDEAN_DISTANCE_CUH
