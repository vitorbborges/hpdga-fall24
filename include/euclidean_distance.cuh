#ifndef HNSW_EUCLIDEAN_DISTANCE_CUH
#define HNSW_EUCLIDEAN_DISTANCE_CUH

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Warp-level reduction to compute the sum of values within a warp
__inline__ __device__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

// Warp-level reduction with CUDA version check for compatibility
__inline__ __device__ float warp_reduce_sum(float val) {
#if __CUDACC_VER_MAJOR__ >= 9
  unsigned int active = __activemask();
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(active, val, offset);
  }
#else
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down(val, offset);
  }
#endif
  return val;
}

// Computes Euclidean distance between two vectors on the GPU
template <typename T>
__inline__ __device__ T euclidean_distance(const T *vec1, const T *vec2,
                                           int dimensions) {
  T sum = 0.0f;
  for (int i = 0; i < dimensions; ++i) {
    T diff = vec1[i] - vec2[i];
    sum += diff * diff;
  }
  return sqrtf(sum);
}

// CPU implementation of Euclidean distance for debugging/comparison
template <typename T>
T euclidean_distance_cpu(const T *vec1, const T *vec2, int dimensions) {
  T sum = 0.0f;
  for (int i = 0; i < dimensions; i++) {
    T diff = vec1[i] - vec2[i];
    sum += diff * diff;
  }
  return sqrtf(sum);
}

// Optimized GPU Euclidean distance calculation with static shared memory
template <typename T>
__inline__ __device__ T euclidean_distance_gpu(const T *vec1, const T *vec2,
                                               const int dimensions) {
  __shared__ T shared[32];     // Shared memory for warp-level reductions
  int warp = threadIdx.x / 32; // Warp index
  int lane = threadIdx.x % 32; // Lane index within warp
  T val = 0;

  // Compute partial sum for thread's assigned portion
  for (int i = threadIdx.x; i < dimensions; i += blockDim.x) {
    T diff = vec1[i] - vec2[i];
    val += diff * diff;
  }

  // Perform warp-level reduction
  val = warp_reduce_sum(val);

  // Write warp-level results to shared memory
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  // Perform block-level reduction using shared memory
  if (warp == 0) {
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    val = warp_reduce_sum(val);
    if (threadIdx.x == 0) {
      shared[0] = val; // Store final result
    }
  }
  __syncthreads();

  // Return the final result (sqrt of sum) from thread 0
  return (threadIdx.x == 0) ? sqrtf(shared[0]) : 0.0f;
}

// Optimized GPU Euclidean distance with dynamic shared memory
template <typename T>
__inline__ __device__ T euclidean_distance_dynamic_shared_gpu(
    const T *vec1, const T *vec2, const int dimensions, T *shared) {
  if (threadIdx.x >= dimensions)
    return 0.0f;
  int warp = threadIdx.x / 32; // Warp index
  int lane = threadIdx.x % 32; // Lane index within warp
  T val = 0;

  // Compute partial sum for thread's assigned portion
  for (int i = threadIdx.x; i < dimensions; i += blockDim.x) {
    T diff = vec1[i] - vec2[i];
    val += diff * diff;
  }

  // Perform warp-level reduction
  val = warpReduceSum(val);

  // Write warp-level results to shared memory
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  // Perform block-level reduction using shared memory
  if (warp == 0) {
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    val = warpReduceSum(val);
    if (threadIdx.x == 0) {
      shared[0] = val; // Store final result
    }
  }
  __syncthreads();

  // Return the final result (sqrt of sum) from thread 0
  return (threadIdx.x == 0) ? sqrtf(shared[0]) : 0.0f;
}

#endif // HNSW_EUCLIDEAN_DISTANCE_CUH
