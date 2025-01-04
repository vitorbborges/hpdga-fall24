#ifndef HNSW_EUCLIDEAN_DISTANCE_CUH
#define HNSW_EUCLIDEAN_DISTANCE_CUH

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

// Performs a warp-level reduction to compute the sum of values within a warp
__inline__ __device__ float warpReduceSum(float val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

// Performs a warp-level reduction to compute the sum of values within a warp
// (CUDA version check included)
__inline__ __device__ float warp_reduce_sum(float val) {
#if __CUDACC_VER_MAJOR__ >= 9
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

// Computes the Euclidean distance between two vectors on the GPU
__inline__ __device__ float
euclidean_distance(const float *vec1, const float *vec2, int dimensions) {
  float sum = 0.0f;
  for (int i = 0; i < dimensions; ++i) {
    float diff = vec1[i] - vec2[i];
    sum += diff * diff;
  }
  return sqrtf(sum);
}

// CPU implementation of Euclidean distance for comparison/debugging
template <typename T>
T euclidean_distance_cpu(const T *vec1, const T *vec2, int dimensions) {
  T sum = 0.0f;
  for (int i = 0; i < dimensions; i++) {
    T diff = vec1[i] - vec2[i];
    sum += diff * diff;
  }
  return sqrtf(sum);
}

// Optimized GPU function for computing Euclidean distance with shared memory
// and warp-level reductions
template <typename T>
__inline__ __device__ T euclidean_distance_gpu(const T *vec1, const T *vec2,
                                      const int dimensions, T *shared) {
  if (threadIdx.x >= dimensions) return 0.0f;
  int warp = threadIdx.x / 32; // Warp index
  int lane = threadIdx.x % 32; // Lane index within the warp
  T val = 0;

  // Accumulate partial sum for each thread's portion of the vector
  for (int i = threadIdx.x; i < dimensions; i += blockDim.x) {
    T diff = vec1[i] - vec2[i];
    val += diff * diff;
  }

  // Perform warp-level reduction
  val = warpReduceSum(val);

  // Write partial warp sums to shared memory
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  // Perform block-level reduction using shared memory
  if (warp == 0) {
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    val = warpReduceSum(val);
    if (threadIdx.x == 0) {
      shared[0] = val; // Store final result in shared memory
    }
  }
  __syncthreads();

  // Return the final result (sqrt of the sum) from the first thread
  return (threadIdx.x == 0) ? sqrtf(shared[0]) : 0.0f;
}

// Alternative GPU function using atomic operations for reduction
template <typename T>
__inline__ __device__ T euclideanGPUatomic(const T *vec1, const T *vec2,
                                           const int dimensions) {
  __syncthreads();
  static __shared__ T shared[32];
  int warp = threadIdx.x / 32;
  int lane = threadIdx.x % 32;
  T val = 0;

  // Compute partial sums within the warp
  for (int i = threadIdx.x; i < dimensions; i += blockDim.x) {
    T _val = vec1[i] - vec2[i];
    val += _val * _val;
  }
  val = warpReduceSum(val);

  // Store partial results from warps into shared memory
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  // Perform final reduction across all warps using atomicAdd
  if (blockDim.x > 32 && threadIdx.x < 32) {
    atomicAdd(&shared[0], shared[threadIdx.x]);
  }
  __syncthreads();

  return shared[0];
}

#endif // HNSW_EUCLIDEAN_DISTANCE_CUH
