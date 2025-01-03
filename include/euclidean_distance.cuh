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
__inline__ __device__ T euclidean_opt(const T *vec1, const T *vec2,
                                      const int dimensions, const int vec_idx) {
  __shared__ T shared[32];     // Shared memory for partial sums
  int warp = threadIdx.x / 32; // Warp index
  int lane = threadIdx.x % 32; // Lane index within the warp
  T val = 0;

  // Accumulate partial sum for each thread's portion of the vector
  for (int i = threadIdx.x; i < dimensions; i += blockDim.x) {
    T diff = vec1[vec_idx * dimensions + i] - vec2[vec_idx * dimensions + i];
    val += diff * diff;
  }

  // Perform warp-level reduction
  val = warp_reduce_sum(val);

  // Write partial warp sums to shared memory
  if (lane == 0) {
    shared[warp] = val;
  }
  __syncthreads();

  // Perform block-level reduction using shared memory
  if (warp == 0) {
    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    val = warp_reduce_sum(val);
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

// Kernel to compute batch Euclidean distances using the atomic function
template <typename T>
__global__ void batch_euclidean_distance_atomic(const T *vec1, const T *vec2,
                                                T *distances, int num_vectors,
                                                int dimensions) {
  int vec_idx = blockIdx.x; // Each block handles a single vector pair
  if (vec_idx >= num_vectors)
    return; // Out-of-bounds check

  T distance = euclideanGPUatomic(&vec1[vec_idx * dimensions],
                                  &vec2[vec_idx * dimensions], dimensions);

  // Store the final distance after applying sqrt
  if (threadIdx.x == 0) {
    distances[vec_idx] = sqrtf(distance);
  }
}

// Kernel to compute batch Euclidean distances without optimizations
__global__ void batch_euclidean_distance(const float *vec1, const float *vec2,
                                         float *distances, int num_vectors,
                                         int dimensions) {
  int vec_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (vec_idx >= num_vectors)
    return;

  distances[vec_idx] = euclidean_distance(
      &vec1[vec_idx * dimensions], &vec2[vec_idx * dimensions], dimensions);
}

// Kernel to compute batch distances using the optimized method
__global__ void batch_gpu(const float *vec1, const float *vec2,
                          float *distances, int num_vectors, int dimensions) {
  int vec_idx = blockIdx.x; // Each block processes one vector pair
  if (vec_idx >= num_vectors)
    return;

  float distance = euclidean_opt(vec1, vec2, dimensions, vec_idx);

  if (threadIdx.x == 0) {
    distances[vec_idx] = distance;
  }
}

#endif // HNSW_EUCLIDEAN_DISTANCE_CUH
