#ifndef HNSW_EUCLIDEAN_DISTANCE_CUH
#define HNSW_EUCLIDEAN_DISTANCE_CUH

#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

__inline__ __device__ float warpReduceSum(float val);

__inline__ __device__ float warp_reduce_sum(float val);

__inline__ __device__ float euclidean_distance(const float* vec1, const float* vec2, int dimensions);

template <typename T = float>
T euclidean_distance_cpu(const T* vec1, const T* vec2, int dimensions);

template <typename T = float>
__inline__ __device__ T euclidean_opt(const T* vec1, const T* vec2, const int dimensions, const int vec_idx);

template <typename T = float>
__inline__ __device__ T euclideanGPUatomic(const T * vec1, const T * vec2, const int dimensions);

template <typename T = float>
__global__ void batch_euclidean_distance_atomic(const T* vec1, const T* vec2, T* distances, int num_vectors, int dimensions);

__global__ void batch_euclidean_distance(const float* vec1, const float* vec2, float* distances, int num_vectors, int dimensions);

__global__ void batch_gpu(const float* vec1, const float* vec2, float* distances, int num_vectors, int dimensions);

#endif  // HNSW_EUCLIDEAN_DISTANCE_CUH