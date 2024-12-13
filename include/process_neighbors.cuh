#ifndef HNSW_PROCESS_NEIGHBORS_CUH
#define HNSW_PROCESS_NEIGHBORS_CUH

#include <cuda_runtime.h>
#include <utils.cuh>

#define cudaSafeCall(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

using namespace utils;

template <typename T = float>
__global__ void process_neighbors_kernel(
    const size_t& ds_size,
    const size_t& vec_dim,
    const T* query,
    bool* visited,
    const size_t& n_neighbors,
    const int* neighbors_id,
    const T* neighbors_flat,
    T* dist_from_neighbors
) {
    extern __shared__ T sharedData[];

    int tid = threadIdx.x;
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;

    if (globalIdx >= vec_dim * n_neighbors) return;

    // Calculate which neighbor and feature index we are processing
    int neighborIdx = globalIdx / vec_dim;  // Which neighbor (from 0 to n_neighbors-1)
    int featureIdx = globalIdx % vec_dim;   // Which feature (from 0 to vec_dim-1)

    // Load squared differences into shared memory
    if (globalIdx < vec_dim * n_neighbors) {
        T diff = query[featureIdx] - neighbors_flat[neighborIdx * vec_dim + featureIdx];
        sharedData[tid] = diff * diff;
    } else {
        sharedData[tid] = 0.0;
    }

    __syncthreads();

    // Perform tree-based reduction (sum over the block)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write the result of this block to the dist_from_neighbors array
    if (tid == 0) {
        dist_from_neighbors[blockIdx.x] = sharedData[0];
    }

    // Mark neighbors as visited if the index is within range
    if (blockIdx.x < n_neighbors && neighbors_id[blockIdx.x] < ds_size) {
        visited[neighbors_id[blockIdx.x]] = true;
    }
}

template <typename T = float>
void process_neighbors_cuda(
    const size_t& ds_size,
    const size_t& vec_dim,
    const T* h_query,
    bool* h_visited,
    const size_t& neighbors_n,
    int* h_neighbors_id,
    const T* neighbors_flat,
    T* h_dist_from_neighbors
) {
    int blockSize = 128; // Ensure it doesn't exceed GPU limit
    int numBlocks = (neighbors_n + blockSize - 1) / blockSize; // Adjust for grid size

    // Allocate memory on the device for input and output data
    T* d_query;
    cudaMalloc(&d_query, vec_dim * sizeof(T));
    cudaMemcpy(d_query, h_query, vec_dim * sizeof(T), cudaMemcpyHostToDevice);

    bool* d_visited;
    cudaMalloc(&d_visited, ds_size * sizeof(bool));
    cudaMemcpy(d_visited, h_visited, ds_size * sizeof(bool), cudaMemcpyHostToDevice);

    int* d_neighbors_id;
    cudaMalloc(&d_neighbors_id, neighbors_n * sizeof(int));
    cudaMemcpy(d_neighbors_id, h_neighbors_id, neighbors_n * sizeof(int), cudaMemcpyHostToDevice);

    T* d_neighbors_flat;
    cudaMalloc(&d_neighbors_flat, neighbors_n * vec_dim * sizeof(T));
    cudaMemcpy(d_neighbors_flat, neighbors_flat, neighbors_n * vec_dim * sizeof(T), cudaMemcpyHostToDevice);

    T* d_dist_from_neighbors;
    cudaMalloc(&d_dist_from_neighbors, neighbors_n * sizeof(T));
    cudaMemcpy(d_dist_from_neighbors, h_dist_from_neighbors, neighbors_n * sizeof(T), cudaMemcpyHostToDevice);

    // Launch the kernel with adjusted grid size and shared memory size
    process_neighbors_kernel<T><<<numBlocks, blockSize, blockSize * sizeof(T)>>>(
        ds_size,
        vec_dim,
        d_query,
        d_visited,
        neighbors_n,
        d_neighbors_id,
        d_neighbors_flat,
        d_dist_from_neighbors
    );
    
    // Check for CUDA kernel errors
    cudaSafeCall(cudaDeviceSynchronize());

    // Copy results back to host
    cudaMemcpy(h_dist_from_neighbors, d_dist_from_neighbors, neighbors_n * sizeof(T), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_visited, d_visited, ds_size * sizeof(bool), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_query);
    cudaFree(d_visited);
    cudaFree(d_neighbors_id);
    cudaFree(d_neighbors_flat);
    cudaFree(d_dist_from_neighbors);
}

#endif // HNSW_PROCESS_NEIGHBORS_CUH
