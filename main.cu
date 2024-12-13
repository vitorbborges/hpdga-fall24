#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

// Function to generate random vectors
void generate_vectors(std::vector<std::vector<float>> &vec, int num_vectors, int dimensions) {
    for (int i = 0; i < num_vectors; ++i) {
        vec[i].resize(dimensions);
        for (int j = 0; j < dimensions; ++j) {
            vec[i][j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
    printf("finished generating vectors... \n");
}

// CPU implementation of Euclidean distance
std::vector<float> euclidean_distance_cpu(const std::vector<std::vector<float>> &vec1, const std::vector<std::vector<float>> &vec2) {
    int num_vectors = vec1.size();
    int dimensions = vec1[0].size();
    std::vector<float> distances(num_vectors);

    for (int i = 0; i < num_vectors; ++i) {
        float sum = 0.0;
        for (int j = 0; j < dimensions; ++j) {
            float diff = vec1[i][j] - vec2[i][j];
            sum += diff * diff;
        }
        distances[i] = sqrt(sum);
    }

    return distances;
}

// CUDA kernel for Euclidean distance
_global_ void euclidean_distance_gpu(const float *vec1, const float *vec2, float *distances, int num_vectors, int dimensions) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_vectors) {
        float sum = 0.0;
        for (int j = 0; j < dimensions; ++j) {
            float diff = vec1[idx * dimensions + j] - vec2[idx * dimensions + j];
            sum += diff * diff;
        }
        distances[idx] = sqrt(sum);
    }
}


// using shared memory and interleaving addresses without divergent branching
_global_ void euclidean_distance_gpu_opt1(const float* vec1, const float* vec2, float* distances, int num_vectors, int dimensions) {
    __syncthreads();
    extern _shared_ float sdata[];  // Shared memory for reduction

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_vectors) return;

    // Load squared differences into shared memory
    float sum = 0.0f;
    for (int j = tid; j < dimensions; j += blockDim.x) {
        float diff = vec1[idx * dimensions + j] - vec2[idx * dimensions + j];
        sum += diff * diff;
    }
    sdata[tid] = sum;
    __syncthreads();

    // Perform reduction in shared memory
    // with divergence at around 231400 ns
    /*
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0 && tid + s < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    */

    // without divergent branching achieves 175100 ns on average
    /*
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    */

    // sequential addressing achieving average of 145400 ns
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }


    // Write the final result to global memory
    if (tid == 0) {
        distances[blockIdx.x] = sqrtf(sdata[0]);
    }
}


// reaches up to 78800 ns but is currently not giving the correct output
_global_ void euclidean_distance_opt2(const float* vec1, const float* vec2, float* distances, int num_vectors, int dimensions) {
    extern _shared_ float sharedData[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_vectors) return;  // Ensure we don't process out-of-bounds vectors

    float sum = 0.0f;

    // Compute squared differences for this vector pair
    for (int j = tid; j < dimensions; j += blockDim.x) {
        float diff = vec1[idx * dimensions + j] - vec2[idx * dimensions + j];
        sum += diff * diff;
    }

    // Store partial sum in shared memory
    sharedData[tid] = sum;
    __syncthreads();

    // Perform tree-based reduction in shared memory
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    // Write final distance to global memory
    if (tid == 0) {
        distances[idx] = sqrtf(sharedData[0]);
    }
}







int main() {
    const int num_vectors = 100000;
    const int dimensions = 128;

    // Initialize vectors
    std::vector<std::vector<float>> vec1(num_vectors), vec2(num_vectors);
    generate_vectors(vec1, num_vectors, dimensions);
    generate_vectors(vec2, num_vectors, dimensions);

    // Flatten vectors for GPU
    std::vector<float> flat_vec1(num_vectors * dimensions);
    std::vector<float> flat_vec2(num_vectors * dimensions);

    for (int i = 0; i < num_vectors; ++i) {
        for (int j = 0; j < dimensions; ++j) {
            flat_vec1[i * dimensions + j] = vec1[i][j];
            flat_vec2[i * dimensions + j] = vec2[i][j];
        }
    }

    // Measure CPU time
    printf("running cpu distances... \n");
    auto start_cpu = std::chrono::high_resolution_clock::now();
    std::vector<float> cpu_distances = euclidean_distance_cpu(vec1, vec2);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::cout << "CPU computation time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu - start_cpu).count() << " ns" << std::endl;

    // Allocate memory on GPU
    float* d_vec1, * d_vec2, * d_distances;
    cudaMalloc(&d_vec1, num_vectors * dimensions * sizeof(float));
    cudaMalloc(&d_vec2, num_vectors * dimensions * sizeof(float));
    cudaMalloc(&d_distances, num_vectors * sizeof(float));

    // Copy data to GPU
    cudaMemcpy(d_vec1, flat_vec1.data(), num_vectors * dimensions * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec2, flat_vec2.data(), num_vectors * dimensions * sizeof(float), cudaMemcpyHostToDevice);

    // Measure time for the original kernel
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_vectors + threads_per_block - 1) / threads_per_block;

    printf("running original GPU distances...\n");
    auto start_gpu = std::chrono::high_resolution_clock::now();
    euclidean_distance_gpu << <blocks_per_grid, threads_per_block >> > (d_vec1, d_vec2, d_distances, num_vectors, dimensions);
    cudaDeviceSynchronize();
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::cout << "Original GPU computation time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu - start_gpu).count() << " ns" << std::endl;

    // Copy results back to CPU for verification
    std::vector<float> gpu_distances_original(num_vectors);
    cudaMemcpy(gpu_distances_original.data(), d_distances, num_vectors * sizeof(float), cudaMemcpyDeviceToHost);

    // Measure time for the optimized kernel (opt1)
    size_t shared_memory_size = threads_per_block * sizeof(float);
    printf("running optimized GPU distances (opt1)...\n");
    auto start_gpu_opt1 = std::chrono::high_resolution_clock::now();
    euclidean_distance_gpu_opt1 << <blocks_per_grid, threads_per_block, shared_memory_size >> > (d_vec1, d_vec2, d_distances, num_vectors, dimensions);
    cudaDeviceSynchronize();
    auto end_gpu_opt1 = std::chrono::high_resolution_clock::now();
    std::cout << "Optimized GPU computation time (opt1): " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu_opt1 - start_gpu_opt1).count() << " ns" << std::endl;

    // Copy results back to CPU for verification
    std::vector<float> gpu_distances_opt1(num_vectors);
    cudaMemcpy(gpu_distances_opt1.data(), d_distances, num_vectors * sizeof(float), cudaMemcpyDeviceToHost);

    // Measure time for the optimized kernel (opt2)
    printf("running optimized GPU distances (opt2)...\n");
    auto start_gpu_opt2 = std::chrono::high_resolution_clock::now();
    euclidean_distance_opt2 << <blocks_per_grid, threads_per_block, shared_memory_size >> > (d_vec1, d_vec2, d_distances, num_vectors, dimensions);
    cudaDeviceSynchronize();

    auto end_gpu_opt2 = std::chrono::high_resolution_clock::now();
    std::cout << "Optimized GPU computation time (opt2): " << std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu_opt2 - start_gpu_opt2).count() << " ns" << std::endl;

    // Copy results back to CPU for verification
    std::vector<float> gpu_distances_opt2(num_vectors);
    cudaMemcpy(gpu_distances_opt2.data(), d_distances, num_vectors * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify that all results are the same
    bool valid = true;
    for (int i = 0; i < num_vectors; ++i) {
        if (fabs(cpu_distances[i] - gpu_distances_original[i]) > 1e-5 ||
            fabs(cpu_distances[i] - gpu_distances_opt1[i]) > 1e-5 ||
            fabs(cpu_distances[i] - gpu_distances_opt2[i]) > 1e-5) {
            valid = false;
            printf("Mismatch found at index %d\n", i);
            break;
        }
    }
    if (valid) {
        printf("All results match!\n");
    }
    else {
        printf("Results do not match.\n");
    }

    for (int i = 0; i < 10; ++i) {
        printf("Index %d: CPU=%.6f, Original=%.6f, Opt1=%.6f, Opt2=%.6f\n",
            i, cpu_distances[i], gpu_distances_original[i], gpu_distances_opt1[i], gpu_distances_opt2[i]);
    }

    // Free GPU memory
    cudaFree(d_vec1);
    cudaFree(d_vec2);
    cudaFree(d_distances);

    return 0;
}