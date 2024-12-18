#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

#include "cpu_eucdist.cuh"
#include "tree_reduction_eucdist.cuh"

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

// // CPU implementation of Euclidean distance
// std::vector<float> euclidean_distance_cpu(const std::vector<std::vector<float>> &vec1, const std::vector<std::vector<float>> &vec2) {
//     int num_vectors = vec1.size();
//     int dimensions = vec1[0].size();
//     std::vector<float> distances(num_vectors);

//     for (int i = 0; i < num_vectors; ++i) {
//         float sum = 0.0;
//         for (int j = 0; j < dimensions; ++j) {
//             float diff = vec1[i][j] - vec2[i][j];
//             sum += diff * diff;
//         }
//         distances[i] = sqrt(sum);
//     }

//     return distances;
// }

// // CUDA kernel for Euclidean distance
// __global__ void euclidean_distance_gpu(const float *vec1, const float *vec2, float *distances, int num_vectors, int dimensions) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if (idx < num_vectors) {
//         float sum = 0.0;
//         for (int j = 0; j < dimensions; ++j) {
//             float diff = vec1[idx * dimensions + j] - vec2[idx * dimensions + j];
//             sum += diff * diff;
//         }
//         distances[idx] = sqrt(sum);
//     }
// }


// // using shared memory and interleaving addresses without divergent branching
// _global_ void euclidean_distance_gpu_opt1(const float* vec1, const float* vec2, float* distances, int num_vectors, int dimensions) {
//     __syncthreads();
//     extern _shared_ float sdata[];  // Shared memory for reduction

//     unsigned int tid = threadIdx.x;
//     unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx >= num_vectors) return;

//     // Load squared differences into shared memory
//     float sum = 0.0f;
//     for (int j = tid; j < dimensions; j += blockDim.x) {
//         float diff = vec1[idx * dimensions + j] - vec2[idx * dimensions + j];
//         sum += diff * diff;
//     }
//     sdata[tid] = sum;
//     __syncthreads();

//     // Perform reduction in shared memory
//     // with divergence at around 231400 ns
//     /*
//     for (unsigned int s = 1; s < blockDim.x; s *= 2) {
//         if (tid % (2 * s) == 0 && tid + s < blockDim.x) {
//             sdata[tid] += sdata[tid + s];
//         }
//         __syncthreads();
//     }
//     */

//     // without divergent branching achieves 175100 ns on average
//     /*
//     for (unsigned int s = 1; s < blockDim.x; s *= 2) {
//         int index = 2 * s * tid;
//         if (index < blockDim.x) {
//             sdata[index] += sdata[index + s];
//         }
//         __syncthreads();
//     }
//     */

//     // sequential addressing achieving average of 145400 ns
//     for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//         if (tid < s) {
//             sdata[tid] += sdata[tid + s];
//         }
//         __syncthreads();
//     }


//     // Write the final result to global memory
//     if (tid == 0) {
//         distances[blockIdx.x] = sqrtf(sdata[0]);
//     }
// }


// // reaches up to 78800 ns but is currently not giving the correct output
// _global_ void euclidean_distance_opt2(const float* vec1, const float* vec2, float* distances, int num_vectors, int dimensions) {
//     extern _shared_ float sharedData[];

//     unsigned int tid = threadIdx.x;
//     unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx >= num_vectors) return;  // Ensure we don't process out-of-bounds vectors

//     float sum = 0.0f;

//     // Compute squared differences for this vector pair
//     for (int j = tid; j < dimensions; j += blockDim.x) {
//         float diff = vec1[idx * dimensions + j] - vec2[idx * dimensions + j];
//         sum += diff * diff;
//     }

//     // Store partial sum in shared memory
//     sharedData[tid] = sum;
//     __syncthreads();

//     // Perform tree-based reduction in shared memory
//     for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
//         if (tid < stride) {
//             sharedData[tid] += sharedData[tid + stride];
//         }
//         __syncthreads();
//     }

//     // Write final distance to global memory
//     if (tid == 0) {
//         distances[idx] = sqrtf(sharedData[0]);
//     }
// }

// Euclidean distance function for two 128-dimensional vectors
float compute_euclidean_distance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    float sum = 0.0;
    for (int i = 0; i < vec1.size(); ++i) {
        sum += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }
    return std::sqrt(sum);
}

// Compute distances between all pairs of vectors in vec1 and vec2
std::vector<std::vector<float>> compute_all_distances(const std::vector<std::vector<float>>& vec1, const std::vector<std::vector<float>>& vec2) {
    int num_vec1 = vec1.size();
    int num_vec2 = vec2.size();
    std::vector<std::vector<float>> distances(num_vec1, std::vector<float>(num_vec2));

    for (int i = 0; i < num_vec1; ++i) {
        for (int j = 0; j < num_vec2; ++j) {
            distances[i][j] = compute_euclidean_distance(vec1[i], vec2[j]);
        }
    }

    return distances;
}


int main() {
    const int num_vectors1 = 20;
    const int num_vectors2 = 5;
    const int dimensions = 128;

    // Initialize vectors
    std::vector<std::vector<float>> vec1(num_vectors1), vec2(num_vectors2);
    generate_vectors(vec1, num_vectors1, dimensions);
    generate_vectors(vec2, num_vectors2, dimensions);

    std::vector<std::vector<float>> expected_distances = compute_all_distances(vec1, vec2);

    // Measure CPU time
    printf("running cpu distances... \n");
    EuclideanDistanceCPU<float> cpu(vec1, vec2);
    std::vector<std::vector<float>> cpu_distances = cpu.compute();
    std::cout << "CPU computation time: " << cpu.get_duration() << " ns" << std::endl;

    // Measure Tree-based reduction GPU time
    printf("running tree reduction distances... \n");
    EuclideanDistanceTree<float> tree(vec1, vec2, 32);
    std::vector<std::vector<float>> gpu_distances = tree.compute();
    std::cout << "Tree-based reduction GPU computation time: " << tree.get_duration() << " ns" << std::endl;

    // Compare results with the expected distances
    bool valid = true;
    for (int i = 0; i < num_vectors1; ++i) {
        for (int j = 0; j < num_vectors2; ++j) {
            if (fabs(cpu_distances[i][j] - expected_distances[i][j]) > 1e-1) {
                valid = false;
                printf("Mismatch found in CPU distance at i=%d, j=%d: expected %.6f, got %.6f\n",
                    i, j, expected_distances[i][j], cpu_distances[i][j]);
            }
            if (fabs(gpu_distances[i][j] - expected_distances[i][j]) > 1e-1) {
                valid = false;
                printf("Mismatch found in GPU distance at i=%d, j=%d: expected %.6f, got %.6f\n",
                    i, j, expected_distances[i][j], gpu_distances[i][j]);
            }
        }
    }

    if (valid) {
        printf("All results match the expected distances!\n");
    } else {
        printf("Results do not match the expected distances.\n");
    }

    return 0;
}