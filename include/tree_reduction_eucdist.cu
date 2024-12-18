#include "tree_reduction_eucdist.cuh"

template class EuclideanDistanceTree<float>;

template <typename T>
__global__ void kernel(
    const T* vec1,
    size_t* n_vec1,
    const T* vec2,
    size_t* n_vec2,
    T* distances,
    size_t* dimension
) {
    __syncthreads();

    // tree reduction based euclidean distance
    extern __shared__ T sharedData[];

    int tid = threadIdx.x;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;
    int bidz = blockIdx.z;
    
    int vecid = bidx * blockDim.x + tid;
    int globalId1 = bidy * (*dimension) + vecid;
    int globalId2 = bidz * (*dimension) + vecid;

    int distid = bidy * (*n_vec2) + bidz;

    if (vecid < (*dimension)) {
        float diff = vec1[globalId1] - vec2[globalId2];
        sharedData[tid] = diff * diff;
    }
    else {
        sharedData[tid] = 0.0;
    }

    __syncthreads();

    // Perform tree-based reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }
    
    __syncthreads();

    if (tid == 0) {
        // print block indexes xyz and shared data value
        printf("Block indexes: (%d, %d, %d), shared data: %f\n", bidx, bidy, bidz, sharedData[0]);
        atomicAdd(&distances[distid], sharedData[0]);
    }

    __syncthreads();

    if (vecid == 0) {
        printf("Distances (%d, %d): %f\n", bidy, bidz, distances[distid]);
        distances[distid] = sqrtf(distances[distid]);
    }

    __syncthreads();
}

__global__ void simpleKernel() {
    printf("Hello from thread %d!\n", threadIdx.x);
}

template <typename T>
std::vector<std::vector<T>> EuclideanDistanceTree<T>::compute() {
    const size_t& h_matrix1_n = this->matrix1_n;
    const size_t& h_matrix2_n = this->matrix2_n;
    const size_t& h_dimension = this->dimension;
    size_t* d_matrix1_n = this->d_matrix1_n;
    size_t* d_matrix2_n = this->d_matrix2_n;
    size_t* d_dimension = this->d_dimension;
    const T* h_matrix1 = this->h_matrix1;
    const T* h_matrix2 = this->h_matrix2;
    T* h_distances = this->h_distances;
    T* d_matrix1 = this->d_matrix1;
    T* d_matrix2 = this->d_matrix2;
    T* d_distances = this->d_distances;

    this->start = std::chrono::high_resolution_clock::now();

    // Allocate and copy memory
    cudaMalloc((void**)&d_matrix1_n, sizeof(size_t));
    cudaMemcpy((void*)d_matrix1_n, (void*)&h_matrix1_n, sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_matrix1, h_matrix1_n * h_dimension * sizeof(T));
    cudaMemcpy(d_matrix1, h_matrix1, h_matrix1_n * h_dimension * sizeof(T), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_matrix2_n, sizeof(size_t));
    cudaMemcpy((void*)d_matrix2_n, (void*)&h_matrix2_n, sizeof(size_t), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_matrix2, h_matrix2_n * h_dimension * sizeof(T));
    cudaMemcpy(d_matrix2, h_matrix2, h_matrix2_n * h_dimension * sizeof(T), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_distances, h_matrix1_n * h_matrix2_n * sizeof(T));

    cudaMalloc((void**)&d_dimension, sizeof(size_t));
    cudaMemcpy((void*)d_dimension, (void*)&h_dimension, sizeof(size_t), cudaMemcpyHostToDevice);

    // Launch kernel
    kernel<<<this->gridDim, this->blockSize, this->blockSize.x * sizeof(T)>>>(
        d_matrix1,
        d_matrix1_n,
        d_matrix2,
        d_matrix2_n,
        d_distances,
        d_dimension
    );
    // simpleKernel<<<this->gridDim, this->blockSize>>>();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Kernel error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_distances, d_distances, h_matrix1_n * h_matrix2_n * sizeof(T), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree((void*)d_matrix1_n);
    cudaFree(d_matrix1);
    cudaFree((void*)d_matrix2_n);
    cudaFree(d_matrix2);
    cudaFree(d_distances);
    cudaFree((void*)d_dimension);

    this->end = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<T>> distances_vec;
    for (size_t i = 0; i < h_matrix1_n; ++i) {
        std::vector<T> row;
        for (size_t j = 0; j < h_matrix2_n; ++j) {
            row.push_back(h_distances[i * h_matrix2_n + j]);
        }
        distances_vec.push_back(row);
    }

    return distances_vec;
}

template <typename T>
EuclideanDistanceTree<T>::EuclideanDistanceTree(
    const std::vector<std::vector<T>>& v1,
    const std::vector<std::vector<T>>& v2,
    size_t block_size
) : EuclideanDistanceCUDA<T>(v1, v2, block_size) {}