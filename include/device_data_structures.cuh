#ifndef HNSW_DEVICE_DATA_STRUCTURES_CUH
#define HNSW_DEVICE_DATA_STRUCTURES_CUH

#include <cuda_runtime.h>

template <typename T = float>
struct d_Data {
    T* x;
    int id;

    // Default constructor
    __host__ __device__ d_Data() : x(nullptr), id(-1) {}

    // Parameterized constructor
    __host__ __device__ d_Data(T* arr, int id) : x(arr), id(id) {}

    // Copy constructor
    __host__ __device__ d_Data(const d_Data& other) 
        : x(other.x), id(other.id) {}

    // Move constructor
    __host__ __device__ d_Data(d_Data&& other) noexcept 
        : x(other.x), id(other.id) {
        other.x = nullptr; // Leave other in a valid state
        other.id = -1;
    }

    // Copy assignment operator
    __host__ __device__ d_Data& operator=(const d_Data& other) {
        if (this != &other) {
            x = other.x;
            id = other.id;
        }
        return *this;
    }

    // Move assignment operator
    __host__ __device__ d_Data& operator=(d_Data&& other) noexcept {
        if (this != &other) {
            x = other.x;
            id = other.id;

            other.x = nullptr; // Leave other in a valid state
            other.id = -1;
        }
        return *this;
    }

    // x getter
    __host__ __device__ T* data() const {
        return x;
    }
};


template <typename T = float>
struct d_Neighbor {
    T dist;
    int id;

    __host__ __device__ d_Neighbor() : dist(0), id(-1) {}

    __host__ __device__ d_Neighbor(T dist, int id) : dist(dist), id(id) {}

    // Explicit copy constructor for CUDA
    __host__ __device__ d_Neighbor(const d_Neighbor<T>& other) : dist(other.dist), id(other.id) {}

    // Copy assignment operator
    __host__ __device__ d_Neighbor<T>& operator=(const d_Neighbor<T>& other) {
        if (this != &other) {
            dist = other.dist;
            id = other.id;
        }
        return *this;
    }

    // Move assignment operator
    __host__ __device__ d_Neighbor<T>& operator=(d_Neighbor<T>&& other) noexcept {
        if (this != &other) {
            dist = std::move(other.dist);
            id = std::move(other.id);
        }
        return *this;
    }

    // Explicit move constructor
    __host__ __device__ d_Neighbor(d_Neighbor<T>&& other) noexcept : dist(std::move(other.dist)), id(std::move(other.id)) {}
};

template <typename T = float>
struct d_Node {
    d_Data<T> data;
    d_Neighbor<T>* neighbors;
    int n_neighbors;

    __host__ __device__ d_Node() : data(), neighbors(nullptr), n_neighbors(0) {}

    __host__ __device__ d_Node(T* arr, int id, d_Neighbor<T>* neighbor_arr, int n_neighbors) : 
        data(arr, id),
        neighbors(neighbor_arr),
        n_neighbors(n_neighbors) {}

    // __host__ __device__ ~d_Node() {
    //     if (neighbors != nullptr) {
    //         delete[] neighbors;
    //     }
    // }
}

#endif // HNSW_DEVICE_DATA_STRUCTURES_CUH