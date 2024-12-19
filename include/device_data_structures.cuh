#ifndef HNSW_DEVICE_DATA_STRUCTURES_CUH
#define HNSW_DEVICE_DATA_STRUCTURES_CUH

#include <cuda_runtime.h>

template <typename T = float>
struct d_Data {
    T* x;
    int id;

    __device__ d_Data(T* x, int id) : x(x), id(id) {}
};

template <typename T = float>
struct d_Neighbor {
    T dist;
    int id;

    __device__ d_Neighbor(T dist, int id) : dist(dist), id(id) {}
};

#endif // HNSW_DEVICE_DATA_STRUCTURES_CUH