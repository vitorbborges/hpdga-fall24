#include <cuda_runtime.h>
#include "priority_queue.cuh"
#include "euclidean_distance.cuh"
#include "data_structures.cuh"
#include "utils.cuh"

#define EUCDIST_THREADS 32

template <typename T = float>
__global__ void search_layer_kernel(
    size_t* layer_len,
    Data<T>* layer_data,
    int* layer_n_neighbors,
    size_t* neighbors_map_size,
    int** neighbors_map,
    Data<T>* query,
    int* start_node_id,
    int* ef,
    size_t* ds_size,
    size_t* vec_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    bool visited[*ds_size] = {false};
    visted[*start_node_id] = true;

    T candidates[*ef];
    int candidates_size = 0;

    T top_candidates[*ef];
    int top_candidates_size = 0;

    size_t numBlocks = (*vec_dim + EUCDIST_THREADS - 1) / EUCDIST_THREADS;
    float block_distances[numBlocks];
    const T dist_from_en = euclidean_distance_kepler<<<numBlocks, EUCDIST_THREADS>>>(
        query->data(),
        layer_data[*start_node_id].data(),
        block_distances,
        *vec_dim
    );

    float dist_from_en = 0;
    for (size_t i = 0; i < numBlocks; i++) {
        dist_from_en += block_distances[i];
    }

    insert(candidates, candidates_size, dist_from_en);
    insert(top_candidates, top_candidates_size, dist_from_en);

    while (candidates_size > 0) {
        // this is wrong because current priority queue implementation holds floats and not Neighbors object
        T current_dist = candidates[0];
        int current_id = candidates[0].id; 
        pop_max(candidates, candidates_size);

        if (current_dist > top_candidates[0]) {
            break;
        }

        // TODO: process neighbors in parallel

        int n_neighbors = layer_n_neighbors[current_id];

        if (n_neighbors > 0) {
            T distances[n_neighbors];
            process_neighbors<<<1, n_neighbors>>>(
                query->data(),
                layer_data,
                vec_dim,
                neighbors_map[current_id],
                distances
            )
        }


    }

}

template <typename T = float>
__device__ void process_neighbors(
    const Data<T>* query,
    const Data<T>* layer_data,
    const int* vec_dim,
    const int* neighbors_id,
    float* distances,
) {
    // Kernel built to be run in a single block

    int tid = threadIdx.x;

    const size_t numBlocks = (*vec_dim + EUCDIST_THREADS - 1) / EUCDIST_THREADS;
    float shared_distances[numBlocks];

    euclidean_distance_kepler<<<numBlocks, EUCDIST_THREADS>>>(
        query->data(),
        layer_data[neighbors_id[idx]].data(),
        shared_distances,
        *vec_dim
    );

    dist = 0;
    for (size_t i = 0; i < numBlocks; i++) {
        dist += shared_distances[i];
    }
    distances[tid] = dist;
}

template <typename T = float>
__device__ void update_queues() 
// check distances calculated in process_neighbors and update the queues accordingly

template <typename T = float>
SearchResult search_layer_launch(
    const Data<T>& query,
    int start_node_id,
    int ef,
    Layer& layer,
    size_t ds_size
) {
    auto result = SearchResult();
    
    size_t layer_len = layer.size();
    Data* layer_data = new Data[layer_len];
    int* layer_n_neighbors = new int[layer_len];
    size_t neighbors_map_size = 0;

    int** neighbors_map = new int*[layer_len];
    for (size_t i = 0; i < layer_len; i++) {
        layer_data[i] = layer[i].data;
        layer_n_neighbors[i] = layer[i].neighbors.size();
        neighbors_map[i] = new int[layer_n_neighbors[i]];
        for (size_t j = 0; j < layer_n_neighbors[i]; j++) {
            neighbors_map[i][j] = layer[i].neighbors[j].id;
        }
        neighbors_map_size += layer_n_neighbors[i];
    }

    Data* query_ptr = &query;
    int* start_node_id_ptr = &start_node_id;
    int* ef_ptr = &ef;
    
    size_t* d_layer_len;
    Data* d_layer_data;
    int* d_layer_n_neighbors;
    size_t* d_neighbors_map_size;
    int** d_neighbors_map;
    Data* d_query;
    int* d_start_node_id;
    int* d_ef;
    size_t* d_ds_size;
    size_t* d_vec_dim;

    // Allocate memory on the device
    cudaMalloc(&d_layer_len, sizeof(size_t));
    cudaMalloc(&d_layer_data, layer_len * sizeof(Data));
    cudaMalloc(&d_layer_n_neighbors, layer_len * sizeof(int));
    cudaMalloc(&d_neighbors_map_size, sizeof(size_t));
    cudaMalloc(&d_neighbors_map, layer_len * sizeof(int*));
    for (size_t i = 0; i < layer_len; i++) {
        cudaMalloc(&d_neighbors_map[i], layer_n_neighbors[i] * sizeof(int));
    }
    cudaMalloc(&d_query, sizeof(Data));
    cudaMalloc(&d_start_node_id, sizeof(int));
    cudaMalloc(&d_ef, sizeof(int));
    cudaMalloc(&d_ds_size, sizeof(size_t));
    cudaMalloc(&d_vec_dim, sizeof(size_t));

    // Copy data to the device
    cudaMemcpy(d_layer_len, &layer_len, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_data, layer_data, layer_len * sizeof(Data), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_n_neighbors, layer_n_neighbors, layer_len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors_map_size, &neighbors_map_size, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors_map, neighbors_map, layer_len * sizeof(int*), cudaMemcpyHostToDevice);
    for (size_t i = 0; i < layer_len; i++) {
        cudaMemcpy(d_neighbors_map[i], neighbors_map[i], layer_n_neighbors[i] * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_query, query_ptr, sizeof(Data), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_node_id, start_node_id_ptr, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ef, ef_ptr, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ds_size, &ds_size, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec_dim, &query.size(), sizeof(size_t), cudaMemcpyHostToDevice);

    // Launch kernel
    search_layer_kernel<<<1, 1>>>(
        d_layer_len,
        d_layer_data,
        d_layer_n_neighbors,
        d_neighbors_map_size,
        d_neighbors_map,
        d_query,
        d_start_node_id,
        d_ef,
        d_ds_size,
        d_vec_dim
    );

    // Free memory
    cudaFree(d_layer_len);
    cudaFree(d_layer_data);
    cudaFree(d_layer_n_neighbors);
    cudaFree(d_neighbors_map_size);
    cudaFree(d_neighbors_map);
    cudaFree(d_query);
    cudaFree(d_start_node_id);
    cudaFree(d_ef);

    delete[] layer_data;
    delete[] layer_n_neighbors;
    for (size_t i = 0; i < layer_len; i++) {
        delete[] neighbors_map[i];
    }
    delete[] neighbors_map;


}

<template class T = float>
__global__ void search_layer_kernel