#ifndef HNSW_SEARCH_LAYER_CUH
#define HNSW_SEARCH_LAYER_CUH

#include <cuda_runtime.h>
#include "priority_queue.cuh"
#include "euclidean_distance.cuh"
#include "data_structures.cuh"
#include "utils.cuh"

#define EUCDIST_THREADS 32

template <typename T = float>
__global__ void search_layer_kernel(
    size_t* layer_len,
    T** layer_data_map,
    int* layer_n_neighbors,
    size_t* neighbors_map_size,
    int** neighbors_map,
    T* query,
    int* start_node_id,
    int* ef,
    size_t* ds_size,
    size_t* vec_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    bool visited[*ds_size] = {false};
    visited[*start_node_id] = true;

    d_Neighbor<T> candidates_array[*ef]; // TODO: check if here is ef or k
    PriorityQueue<T> candidates(candidates_array);

    d_Neighbor<T> top_candidates_array[*ef]; // TODO: check if here is ef or k
    PriorityQueue<T> top_candidates(top_candidates_array);

    size_t numBlocks = (*vec_dim + EUCDIST_THREADS - 1) / EUCDIST_THREADS;
    T block_distances[numBlocks];
    euclidean_distance_kepler<<<numBlocks, EUCDIST_THREADS>>>(
        query,
        layer_data_map[*start_node_id],
        block_distances,
        *vec_dim
    );

    T dist_from_en = 0;
    for (size_t i = 0; i < numBlocks; i++) {
        dist_from_en += block_distances[i];
    }

    d_Neighbor<T> entry_point(dist_from_en, *start_node_id);
    candidates.insert(entry_point);
    top_candidates.insert(entry_point);

    while (candidates.get_size() > 0) {
        // this is wrong because current priority queue implementation holds floats and not Neighbors object
        d_Neighbor<T> nearest_candidate = candidates[0];

        if (nearest_candidate.dist > top_candidates[0].dist) break;

        int neighbors_n = layer_n_neighbors[nearest_candidate.id];
        if (neighbors_n > 0) {
            T dist_from_neighbors[neighbors_n];
            process_neighbors<<<1, neighbors_n>>>(
                query,
                layer_data_map,
                vec_dim,
                neighbors_map[nearest_candidate.id],
                dist_from_neighbors
            );

            update_queues<<<1, neighbors_n>>>(
                visited,
                dist_from_neighbors,
                candidates,
                top_candidates,
                neighbors_map[nearest_candidate.id],
                ef
            );
        }


    }

}

template <typename T = float>
__device__ void process_neighbors(
    const T* query,
    const T** layer_data_map,
    const int* vec_dim,
    const int* neighbors_id,
    T* distances
) {
    // Kernel built to be run in a single block

    int tid = threadIdx.x;

    const size_t numBlocks = (*vec_dim + EUCDIST_THREADS - 1) / EUCDIST_THREADS;
    T shared_distances[numBlocks]; 
    // TODO: find a way to use shared memory to store distances

    euclidean_distance_kepler<<<numBlocks, EUCDIST_THREADS>>>(
        query,
        layer_data_map[neighbors_id[tid]],
        shared_distances,
        *vec_dim
    );

    T dist = 0;
    for (size_t i = 0; i < numBlocks; i++) {
        dist += shared_distances[i];
    }
    distances[tid] = dist;
}

template <typename T = float>
__device__ void update_queues(
    bool* visited,
    const T* dist_from_neighbors,
    PriorityQueue<T>* candidates,
    PriorityQueue<T>* top_candidates,
    const int* neighbors_id,
    const int* pq_capacity
) {
    int tid = threadIdx.x;

    if (!visited[neighbors_id[tid]]) {
        if (dist_from_neighbors[tid] < top_candidates[0].dist ||
            top_candidates->get_size() < *pq_capacity) {
            d_Neighbor<T> new_candidate(dist_from_neighbors[tid], neighbors_id[tid]);
            candidates->insert(new_candidate);
            top_candidates->insert(new_candidate);
        }
    }
}

template <typename T = float>
SearchResult search_layer_launch(
    const Data<T>& query,
    int start_node_id,
    int ef,
    Layer& layer,
    size_t ds_size
) {
    auto result = SearchResult();

    int vec_dim = query.size();
    
    size_t layer_len = layer.size();
    T** layer_data = new T[layer_len];
    int* layer_n_neighbors = new int[layer_len];
    size_t neighbors_map_size = 0;

    int** neighbors_map = new int*[layer_len];
    for (size_t i = 0; i < layer_len; i++) {
        layer_data[i] = layer[i].data.data(); //doing this in a weird way (probable source of error)
        layer_n_neighbors[i] = layer[i].neighbors.size();
        neighbors_map[i] = new int[layer_n_neighbors[i]];
        for (size_t j = 0; j < layer_n_neighbors[i]; j++) {
            neighbors_map[i][j] = layer[i].neighbors[j].id;
        }
        neighbors_map_size += layer_n_neighbors[i];
    }

    int* start_node_id_ptr = &start_node_id;
    int* ef_ptr = &ef;
    
    size_t* d_layer_len;
    T** d_layer_data_map;
    int* d_layer_n_neighbors;
    size_t* d_neighbors_map_size;
    int** d_neighbors_map;
    T* d_query;
    int* d_start_node_id;
    int* d_ef;
    size_t* d_ds_size;
    size_t* d_vec_dim;

    // Allocate memory on the device
    cudaMalloc(&d_layer_len, sizeof(size_t));
    cudaMalloc(&d_layer_data_map, layer_len * sizeof(T*));
    for (size_t i = 0; i < layer_len; i++) {
        cudaMalloc(&d_layer_data_map[i], vec_dim * sizeof(T));
    }
    cudaMalloc(&d_layer_n_neighbors, layer_len * sizeof(int));
    cudaMalloc(&d_neighbors_map_size, sizeof(size_t));
    cudaMalloc(&d_neighbors_map, layer_len * sizeof(int*));
    for (size_t i = 0; i < layer_len; i++) {
        cudaMalloc(&d_neighbors_map[i], layer_n_neighbors[i] * sizeof(int));
    }
    cudaMalloc(&d_query, vec_dim * sizeof(T));
    cudaMalloc(&d_start_node_id, sizeof(int));
    cudaMalloc(&d_ef, sizeof(int));
    cudaMalloc(&d_ds_size, sizeof(size_t));
    cudaMalloc(&d_vec_dim, sizeof(size_t));

    // Copy data to the device
    cudaMemcpy(d_layer_len, &layer_len, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_data_map, layer_data, layer_len * sizeof(T*), cudaMemcpyHostToDevice);
    for (size_t i = 0; i < layer_len; i++) {
        cudaMemcpy(d_layer_data_map[i], layer_data[i].data(), vec_dim * sizeof(T), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_layer_n_neighbors, layer_n_neighbors, layer_len * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors_map_size, &neighbors_map_size, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neighbors_map, neighbors_map, layer_len * sizeof(int*), cudaMemcpyHostToDevice);
    for (size_t i = 0; i < layer_len; i++) {
        cudaMemcpy(d_neighbors_map[i], neighbors_map[i], layer_n_neighbors[i] * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_query, query.data(), vec_dim * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_node_id, start_node_id_ptr, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ef, ef_ptr, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ds_size, &ds_size, sizeof(size_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec_dim, &query.size(), sizeof(size_t), cudaMemcpyHostToDevice);

    // Launch kernel
    search_layer_kernel<<<1, 1>>>(
        d_layer_len,
        d_layer_data_map,
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
    cudaFree(d_layer_data_map);
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

#endif // HNSW_SEARCH_LAYER_CUH