#ifndef HNSW_SEARCH_LAYER_CUH
#define HNSW_SEARCH_LAYER_CUH

#include <cuda_runtime.h>
#include "priority_queue.cuh"
#include "euclidean_distance.cuh"
#include "data_structures.cuh"
#include "utils.cuh"

using namespace ds;

#define K 32
#define VEC_DIM 128
#define DATASET_SIZE 10000

template <typename T = float>
__global__ void search_layer_kernel(
    int* start_node_id,
    T* query,
    int* adjaecy_list,
    T* dataset,
    int* ef,
    d_Neighbor<T>* result
) {
    int tid = threadIdx.x;

    // Shared memory for query data
    static __shared__ T shared_query[VEC_DIM];
    if (tid < VEC_DIM) {
        shared_query[tid] = query[tid];
    }
    __syncthreads();

    // Priority queues initialization
    __shared__ d_Neighbor<T> candidates_array[MAX_HEAP_SIZE];
    __shared__ int candidates_size;
    candidates_size = 0;
    PriorityQueue<T> q(candidates_array, &candidates_size, MIN_HEAP);

    __shared__ d_Neighbor<T> top_candidates_array[MAX_HEAP_SIZE];
    __shared__ int top_candidates_size;
    top_candidates_size = 0;
    PriorityQueue<T> topk(top_candidates_array, &top_candidates_size, MAX_HEAP);

    // calculate distance from start node to query and add to queue
    T start_dist = euclidean_distance_gpu<T>(
        shared_query,
        dataset + (*start_node_id) * VEC_DIM,
        VEC_DIM
    );
    __syncthreads();

    if (tid == 0) {
        q.insert({start_dist, *start_node_id});
    }
    __syncthreads();

    // Shared memory for visited nodes
    bool shared_visited[DATASET_SIZE];
    if (tid == 0) {
        shared_visited[*start_node_id] = true;
    }

    // Main loop
    while (q.get_size() > 0) {
        __syncthreads();

        // Get the current node information
        d_Neighbor<T> now = q.pop();
        int* neighbors_ids = adjaecy_list + now.id * K;

        if (tid == 0) {
            // Mark current node as visited
            shared_visited[now.id] = true;
        }

        int n_neighbors = 0;
        while (neighbors_ids[n_neighbors] != -1 && n_neighbors < K) {
            // count number of neighbors in current node
            n_neighbors++;
        }

        // Copy neighbor data to shared memory (test if this is faster)
        static __shared__ T shared_neighbor_data[K * VEC_DIM];
        for (size_t i = 0; i < n_neighbors; i++) {
            shared_neighbor_data[i * VEC_DIM + tid] = dataset[neighbors_ids[i] * VEC_DIM + tid];
        }
        __syncthreads();

        // Update topk
        if (topk.get_size() == *ef && topk.top().dist < now.dist) {
            break;
        } else if (tid == 0) {
            topk.insert(now);
        }
        __syncthreads();

        // distance computation (convert to bulk?)
        static __shared__ T shared_distances[K];
        for (size_t i = 0; i < n_neighbors; i++) {
            T dist = euclidean_distance_gpu<T>(
                shared_query,
                shared_neighbor_data + i * VEC_DIM,
                VEC_DIM
            );
            __syncthreads();
            if (tid == 0) {
                shared_distances[i] = dist;
            }
        }
        __syncthreads();
        
        // Update priority queues
        if (tid == 0) {
            for (size_t i = 0; i < n_neighbors; i++) {
                if (!shared_visited[neighbors_ids[i]]) {
                    shared_visited[neighbors_ids[i]] = true;
                    q.insert({shared_distances[i], neighbors_ids[i]});
                }
            }
        }
        __syncthreads();
    }

    // Flush the heap and get results
    if (tid == 0) {
        while (topk.get_size() > *ef) {
            topk.pop();
        }
        for (size_t i = 0; i < *ef; i++) {
            result[i] = topk.pop();
        }
    }
}

template <typename T = float>
auto search_layer_launch(
    const Data<T>& query,
    const int& start_node_id,
    const int& ef,
    const std::vector<Layer>& layers,
    const int& l_c,
    const size_t& ds_size,
    const Dataset<T>& dataset
) {
    // Allocate start node id on device
    int* d_start_node_id;
    cudaMalloc(&d_start_node_id, sizeof(int));
    cudaMemcpy(d_start_node_id, &start_node_id, sizeof(int), cudaMemcpyHostToDevice);

    // Allocate query on device
    T* d_query;
    cudaMalloc(&d_query, VEC_DIM * sizeof(T));
    cudaMemcpy(d_query, query.data(), VEC_DIM * sizeof(T), cudaMemcpyHostToDevice);

    // Allocate fixed degree graph adjency list on device
    int* d_adjaency_list;
    cudaMalloc(&d_adjaency_list, ds_size * K * sizeof(int));
    for (Node node: layers[l_c]) {
        int neighbors_ids[K];
        std::fill(neighbors_ids, neighbors_ids + K, -1);
        int node_neighbors_size = node.neighbors.size();
        if (node_neighbors_size > 0) {
            for (size_t i = 0; i < node_neighbors_size; i++) {
                neighbors_ids[i] = node.neighbors[i].id;
            }
            cudaMemcpy(d_adjaency_list + node.data.id() * K, neighbors_ids, K * sizeof(int), cudaMemcpyHostToDevice);
        }
    }    

    // Allocate dataset on device
    T* d_dataset;
    cudaMalloc(&d_dataset, ds_size * VEC_DIM * sizeof(T));
    for (size_t i = 0; i < ds_size; i++) {
        cudaMemcpy(d_dataset + i * VEC_DIM, dataset[i].data(), VEC_DIM * sizeof(T), cudaMemcpyHostToDevice);
    }

    // Allocate Target output lenght
    int* d_ef;
    cudaMalloc(&d_ef, sizeof(int));
    cudaMemcpy(d_ef, &ef, sizeof(int), cudaMemcpyHostToDevice);

    // Allocate result on device
    d_Neighbor<T> result[ef];
    d_Neighbor<T>* d_result;
    cudaMalloc(&d_result, ef * sizeof(d_Neighbor<T>));

    // Launch kernel
    search_layer_kernel<<<1, VEC_DIM>>>(
        d_start_node_id,
        d_query,
        d_adjaency_list,
        d_dataset,
        d_ef,
        d_result
    );

    

    // copy result back to host
    cudaMemcpy(result, d_result, ef * sizeof(d_Neighbor<T>), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_start_node_id);
    cudaFree(d_query);
    cudaFree(d_adjaency_list);
    cudaFree(d_dataset);
    cudaFree(d_ef);
    cudaFree(d_result);

    // Prepare output
    SearchResult search_result = SearchResult();
    for (size_t i = 0; i < ef; i++) {
        search_result.result.emplace_back(result[i].dist, result[i].id);
    }
    std::sort(search_result.result.begin(), search_result.result.end(), CompLess());

    return search_result;
}

#endif // HNSW_SEARCH_LAYER_CUH