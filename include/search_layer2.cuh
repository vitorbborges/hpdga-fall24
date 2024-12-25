#ifndef HNSW_SEARCH_LAYER2_CUH
#define HNSW_SEARCH_LAYER2_CUH

#include <cuda_runtime.h>
#include "priority_queue.cuh"
#include "euclidean_distance.cuh"
#include "data_structures.cuh"
#include "device_data_structures.cuh"
#include "utils.cuh"

#define K 32
#define VEC_DIM 128
#define DATASET_SIZE 10000

template <typename T = float>
__global__ void search_layer_kernel(
    int* start_node_id,
    T* query,
    int* adjaecy_list,
    T* dataset,
    int* ef
) {
    int tid = threadIdx.x;

    // Shared memory for query data
    __shared__ T shared_query[VEC_DIM];
    if (tid < VEC_DIM) {
        shared_query[tid] = query[tid];
    }
    __syncthreads();

    // Priority queues initialization
    static __shared__ d_Neighbor<T> candidates_array[MAX_HEAP_SIZE];
    static __shared__ int* candidates_size;
    PriorityQueue<T> q(candidates_array, candidates_size, MIN_HEAP);

    static __shared__ d_Neighbor<T> top_candidates_array[MAX_HEAP_SIZE];
    static __shared__ int* top_candidates_size;
    PriorityQueue<T> topk(top_candidates_array, top_candidates_size, MAX_HEAP);

    // calculate distance from start node to query and add to queue
    T start_dist = euclidean_distance_gpu<T>(
        shared_query,
        dataset + (*start_node_id) * VEC_DIM,
        VEC_DIM
    );

    printf("start_dist: %f\n", start_dist);


    if (tid == 0) {
        q.insert({start_dist, *start_node_id});
    }

    printf("f\n");


    // Shared memory for visited nodes
    static __shared__ bool shared_visited[DATASET_SIZE];
    if (tid == 0) {
        shared_visited[*start_node_id] = true;
    }

    printf("g\n");

    // Main loop
    bool q_empty = false;
    while (!q_empty) {
        __syncthreads();

        // Get the current node information
        d_Neighbor<T> now;
        if (tid == 0) {
            now = q.pop();
        }

        int* neighbors_ids = adjaecy_list + now.id * K;

        if (tid == 0) {
            // Mark current node as visited
            shared_visited[now.id] = true;
        }

        size_t n_neighbors = 0;
        while (neighbors_ids[n_neighbors] != -1) {
            n_neighbors++;
        }

        // Copy neighbor data to shared memory
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

        // distance computation
        static __shared__ T shared_distances[K];
        for (size_t i = 0; i < n_neighbors; i++) {
            shared_distances[i] = euclidean_distance_gpu<T>(
                shared_query,
                shared_neighbor_data + i * VEC_DIM,
                VEC_DIM
            );
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
            // check if queue is empty
            q_empty = q.get_size() == 0;
        }
        __syncthreads();
    }
    // Print final heap
    if (tid == 0) {
        printf("final heap: ");
        topk.print_heap();
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
    for (Node node: layers[0]) {
        int neighbors_ids[K] = {-1};
        int node_neighbors_size = node.neighbors.size();
        for (int i = 0; i < node_neighbors_size; i++) {
            neighbors_ids[i] = node.neighbors[i].id;
        }
        cudaMemcpy(d_adjaency_list + node.data.id() * K, neighbors_ids, K * sizeof(int), cudaMemcpyHostToDevice);
    

    // Allocate dataset on device
    T* d_dataset;
    cudaMalloc(&d_dataset, ds_size * VEC_DIM * sizeof(T));
    for (size_t i = 0; i < ds_size; i++) {
        cudaMemcpy(d_dataset + i * VEC_DIM, dataset[i].data(), VEC_DIM * sizeof(T), cudaMemcpyHostToDevice);
    }

    // Target output lenght
    int* d_ef;
    cudaMalloc(&d_ef, sizeof(int));
    cudaMemcpy(d_ef, &ef, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    search_layer_kernel<<<1, VEC_DIM>>>(
        d_start_node_id,
        d_query,
        d_adjaency_list,
        d_dataset,
        d_ef
    );

    // Free memory
    cudaFree(d_start_node_id);
    cudaFree(d_query);
    cudaFree(d_adjaency_list);
    cudaFree(d_dataset);
    cudaFree(d_ef);

    return 0;
}


#endif // HNSW_SEARCH_LAYER2_CUH