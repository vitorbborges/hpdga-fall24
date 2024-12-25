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
    d_Neighbor<T>* candidates_array,
    int* candidates_size,
    d_Neighbor<T>* top_candidates_array,
    int* top_candidates_size,
    size_t* layer_len,
    int* ef
) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Shared memory for query data
    __shared__ T shared_query[VEC_DIM];
    if (globalIdx < VEC_DIM) {
        shared_query[globalIdx] = query[globalIdx];
    }

    // Priority queues initialization
    PriorityQueue<T> q(candidates_array, candidates_size, MIN_HEAP);
    PriorityQueue<T> topk(top_candidates_array, top_candidates_size, MAX_HEAP);

    // calculate distance from start node to query
    if (bid == 0) {
        T start_dist = euclidean_opt<T>(
            shared_query,
            dataset,
            VEC_DIM,
            0,
            *start_node_id
        );

        if (globalIdx == 0) {
            q.insert({start_dist, *start_node_id});
            printf("start dist: %f\n", start_dist);
        }
    }
    __syncthreads();

    static __shared__ bool shared_visited[DATASET_SIZE];

    // Main loop
    while (q.get_size() > 0) {
        __syncthreads();

        // Get the current node information
        d_Neighbor<T> now = q.pop();
        int* neighbors_ids = adjaecy_list + now.id * K;

        size_t n_neighbors = 0;
        while (neighbors_ids[n_neighbors] != -1) {
            n_neighbors++;
        }
        __syncthreads();

        // Update topk
        if (topk.get_size() == *ef && topk.top().dist < now.dist) {
            break;
        } else if (globalIdx == 0) {
            topk.insert(now);
        }

        // BULK distance computation
        T distance[K];
        if (bid <= n_neighbors) {
            T dist = euclidean_opt(
                shared_query,
                dataset,
                VEC_DIM,
                0,
                neighbors_ids[bid]
            );
            if (tid == 0) {
                distance[bid] = dist;
            }
        }
        __syncthreads();

        // Update priority queues
        if (globalIdx == 0) {
            for (size_t i = 0; i < n_neighbors; i++) {
                if (!shared_visited[neighbors_ids[i]]) {
                    shared_visited[neighbors_ids[i]] = true;
                    q.insert({distance[i], neighbors_ids[i]});
                }
            }
        }
        __syncthreads();
    }
    // Print final heap
    if (globalIdx == 0) {
        printf("final heap: ");
        topk.print_heap();
    }
}

template <typename T = float>
__device__ void copy_array_to_shared(
    T* array,
    T* shared,
    size_t size
) {
    if (blockIdx.x * blockDim.x + threadIdx.x < size) {
        shared[threadIdx.x] = array[threadIdx.x];
    }
}

template <typename T = float>
auto search_layer_launch(
    const Data<T>& query,
    const int& start_node_id,
    const int& ef,
    Layer& layer,
    const size_t& ds_size
) {
    size_t layer_len = layer.size();

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
    cudaMalloc(&d_adjaency_list, layer_len * K * sizeof(int));
    for (Node node: layer) {
        int neighbors_ids[K] = {-1};
        for (int i = 0; i < K; i++) {
            neighbors_ids[i] = node.neighbors[i].id;
        }
        cudaMemcpy(d_adjaency_list + node.data.id() * K, neighbors_ids, K * sizeof(int), cudaMemcpyHostToDevice);
    }

    // Allocate dataset on device
    T* d_dataset;
    cudaMalloc(&d_dataset, layer_len * VEC_DIM * sizeof(T));
    for (Node node: layer) {
        cudaMemcpy(d_dataset + node.data.id() * VEC_DIM, node.data.data(), VEC_DIM * sizeof(T), cudaMemcpyHostToDevice);
    }

    // Allocate priority queues on device
    d_Neighbor<T>* candidates_array;
    cudaMalloc(&candidates_array, MAX_HEAP_SIZE * sizeof(d_Neighbor<T>));
    int* candidates_size;
    cudaMalloc(&candidates_size, sizeof(int));

    d_Neighbor<T>* top_candidates_array;
    cudaMalloc(&top_candidates_array, MAX_HEAP_SIZE * sizeof(d_Neighbor<T>));
    int* top_candidates_size;
    cudaMalloc(&top_candidates_size, sizeof(int));

    // Allocate number vectors in current layer on device
    size_t* d_layer_len;
    cudaMalloc(&d_layer_len, sizeof(size_t));
    cudaMemcpy(d_layer_len, &layer_len, sizeof(size_t), cudaMemcpyHostToDevice);

    // Target output lenght
    int* d_ef;
    cudaMalloc(&d_ef, sizeof(int));
    cudaMemcpy(d_ef, &ef, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    search_layer_kernel<<<K, VEC_DIM>>>(
        d_start_node_id,
        d_query,
        d_adjaency_list,
        d_dataset,
        candidates_array,
        candidates_size,
        top_candidates_array,
        top_candidates_size,
        d_layer_len,
        d_ef
    );


    // Free memory
    cudaFree(d_adjaency_list);
    cudaFree(d_dataset);
    cudaFree(d_query);
    cudaFree(d_start_node_id);
    cudaFree(candidates_array);
    cudaFree(candidates_size);
    cudaFree(top_candidates_array);
    cudaFree(top_candidates_size);
    cudaFree(d_layer_len);
    cudaFree(d_ef);

    return 0;
}


#endif // HNSW_SEARCH_LAYER2_CUH