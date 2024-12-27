
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
    T* queries,
    int* start_ids,
    int* adjaecy_list,
    T* dataset,
    int* ef,
    d_Neighbor<T>* result
) {
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;

    T* query = queries + bidx * VEC_DIM;
    int start_node_id = start_ids[bidx];

    // Shared memory for query data
    static __shared__ T shared_query[VEC_DIM];
    if (tidx < VEC_DIM) {
        shared_query[tidx] = query[tidx];
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
        dataset + start_node_id * VEC_DIM,
        VEC_DIM
    );
    __syncthreads();

    if (tidx == 0) {
        q.insert({start_dist, start_node_id});
        topk.insert({start_dist, start_node_id});
    }
    __syncthreads();

    // Shared memory for visited nodes
    bool shared_visited[DATASET_SIZE];
    if (tidx == 0) {
        shared_visited[start_node_id] = true;
    }

    // Main loop
    while (q.get_size() > 0) {
        __syncthreads();

        // Get the current node information
        d_Neighbor<T> now = q.pop();
        int* neighbors_ids = adjaecy_list + now.id * K;

        if (tidx == 0) {
            // Mark current node as visited
            shared_visited[now.id] = true;
            // printf("Visiting node: %d\n", now.id);
        }

        int n_neighbors = 0;
        while (neighbors_ids[n_neighbors] != -1 && n_neighbors < K) {
            // count number of neighbors in current node
            n_neighbors++;
        }

        // Copy neighbor data to shared memory (test if this is faster)
        static __shared__ T shared_neighbor_data[K * VEC_DIM];
        for (size_t i = 0; i < n_neighbors; i++) {
            shared_neighbor_data[i * VEC_DIM + tidx] = dataset[neighbors_ids[i] * VEC_DIM + tidx];
        }
        __syncthreads();

        // Check the exit condidtion
        if (topk.top().dist < now.dist) {
            break;
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
            if (tidx == 0) {
                shared_distances[i] = dist;
            }
        }
        __syncthreads();
        
        // Update priority queues
        if (tidx == 0) {
            for (size_t i = 0; i < n_neighbors; i++) {
                if (shared_visited[neighbors_ids[i]]) continue;
                shared_visited[neighbors_ids[i]] = true;

                if (shared_distances[i < topk.top().dist] ||
                    topk.get_size() < *ef) {
                    q.insert({shared_distances[i], neighbors_ids[i]});
                    topk.insert({shared_distances[i], neighbors_ids[i]});

                    if (topk.get_size() > *ef) {
                        topk.pop();
                    }

                }
            }
            // printf("Queue size: %d\n", q.get_size());
            // q.print_heap();
            // printf("\n");
            // printf("Topk size: %d\n", topk.get_size());
            // topk.print_heap();
            // printf("=====================================\n");
            // printf("\n");
        }
        __syncthreads();
    }

    // Flush the heap and get results
    if (tidx == 0) {
        while (topk.get_size() > *ef) {
            topk.pop();
        }
        for (size_t i = 0; i < *ef; i++) {
            result[blockIdx.x * (*ef) + i] = topk.pop();
        }
    }
}

template <typename T = float>
SearchResults search_layer_launch(
    const Dataset<T>& queries,
    const int& start_node_id,
    const int& ef,
    const std::vector<Layer>& layers,
    const int& l_c,
    const size_t& ds_size,
    const Dataset<T>& dataset
) {
    // Allocate query on device
    T* d_queries;
    cudaMalloc(&d_queries, queries.size() * VEC_DIM * sizeof(T));
    for (size_t i = 0; i < queries.size(); i++) {
        cudaMemcpy(d_queries + i * VEC_DIM, queries[i].data(), VEC_DIM * sizeof(T), cudaMemcpyHostToDevice);
    }

    // Allocate initial search id of each query on device
    int* d_start_ids;
    cudaMalloc(&d_start_ids, queries.size() * sizeof(int));
    int support_array[queries.size()]; // create support array to ease memory copy
    std::fill(support_array, support_array + queries.size(), start_node_id);
    cudaMemcpy(d_start_ids, support_array, queries.size() * sizeof(int), cudaMemcpyHostToDevice);

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
    d_Neighbor<T> result[ef * queries.size()];
    d_Neighbor<T>* d_result;
    cudaMalloc(&d_result, ef * queries.size() * sizeof(d_Neighbor<T>));

    // Launch kernel
    search_layer_kernel<<<queries.size(), VEC_DIM>>>(
        d_queries,
        d_start_ids,
        d_adjaency_list,
        d_dataset,
        d_ef,
        d_result
    );

    // copy result back to host
    cudaMemcpy(result, d_result, ef * queries.size() * sizeof(d_Neighbor<T>), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_start_ids);
    cudaFree(d_queries);
    cudaFree(d_adjaency_list);
    cudaFree(d_dataset);
    cudaFree(d_ef);
    cudaFree(d_result);

    // Prepare output
    SearchResults search_results(queries.size());
    for (size_t i = 0; i < queries.size(); i++) {
        SearchResult search_result = SearchResult();
        for (size_t j = 0; j < ef; j++) {
            search_result.result.emplace_back(result[i * ef + j].dist, result[i * ef + j].id);
        }
        std::sort(search_result.result.begin(), search_result.result.end(), CompLess());
        search_results[i] = search_result;
    }

    return search_results;
}

#endif // HNSW_SEARCH_LAYER_CUH
