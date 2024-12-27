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
    // __syncthreads();

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
    const int num_queries = queries.size();

    // Allocate pinned memory for queries and results
    T* h_queries_pinned;
    int* h_start_ids_pinned;
    d_Neighbor<T>* h_result_pinned;

    cudaHostAlloc(&h_queries_pinned, num_queries * VEC_DIM * sizeof(T), cudaHostAllocDefault);
    cudaHostAlloc(&h_start_ids_pinned, num_queries * sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc(&h_result_pinned, num_queries * ef * sizeof(d_Neighbor<T>), cudaHostAllocDefault);

    // Copy query data into pinned memory
    for (size_t i = 0; i < num_queries; i++) {
        std::copy(queries[i].data(), queries[i].data() + VEC_DIM, h_queries_pinned + i * VEC_DIM);
        h_start_ids_pinned[i] = start_node_id;
    }

    // Allocate device memory
    T* d_queries;
    int* d_start_ids;
    int* d_adjacency_list;
    T* d_dataset;
    int* d_ef;
    d_Neighbor<T>* d_result;

    cudaMalloc(&d_queries, num_queries * VEC_DIM * sizeof(T));
    cudaMalloc(&d_start_ids, num_queries * sizeof(int));
    cudaMalloc(&d_adjacency_list, ds_size * K * sizeof(int));
    cudaMalloc(&d_dataset, ds_size * VEC_DIM * sizeof(T));
    cudaMalloc(&d_ef, sizeof(int));
    cudaMalloc(&d_result, num_queries * ef * sizeof(d_Neighbor<T>));

    // Copy adjacency list to device
    std::vector<int> adjacency_host(ds_size * K, -1);
    for (Node node : layers[l_c]) {
        int offset = node.data.id() * K;
        for (size_t i = 0; i < node.neighbors.size(); i++) {
            adjacency_host[offset + i] = node.neighbors[i].id;
        }
    }
    cudaMemcpy(d_adjacency_list, adjacency_host.data(), ds_size * K * sizeof(int), cudaMemcpyHostToDevice);


    // Copy dataset to device
    for (size_t i = 0; i < ds_size; i++) {
        cudaMemcpy(d_dataset + i * VEC_DIM, dataset[i].data(), VEC_DIM * sizeof(T), cudaMemcpyHostToDevice);
    }

    // Copy ef parameter to device
    cudaMemcpy(d_ef, &ef, sizeof(int), cudaMemcpyHostToDevice);

    // Copy queries and start IDs to device
    cudaMemcpy(d_queries, h_queries_pinned, num_queries * VEC_DIM * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_ids, h_start_ids_pinned, num_queries * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    search_layer_kernel<<<num_queries, VEC_DIM>>>(
        d_queries,
        d_start_ids,
        d_adjacency_list,
        d_dataset,
        d_ef,
        d_result
    );

    // Copy results back to pinned memory
    cudaMemcpy(h_result_pinned, d_result, num_queries * ef * sizeof(d_Neighbor<T>), cudaMemcpyDeviceToHost);

    // Prepare output
    SearchResults search_results(num_queries);
    for (size_t i = 0; i < num_queries; i++) {
        SearchResult search_result;
        for (size_t j = 0; j < ef; j++) {
            search_result.result.emplace_back(h_result_pinned[i * ef + j].dist, h_result_pinned[i * ef + j].id);
        }
        std::sort(search_result.result.begin(), search_result.result.end(), CompLess());
        search_results[i] = search_result;
    }

    // Free pinned and device memory
    cudaFreeHost(h_queries_pinned);
    cudaFreeHost(h_start_ids_pinned);
    cudaFreeHost(h_result_pinned);
    cudaFree(d_queries);
    cudaFree(d_start_ids);
    cudaFree(d_adjacency_list);
    cudaFree(d_dataset);
    cudaFree(d_ef);
    cudaFree(d_result);

    return search_results;
}


#endif // HNSW_SEARCH_LAYER_CUH