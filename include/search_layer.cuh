#ifndef HNSW_SEARCH_LAYER_CUH
#define HNSW_SEARCH_LAYER_CUH

#include <cuda_runtime.h>
#include "priority_queue.cuh"

#include "euclidean_distance.cuh"
#include "data_structures.cuh"
#include "utils.cuh"

using namespace ds;

template <typename T = float>
__global__ void search_layer_kernel(
    T* queries,
    d_Neighbor<T>* start_ids,
    int* adjacency_list,
    T* dataset,
    int* ef,
    d_Neighbor<T>* result
) {
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;

    T* query = queries + bidx * VEC_DIM;
    int start_node_id = start_ids[bidx].id;


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
        int* neighbors_ids = adjacency_list + now.id * K;

        if (tidx == 0) {
            // Mark current node as visited
            shared_visited[now.id] = true;
        }

        int n_neighbors = 0;
        while (neighbors_ids[n_neighbors] != -1 && n_neighbors < K) {
            // count number of neighbors in current node
            n_neighbors++;
        }

        

        // Check the exit condidtion
        bool c = topk.top().dist < now.dist;
        if (c) {
            break;
        }
        __syncthreads();

        // distance computation (convert to bulk?)
        static __shared__ T shared_distances[K];
        for (size_t i = 0; i < n_neighbors; i++) {
            T dist = euclidean_distance_gpu<T>(
                shared_query,
                dataset + neighbors_ids[i] * VEC_DIM,
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
    const Layer& layer,
    const std::vector<int>& layer_map,
    const size_t& ds_size,
    const Dataset<T>& dataset,
    std::chrono::system_clock::time_point& start,
    std::chrono::system_clock::time_point& end
) {
    size_t num_queries = queries.size();

    //////////////////////////////////////////////////////////////////////////////

    // MEMORY ALLOCATIONS

    // Allocate query on device
    T* d_queries;
    cudaMalloc(&d_queries, num_queries * VEC_DIM * sizeof(T));

    // Allocate dataset on device
    T* d_dataset;
    cudaMalloc(&d_dataset, ds_size * VEC_DIM * sizeof(T));

    // Allocate Target output lenght
    int* d_ef;
    cudaMalloc(&d_ef, sizeof(int));

    // Allocate initial search id of each query on device
    d_Neighbor<T>* d_start_ids;
    cudaMalloc(&d_start_ids, num_queries * sizeof(d_Neighbor<T>));

    // Allocate fixed degree graph adjency list on device
    int* d_adjacency_list;
    cudaMalloc(&d_adjacency_list, ds_size * K * sizeof(int));
    std::vector<int> adjacency_host(ds_size * K, -1);

    // Allocate layer search result on device
    d_Neighbor<T>* d_result;
    cudaMalloc(&d_result, ef * num_queries * sizeof(d_Neighbor<T>));

    //////////////////////////////////////////////////////////////////////////////

    // MEMORY ALLOCATIONS

    start = get_now();

    // copy queries to device
    std::vector<T> queries_host(queries.size() * VEC_DIM);
    for (size_t i = 0; i < queries.size(); i++) {
        std::copy(queries[i].data(), queries[i].data() + VEC_DIM, queries_host.data() + i * VEC_DIM);
    }
    cudaMemcpy(d_queries, queries_host.data(), queries.size() * VEC_DIM * sizeof(T), cudaMemcpyHostToDevice);

    // Copy dataset to device
    std::vector<T> dataset_host(ds_size * VEC_DIM);
    for (size_t i = 0; i < ds_size; i++) {
        std::copy(dataset[i].data(), dataset[i].data() + VEC_DIM, dataset_host.data() + i * VEC_DIM);
    }
    cudaMemcpy(d_dataset, dataset_host.data(), ds_size * VEC_DIM * sizeof(T), cudaMemcpyHostToDevice);

    // copy ef to device
    cudaMemcpy(d_ef, &ef, sizeof(int), cudaMemcpyHostToDevice);

    // Copy start ids to device
    d_Neighbor<T> start_ids[queries.size()];
    for (size_t i = 0; i < queries.size(); i++) {
        start_ids[i] = {0.0f, start_node_id};
    }
    cudaMemcpy(d_start_ids, start_ids, queries.size() * sizeof(d_Neighbor<T>), cudaMemcpyHostToDevice);

    // Copy adjacency list to device
    for (const int& node_id : layer_map) {
        const Node& node = layer[node_id];
        for (size_t i = 0; i < node.neighbors.size(); i++) {
            adjacency_host[node.data.id() * K + i] = node.neighbors[i].id;
        }
    }
    cudaMemcpy(d_adjacency_list, adjacency_host.data(), ds_size * K * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    search_layer_kernel<<<num_queries, VEC_DIM>>>(
        d_queries,
        d_start_ids,
        d_adjacency_list,
        d_dataset,
        d_ef,
        d_result
    );

    d_Neighbor<T> result_host[ef * num_queries];
    cudaMemcpy(result_host, d_result, ef * num_queries * sizeof(d_Neighbor<T>), cudaMemcpyDeviceToHost);

    end = get_now();

    // Prepare output
    SearchResults search_results(num_queries);
    for (size_t i = 0; i < num_queries; i++) {
        SearchResult search_result;
        for (size_t j = 0; j < ef; j++) {
            search_result.result.emplace_back(result_host[i * ef + j].dist, result_host[i * ef + j].id);
        }
        std::sort(search_result.result.begin(), search_result.result.end(), CompLess());
        search_results[i] = search_result;
    }

    // Free unified and device memory
    cudaFree(d_queries);
    cudaFree(d_dataset);
    cudaFree(d_ef);
    cudaFree(d_start_ids);
    cudaFree(d_adjacency_list);
    cudaFree(d_result);

    return search_results;
}

#endif // HNSW_SEARCH_LAYER_CUH