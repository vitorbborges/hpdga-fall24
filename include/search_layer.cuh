#ifndef HNSW_SEARCH_LAYER_CUH
#define HNSW_SEARCH_LAYER_CUH

#include <cuda_runtime.h>
#include "priority_queue.cuh"
#include "symmetric_mmh.cuh"
#include "bloom_filter.cuh"
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

    extern __shared__ unsigned char sharedMemory[];

    T* shared_query = (T*)sharedMemory;
    for (size_t i = 0; i < VEC_DIM; i++) {
        shared_query[i] = query[i];
    }

    SharedBloomFilter* visited = (SharedBloomFilter*)&shared_query[VEC_DIM];
    visited->init();

    d_Neighbor<T>* candidates_array = (d_Neighbor<T>*)&visited[1];
    SymmetricMinMaxHeap<T> q(candidates_array, MIN_HEAP, *ef);

    d_Neighbor<T>* top_candidates_array = (d_Neighbor<T>*)&candidates_array[*ef + 1];
    SymmetricMinMaxHeap<T> topk(top_candidates_array, MAX_HEAP, *ef);

    // calculate distance from start node to query and add to queue
    T* computation_mem_area = (T*)&top_candidates_array[*ef + 1];
    T start_dist = euclidean_distance_gpu<T>(
        shared_query,
        dataset + start_node_id * VEC_DIM,
        VEC_DIM,
        computation_mem_area
    );

    if (tidx == 0) {
        q.insert({start_dist, start_node_id});
        topk.insert({start_dist, start_node_id});
        visited->set(start_node_id);
    }
    __syncthreads();

    // Main loop
    int* exit = (int*)&computation_mem_area[32];
    while (*exit == 0) {
        __syncthreads();

        if (tidx == 0 && q.isEmpty()) *exit = 1;
        __syncthreads();
        if (*exit == 1) break;

        // Get the current node information
        d_Neighbor<T>* now = (d_Neighbor<T>*)&exit[1];
        if (tidx == 0) {
            *now = q.pop();
        }
        __syncthreads();

        int* neighbors_ids = adjacency_list + now->id * K;

        int n_neighbors = 0;
        while (neighbors_ids[n_neighbors] != -1 && n_neighbors <= K) {
            // count number of neighbors in current node
            n_neighbors++;
        }
        __syncthreads();

        // Check the exit condidtion
        if (tidx == 0) {
            if (topk.top().dist < now->dist) {
                *exit = 1;
            } else {
                *exit = 0;
            }
        }
        __syncthreads();
        if (*exit == 1) break;

        // distance computation
        T* shared_distances = (T*)&now[1];
        __syncthreads();

        for (size_t i = 0; i < n_neighbors; i++) {
            __syncthreads();
            T dist = euclidean_distance_gpu<T>(
                shared_query,
                dataset + neighbors_ids[i] * VEC_DIM,
                VEC_DIM,
                computation_mem_area
            );
            __syncthreads();
            if (tidx == 0) {
                shared_distances[i] = dist;
            }
        }
        __syncthreads();

        if (tidx == 0) {
            for (size_t i = 0; i < n_neighbors; i++) {
                int neighbor_id = neighbors_ids[i];
                printf("neighbor_id: %d\n", neighbor_id);
                if (visited->test(neighbor_id)) continue;
                visited->set(neighbor_id);

                q.insert({shared_distances[i], neighbor_id});
                
                topk.insert({shared_distances[i], neighbor_id});
                printf("q: ");
                q.print();
                printf("topk: ");
                topk.print();
            }
        }
    }

    // get results
    if (tidx == 0) {
        for (size_t i = 0; i < *ef; i++) {
            result[blockIdx.x * (*ef) + i] = topk.pop();
        }
    }

}

static inline size_t alignUp(size_t offset, size_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

/**
 * @brief Computes the number of bytes needed in dynamic shared memory
 *        for one block in search_layer_kernel, given ef and T.
 * 
 * The layout we assume (same as in the kernel example):
 *   1) shared_query: VEC_DIM * sizeof(T)
 *   2) visited:      1 * sizeof(SharedBloomFilter)
 *   3) candidates_array: (ef+1) * sizeof(d_Neighbor<T>)
 *   4) top_candidates_array: (ef+1) * sizeof(d_Neighbor<T>)
 *   5) computation_mem_area: e.g. 64 * sizeof(T)
 *   6) exit_flag:    1 * sizeof(int)
 *
 * @tparam T     data type of the vectors (float, half, etc.)
 * @param ef     search parameter (size of each heap)
 * @return size_t the total number of bytes required
 */
template <typename T>
size_t computeSharedMemoryBytesHost(int ef) 
{
    size_t offset = 0;
    
    // 1) shared_query
    offset = alignUp(offset, alignof(T));
    offset += VEC_DIM * sizeof(T);

    // 2) visited (Bloom filter)
    offset = alignUp(offset, alignof(SharedBloomFilter));
    offset += sizeof(SharedBloomFilter);

    // 3) candidates_array
    offset = alignUp(offset, alignof(d_Neighbor<T>));
    offset += (ef + 1) * sizeof(d_Neighbor<T>);

    // 4) top_candidates_array
    offset = alignUp(offset, alignof(d_Neighbor<T>));
    offset += (ef + 1) * sizeof(d_Neighbor<T>);

    // 5) now
    offset = alignUp(offset, alignof(d_Neighbor<T>));
    offset += sizeof(d_Neighbor<T>);

    // 6) computation_mem_area (example: 64 floats or T’s)
    offset = alignUp(offset, alignof(T));
    offset += 32 * sizeof(T);

    // 7) exit_flag
    offset = alignUp(offset, alignof(int));
    offset += sizeof(int);

    return offset;
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

    size_t smemBytes = computeSharedMemoryBytesHost<T>(ef);

    // Launch kernel shared memory size 48KB
    search_layer_kernel<<<num_queries, VEC_DIM, smemBytes>>>(
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