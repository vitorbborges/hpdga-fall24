#ifndef HNSW_SEARCH_LAYER_CUH
#define HNSW_SEARCH_LAYER_CUH

#include <cuda_runtime.h>
#include "priority_queue.cuh"
#include "euclidean_distance.cuh"
#include "data_structures.cuh"
#include "utils.cuh"
#include "bloom_filter.cuh"


using namespace ds;

#define K 32
#define VEC_DIM 128
#define DATASET_SIZE 10000
#define maxk 100

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
    __shared__ T shared_query[VEC_DIM];
    __shared__ SharedBloomFilter bloom_filter;

    if (tidx < VEC_DIM) {
        shared_query[tidx] = query[tidx];
    }
    if (tidx == 0) {
        bloom_filter.init();
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

    // Calculate distance from start node to query and add to queue
    T start_dist = euclidean_distance_gpu<T>(
        shared_query,
        dataset + start_node_id * VEC_DIM,
        VEC_DIM
    );

    if (tidx == 0) {
        q.insert({start_dist, start_node_id});
        topk.insert({start_dist, start_node_id});
        bloom_filter.set(start_node_id); // Mark the start node as visited
    }
    __syncthreads();

    // Main loop
    while (q.get_size() > 0) {
        __syncthreads();

        // Get the current node information
        d_Neighbor<T> now = q.pop();
        int* neighbors_ids = adjacency_list + now.id * K;

        int n_neighbors = 0;
        while (neighbors_ids[n_neighbors] != -1 && n_neighbors < K) {
            n_neighbors++;
        }

        // Check the exit condition
        if (topk.top().dist < now.dist) {
            break;
        }
        __syncthreads();

        // Shared memory for distances
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
                int neighbor_id = neighbors_ids[i];
                if (bloom_filter.test(neighbor_id)) continue; // Check if visited
                bloom_filter.set(neighbor_id); // Mark as visited

                if (shared_distances[i] < topk.top().dist ||
                    topk.get_size() < *ef) {
                    
                    q.insert({shared_distances[i], neighbor_id});

                    // Bound the size of candidates PQ by maxk
                    if (q.get_size() > maxk) {
                        q.pop(); // Remove the element with the lowest priority
                    }


                    topk.insert({shared_distances[i], neighbor_id});
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


#endif // HNSW_SEARCH_LAYER_CUH