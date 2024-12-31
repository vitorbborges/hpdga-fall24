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
    d_Neighbor<T>* start_ids,
    int* adjaecy_list,
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
        int* neighbors_ids = adjaecy_list + now.id * K;

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

#endif // HNSW_SEARCH_LAYER_CUH