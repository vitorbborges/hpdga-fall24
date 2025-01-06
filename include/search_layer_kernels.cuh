#ifndef HNSW_SEARCH_LAYER_KERNELS_CUH
#define HNSW_SEARCH_LAYER_KERNELS_CUH

#include "data_structures.cuh"
#include "priority_queue.cuh"
#include "symmetric_mmh.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////////////////

// Non-optimized baseline gpu kernel

template <typename T = float>
__global__ void search_layer_non_opt_kernel(T *queries,
                                            d_Neighbor<T> *start_ids,
                                            int *adjacency_list, T *dataset,
                                            int *ef, d_Neighbor<T> *result) {
  int tidx = threadIdx.x;
  int bidx = blockIdx.x;

  T *query = queries + bidx * VEC_DIM;
  int start_node_id = start_ids[bidx].id;

  __syncthreads();

  // Priority queues initialization
  d_Neighbor<T> candidates_array[MAX_HEAP_SIZE];
  int candidates_size;
  candidates_size = 0;
  PriorityQueue<T> q(candidates_array, &candidates_size, MIN_HEAP);

  d_Neighbor<T> top_candidates_array[MAX_HEAP_SIZE];
  int top_candidates_size;
  top_candidates_size = 0;
  PriorityQueue<T> topk(top_candidates_array, &top_candidates_size, MAX_HEAP);

  // Calculate distance from start node to query and add to queue
  T start_dist =
      euclidean_distance_non_opt<T>(query, dataset + start_node_id * VEC_DIM, VEC_DIM);

  if (tidx == 0) {
    q.insert({start_dist, start_node_id});
    topk.insert({start_dist, start_node_id});
  }
  __syncthreads();

  // Shared memory for visited nodes
  bool visited[DATASET_SIZE];
  if (tidx == 0) {
    visited[start_node_id] = true;
  }

  // Main loop
  while (q.get_size() > 0) {
    __syncthreads();

    // Get the current node information
    d_Neighbor<T> now = q.pop();
    int *neighbors_ids = adjacency_list + now.id * K;

    if (tidx == 0) {
      // Mark current node as visited
      visited[now.id] = true;
      // printf("Visiting node: %d\n", now.id);
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
    T distances[K];
    for (size_t i = 0; i < n_neighbors; i++) {
      T dist = euclidean_distance_non_opt<T>(
          query, dataset + neighbors_ids[i] * VEC_DIM, VEC_DIM);
      __syncthreads();
      if (tidx == 0) {
        distances[i] = dist;
      }
    }
    __syncthreads();

    // Update priority queues
    if (tidx == 0) {
      for (size_t i = 0; i < n_neighbors; i++) {
        if (visited[neighbors_ids[i]])
          continue;
        visited[neighbors_ids[i]] = true;

        if (distances[i < topk.top().dist] || topk.get_size() < *ef) {

          q.insert({distances[i], neighbors_ids[i]});
          topk.insert({distances[i], neighbors_ids[i]});

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

///////////////////////////////////////////////////////////////////////////////////////////////////////////

// Shared memory optimization

template <typename T = float>
__global__ void search_layer_shared_mem_kernel(T *queries,
                                               d_Neighbor<T> *start_ids,
                                               int *adjacency_list, T *dataset,
                                               int *ef, d_Neighbor<T> *result) {
  int tidx = threadIdx.x;
  int bidx = blockIdx.x;

  T *query = queries + bidx * VEC_DIM;
  int start_node_id = start_ids[bidx].id;

  // Shared memory for query data
  __shared__ T shared_query[VEC_DIM];

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

  // Calculate distance from start node to query and add to queue
  T start_dist = euclidean_distance_non_opt<T>(
      shared_query, dataset + start_node_id * VEC_DIM, VEC_DIM);

  if (tidx == 0) {
    q.insert({start_dist, start_node_id});
    topk.insert({start_dist, start_node_id});
  }
  __syncthreads();

  // Shared memory for visited nodes
  bool visited[DATASET_SIZE];
  if (tidx == 0) {
    visited[start_node_id] = true;
  }

  // Main loop
  while (q.get_size() > 0) {
    __syncthreads();

    // Get the current node information
    d_Neighbor<T> now = q.pop();
    int *neighbors_ids = adjacency_list + now.id * K;

    if (tidx == 0) {
      // Mark current node as visited
      visited[now.id] = true;
      // printf("Visiting node: %d\n", now.id);
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
      T dist = euclidean_distance_non_opt<T>(
          shared_query, dataset + neighbors_ids[i] * VEC_DIM, VEC_DIM);
      __syncthreads();
      if (tidx == 0) {
        shared_distances[i] = dist;
      }
    }
    __syncthreads();

    // Update priority queues
    if (tidx == 0) {
      for (size_t i = 0; i < n_neighbors; i++) {
        if (visited[neighbors_ids[i]])
          continue;
        visited[neighbors_ids[i]] = true;

        if (shared_distances[i < topk.top().dist] || topk.get_size() < *ef) {

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

///////////////////////////////////////////////////////////////////////////////////////////////////////////

// Euclidean distance optimization using warp reduction

template <typename T = float>
__global__ void
search_layer_eucdist_opt_kernel(T *queries, d_Neighbor<T> *start_ids,
                                int *adjacency_list, T *dataset, int *ef,
                                d_Neighbor<T> *result) {
  int tidx = threadIdx.x;
  int bidx = blockIdx.x;

  T *query = queries + bidx * VEC_DIM;
  int start_node_id = start_ids[bidx].id;

  // Shared memory for query data
  __shared__ T shared_query[VEC_DIM];

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

  // Calculate distance from start node to query and add to queue
  T start_dist = euclidean_distance_gpu<T>(
      shared_query, dataset + start_node_id * VEC_DIM, VEC_DIM);

  if (tidx == 0) {
    q.insert({start_dist, start_node_id});
    topk.insert({start_dist, start_node_id});
  }
  __syncthreads();

  // Shared memory for visited nodes
  bool visited[DATASET_SIZE];
  if (tidx == 0) {
    visited[start_node_id] = true;
  }

  // Main loop
  while (q.get_size() > 0) {
    __syncthreads();

    // Get the current node information
    d_Neighbor<T> now = q.pop();
    int *neighbors_ids = adjacency_list + now.id * K;

    if (tidx == 0) {
      // Mark current node as visited
      visited[now.id] = true;
      // printf("Visiting node: %d\n", now.id);
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
          shared_query, dataset + neighbors_ids[i] * VEC_DIM, VEC_DIM);
      __syncthreads();
      if (tidx == 0) {
        shared_distances[i] = dist;
      }
    }
    __syncthreads();

    // Update priority queues
    if (tidx == 0) {
      for (size_t i = 0; i < n_neighbors; i++) {
        if (visited[neighbors_ids[i]])
          continue;
        visited[neighbors_ids[i]] = true;

        if (shared_distances[i < topk.top().dist] || topk.get_size() < *ef) {

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

///////////////////////////////////////////////////////////////////////////////////////////////////////////

// Bloom filter and Priority Queue optimization

template <typename T = float>
__global__ void search_layer_bloom_pq_kernel(T *queries,
                                             d_Neighbor<T> *start_ids,
                                             int *adjacency_list, T *dataset,
                                             int *ef, d_Neighbor<T> *result) {
  int tidx = threadIdx.x;
  int bidx = blockIdx.x;

  T *query = queries + bidx * VEC_DIM;
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
      shared_query, dataset + start_node_id * VEC_DIM, VEC_DIM);

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
    int *neighbors_ids = adjacency_list + now.id * K;

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
          shared_query, dataset + neighbors_ids[i] * VEC_DIM, VEC_DIM);
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
        if (bloom_filter.test(neighbor_id))
          continue;                    // Check if visited
        bloom_filter.set(neighbor_id); // Mark as visited

        if (shared_distances[i] < topk.top().dist || topk.get_size() < *ef) {

          q.insert({shared_distances[i], neighbor_id});
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

///////////////////////////////////////////////////////////////////////////////////////////////////////////

// Bloom filter and Symmetric Min-Max Heap optimization

template <typename T = float>
__global__ void search_layer_bloom_smmh_kernel(T *queries,
                                               d_Neighbor<T> *start_ids,
                                               int *adjacency_list, T *dataset,
                                               int *ef, d_Neighbor<T> *result) {
  int tidx = threadIdx.x;
  int bidx = blockIdx.x;

  T *query = queries + bidx * VEC_DIM;
  int start_node_id = start_ids[bidx].id;

  extern __shared__ unsigned char sharedMemory[];

  T *shared_query = (T *)sharedMemory;
  for (size_t i = 0; i < VEC_DIM; i++) {
    shared_query[i] = query[i];
  }

  SharedBloomFilter *visited = (SharedBloomFilter *)&shared_query[VEC_DIM];
  visited->init();

  d_Neighbor<T> *candidates_array = (d_Neighbor<T> *)&visited[1];
  SymmetricMinMaxHeap<T> q(candidates_array, MIN_HEAP);

  d_Neighbor<T> *top_candidates_array =
      (d_Neighbor<T> *)&candidates_array[*ef + 2];
  SymmetricMinMaxHeap<T> topk(top_candidates_array, MAX_HEAP);

  // calculate distance from start node to query and add to queue
  T *computation_mem_area = (T *)&top_candidates_array[*ef + 2];
  T start_dist = euclidean_distance_dynamic_shared_gpu<T>(
      shared_query, dataset + start_node_id * VEC_DIM, VEC_DIM,
      computation_mem_area);

  if (tidx == 0) {
    q.insert({start_dist, start_node_id});
    topk.insert({start_dist, start_node_id});
    visited->set(start_node_id);
  }
  __syncthreads();

  // Main loop
  int *exit = (int *)&computation_mem_area[32];
  while (*exit == 0) {
    __syncthreads();

    if (tidx == 0 && q.isEmpty())
      *exit = 1;
    __syncthreads();
    if (*exit == 1)
      break;

    // Get the current node information
    d_Neighbor<T> *now = (d_Neighbor<T> *)&exit[1];
    if (tidx == 0) {
      *now = q.pop();
    }
    __syncthreads();

    int *neighbors_ids = adjacency_list + now->id * K;

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
    if (*exit == 1)
      break;

    // distance computation
    T *shared_distances = (T *)&now[1];
    __syncthreads();

    for (size_t i = 0; i < n_neighbors; i++) {
      __syncthreads();
      T dist = euclidean_distance_dynamic_shared_gpu<T>(
          shared_query, dataset + neighbors_ids[i] * VEC_DIM, VEC_DIM,
          computation_mem_area);
      __syncthreads();
      if (tidx == 0) {
        shared_distances[i] = dist;
      }
    }
    __syncthreads();

    if (tidx == 0) {
      for (size_t i = 0; i < n_neighbors; i++) {
        int neighbor_id = neighbors_ids[i];
        if (visited->test(neighbor_id))
          continue;
        visited->set(neighbor_id);

        if (shared_distances[i] < topk.top().dist) {
          q.insert({shared_distances[i], neighbor_id});
          if (q.getSize() > *ef) {
            q.popBottom();
          }
          topk.insert({shared_distances[i], neighbor_id});
          if (topk.getSize() > *ef) {
            topk.pop();
          }
        }
      }
    }
  }

  // get results
  if (tidx == 0) {
    for (size_t i = 0; i < *ef; i++) {
      result[blockIdx.x * (*ef) + i] = topk.popMin();
    }
  }
}

#endif // HNSW_SEARCH_LAYER_KERNELS_CUH
