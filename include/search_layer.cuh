#ifndef HNSW_SEARCH_LAYER_CUH
#define HNSW_SEARCH_LAYER_CUH

#include "bloom_filter.cuh"
#include "data_structures.cuh"
#include "euclidean_distance.cuh"
#include "priority_queue.cuh"
#include "search_layer_kernels.cuh"
#include "symmetric_mmh.cuh"
#include "utils.cuh"
#include <cuda_runtime.h>

using namespace ds;
using namespace utils;

// Common function to launch search layer kernels

template <typename T = float>
SearchResults
search_layer_launch(const Dataset<T> &queries, const int &start_node_id,
                    const int &ef, const Layer &layer,
                    const std::vector<int> &layer_map, const size_t &ds_size,
                    const Dataset<T> &dataset, const std::string &kernel_name) {
  size_t num_queries = queries.size();
  SearchResults search_results(num_queries);
  search_results.params.k = ef;
  search_results.params.n_query = num_queries;
  search_results.params.experiment_name = kernel_name;

  //////////////////////////////////////////////////////////////////////////////

  // MEMORY ALLOCATIONS

  search_results.params.start_alloc = get_now();

  // Allocate query on device
  T *d_queries;
  cudaMalloc(&d_queries, num_queries * VEC_DIM * sizeof(T));

  // Allocate dataset on device
  T *d_dataset;
  cudaMalloc(&d_dataset, ds_size * VEC_DIM * sizeof(T));

  // Allocate Target output lenght
  int *d_ef;
  cudaMalloc(&d_ef, sizeof(int));

  // Allocate initial search id of each query on device
  d_Neighbor<T> *d_start_ids;
  cudaMalloc(&d_start_ids, num_queries * sizeof(d_Neighbor<T>));

  // Allocate fixed degree graph adjency list on device
  int *d_adjacency_list;
  cudaMalloc(&d_adjacency_list, ds_size * K * sizeof(int));
  std::vector<int> adjacency_host(ds_size * K, -1);

  // Allocate layer search result on device
  d_Neighbor<T> *d_result;
  cudaMalloc(&d_result, ef * num_queries * sizeof(d_Neighbor<T>));

  //////////////////////////////////////////////////////////////////////////////

  // MEMORY ALLOCATIONS

  search_results.params.start_calc = get_now();

  // copy queries to device
  std::vector<T> queries_host(queries.size() * VEC_DIM);
  for (size_t i = 0; i < queries.size(); i++) {
    std::copy(queries[i].data(), queries[i].data() + VEC_DIM,
              queries_host.data() + i * VEC_DIM);
  }
  cudaMemcpy(d_queries, queries_host.data(),
             queries.size() * VEC_DIM * sizeof(T), cudaMemcpyHostToDevice);

  // Copy dataset to device
  std::vector<T> dataset_host(ds_size * VEC_DIM);
  for (size_t i = 0; i < ds_size; i++) {
    std::copy(dataset[i].data(), dataset[i].data() + VEC_DIM,
              dataset_host.data() + i * VEC_DIM);
  }
  cudaMemcpy(d_dataset, dataset_host.data(), ds_size * VEC_DIM * sizeof(T),
             cudaMemcpyHostToDevice);

  // copy ef to device
  cudaMemcpy(d_ef, &ef, sizeof(int), cudaMemcpyHostToDevice);

  // Copy start ids to device
  d_Neighbor<T> start_ids[queries.size()];
  for (size_t i = 0; i < queries.size(); i++) {
    start_ids[i] = {0.0f, start_node_id};
  }
  cudaMemcpy(d_start_ids, start_ids, queries.size() * sizeof(d_Neighbor<T>),
             cudaMemcpyHostToDevice);

  // Copy adjacency list to device
  for (const int &node_id : layer_map) {
    const Node &node = layer[node_id];
    for (size_t i = 0; i < node.neighbors.size(); i++) {
      adjacency_host[node.data.id() * K + i] = node.neighbors[i].id;
    }
  }
  cudaMemcpy(d_adjacency_list, adjacency_host.data(), ds_size * K * sizeof(int),
             cudaMemcpyHostToDevice);

  // Launch kernel
  if (kernel_name == "gpu_non_opt") {
    search_layer_non_opt_kernel<<<num_queries, VEC_DIM>>>(
        d_queries, d_start_ids, d_adjacency_list, d_dataset, d_ef, d_result);
  } else if (kernel_name == "gpu_shared_mem") {
    search_layer_shared_mem_kernel<<<num_queries, VEC_DIM>>>(
        d_queries, d_start_ids, d_adjacency_list, d_dataset, d_ef, d_result);
  } else if (kernel_name == "gpu_eucdist_opt") {
    search_layer_eucdist_opt_kernel<<<num_queries, VEC_DIM>>>(
        d_queries, d_start_ids, d_adjacency_list, d_dataset, d_ef, d_result);
  } else if (kernel_name == "gpu_bloom_pq") {
    search_layer_bloom_pq_kernel<<<num_queries, VEC_DIM>>>(
        d_queries, d_start_ids, d_adjacency_list, d_dataset, d_ef, d_result);
  } else {
    std::cerr << "Invalid kernel name" << std::endl;
    exit(1);
  }

  // Copy results back to host
  d_Neighbor<T> result_host[ef * num_queries];
  cudaMemcpy(result_host, d_result, ef * num_queries * sizeof(d_Neighbor<T>),
             cudaMemcpyDeviceToHost);

  search_results.params.end_calc = get_now();

  // Free unified and device memory
  cudaFree(d_queries);
  cudaFree(d_dataset);
  cudaFree(d_ef);
  cudaFree(d_start_ids);
  cudaFree(d_adjacency_list);
  cudaFree(d_result);

  search_results.params.end_alloc = get_now();

  // Prepare output
  for (size_t i = 0; i < num_queries; i++) {
    SearchResult search_result;
    for (size_t j = 0; j < ef; j++) {
      search_result.result.emplace_back(result_host[i * ef + j].dist,
                                        result_host[i * ef + j].id);
    }
    std::sort(search_result.result.begin(), search_result.result.end(),
              CompLess());
    search_results[i] = search_result;
  }

  return search_results;
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
template <typename T> size_t computeSharedMemoryBytesHost(int ef) {
  size_t offset = 0;

  // 1) shared_query
  offset = alignUp(offset, alignof(T));
  offset += VEC_DIM * sizeof(T);

  // 2) visited (Bloom filter)
  offset = alignUp(offset, alignof(SharedBloomFilter));
  offset += sizeof(SharedBloomFilter);

  // 3) candidates_array
  offset = alignUp(offset, alignof(d_Neighbor<T>));
  offset += (ef + 2) * sizeof(d_Neighbor<T>);

  // 4) top_candidates_array
  offset = alignUp(offset, alignof(d_Neighbor<T>));
  offset += (ef + 2) * sizeof(d_Neighbor<T>);

  // 5) now
  offset = alignUp(offset, alignof(d_Neighbor<T>));
  offset += sizeof(d_Neighbor<T>);

  // 6) computation_mem_area (example: 64 floats or T’s)
  offset = alignUp(offset, alignof(T));
  offset += 32 * sizeof(T);

  // 7) exit_flag
  offset = alignUp(offset, alignof(int));
  offset += sizeof(int);

  // 8) shared_distances
  offset = alignUp(offset, alignof(T));
  offset += K * sizeof(T);

  return offset;
}

template <typename T = float>
SearchResults search_layer_bloom_smmh_launch(const Dataset<T> &queries,
                                             const int &start_node_id,
                                             const int &ef, const Layer &layer,
                                             const std::vector<int> &layer_map,
                                             const size_t &ds_size,
                                             const Dataset<T> &dataset) {
  size_t num_queries = queries.size();
  SearchResults search_results(num_queries);
  search_results.params.k = ef;
  search_results.params.n_query = num_queries;

  //////////////////////////////////////////////////////////////////////////////

  search_results.params.start_alloc = get_now();

  // MEMORY ALLOCATIONS

  // Allocate query on device
  T *d_queries;
  cudaMalloc(&d_queries, num_queries * VEC_DIM * sizeof(T));

  // Allocate dataset on device
  T *d_dataset;
  cudaMalloc(&d_dataset, ds_size * VEC_DIM * sizeof(T));

  // Allocate Target output lenght
  int *d_ef;
  cudaMalloc(&d_ef, sizeof(int));

  // Allocate initial search id of each query on device
  d_Neighbor<T> *d_start_ids;
  cudaMalloc(&d_start_ids, num_queries * sizeof(d_Neighbor<T>));

  // Allocate fixed degree graph adjency list on device
  int *d_adjacency_list;
  cudaMalloc(&d_adjacency_list, ds_size * K * sizeof(int));
  std::vector<int> adjacency_host(ds_size * K, -1);

  // Allocate layer search result on device
  d_Neighbor<T> *d_result;
  cudaMalloc(&d_result, ef * num_queries * sizeof(d_Neighbor<T>));

  //////////////////////////////////////////////////////////////////////////////

  // MEMORY ALLOCATIONS

  search_results.params.start_calc = get_now();

  // copy queries to device
  std::vector<T> queries_host(queries.size() * VEC_DIM);
  for (size_t i = 0; i < queries.size(); i++) {
    std::copy(queries[i].data(), queries[i].data() + VEC_DIM,
              queries_host.data() + i * VEC_DIM);
  }
  cudaMemcpy(d_queries, queries_host.data(),
             queries.size() * VEC_DIM * sizeof(T), cudaMemcpyHostToDevice);

  // Copy dataset to device
  std::vector<T> dataset_host(ds_size * VEC_DIM);
  for (size_t i = 0; i < ds_size; i++) {
    std::copy(dataset[i].data(), dataset[i].data() + VEC_DIM,
              dataset_host.data() + i * VEC_DIM);
  }
  cudaMemcpy(d_dataset, dataset_host.data(), ds_size * VEC_DIM * sizeof(T),
             cudaMemcpyHostToDevice);

  // copy ef to device
  cudaMemcpy(d_ef, &ef, sizeof(int), cudaMemcpyHostToDevice);

  // Copy start ids to device
  d_Neighbor<T> start_ids[queries.size()];
  for (size_t i = 0; i < queries.size(); i++) {
    start_ids[i] = {0.0f, start_node_id};
  }
  cudaMemcpy(d_start_ids, start_ids, queries.size() * sizeof(d_Neighbor<T>),
             cudaMemcpyHostToDevice);

  // Copy adjacency list to device
  for (const int &node_id : layer_map) {
    const Node &node = layer[node_id];
    for (size_t i = 0; i < node.neighbors.size(); i++) {
      adjacency_host[node.data.id() * K + i] = node.neighbors[i].id;
    }
  }
  cudaMemcpy(d_adjacency_list, adjacency_host.data(), ds_size * K * sizeof(int),
             cudaMemcpyHostToDevice);

  size_t smemBytes = computeSharedMemoryBytesHost<T>(ef);

  // Launch kernel shared memory size 48KB
  search_layer_bloom_smmh_kernel<<<num_queries, VEC_DIM, smemBytes>>>(
      d_queries, d_start_ids, d_adjacency_list, d_dataset, d_ef, d_result);

  d_Neighbor<T> result_host[ef * num_queries];
  cudaMemcpy(result_host, d_result, ef * num_queries * sizeof(d_Neighbor<T>),
             cudaMemcpyDeviceToHost);

  search_results.params.end_calc = get_now();

  // Prepare output
  for (size_t i = 0; i < num_queries; i++) {
    SearchResult search_result;
    for (size_t j = 0; j < ef; j++) {
      search_result.result.emplace_back(result_host[i * ef + j].dist,
                                        result_host[i * ef + j].id);
    }
    std::sort(search_result.result.begin(), search_result.result.end(),
              CompLess());
    search_results[i] = search_result;
  }

  // Free unified and device memory
  cudaFree(d_queries);
  cudaFree(d_dataset);
  cudaFree(d_ef);
  cudaFree(d_start_ids);
  cudaFree(d_adjacency_list);
  cudaFree(d_result);

  search_results.params.end_alloc = get_now();

  return search_results;
}

#endif // HNSW_SEARCH_LAYER_CUH
