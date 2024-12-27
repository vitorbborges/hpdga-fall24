#ifndef HNSW_KNN_SEARCH_CUH
#define HNSW_KNN_SEARCH_CUH

#include <cuda_runtime.h>
#include "search_layer.cuh"

#define CHECK_CUDA_CALL(call)                                                                \
    do {                                                                                     \
        cudaError_t err = call;                                                              \
        if (err != cudaSuccess) {                                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": "             \
                      << cudaGetErrorString(err) << std::endl;                               \
            exit(EXIT_FAILURE);                                                              \
        }                                                                                    \
    } while (0)

template <typename T = float>
void copy_constant_values(
    const Dataset<T>& queries,
    T*& d_queries,
    T* d_dataset,
    const Dataset<float>& dataset,
    const size_t& ds_size,
    int* d_ef,
    const int& ef,
    int* d_layer_ef,
    const size_t& one
) {

    // copy queries to device
    for (size_t i = 0; i < queries.size(); i++) {
        cudaMemcpy(d_queries + i * VEC_DIM, queries[i].data(), VEC_DIM * sizeof(T), cudaMemcpyHostToDevice);
    }

    // copy dataset to device
    for (size_t i = 0; i < ds_size; i++) {
        cudaMemcpy(d_dataset + i * VEC_DIM, dataset[i].data(), VEC_DIM * sizeof(T), cudaMemcpyHostToDevice);
    }

    // copy ef to device
    cudaMemcpy(d_ef, &ef, sizeof(int), cudaMemcpyHostToDevice);

    // copy layer ef to device
    cudaMemcpy(d_layer_ef, &one, sizeof(int), cudaMemcpyHostToDevice);
    
}

void copy_layer_values(
    const int* start_ids,
    const size_t& queries_size,
    int* d_start_ids,
    const std::vector<Layer>& layers,
    const int& l_c,
    int* d_adjaency_list
) {

    // copy start ids array to device
    cudaMemcpy(d_start_ids, start_ids, queries_size * sizeof(int), cudaMemcpyHostToDevice);

    // copy fixed degree graph adjency list to device
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
}

template <typename T = float>
SearchResults prepare_output(
    d_Neighbor<T>* hnsw_result,
    const int& ef,
    const size_t& queries_size
) {
    SearchResults search_results(queries_size);
    for (size_t i = 0; i < queries_size; i++) {
        SearchResult search_result = SearchResult();
        for (size_t j = 0; j < ef; j++) {
            search_result.result.emplace_back(hnsw_result[i * ef + j].dist, hnsw_result[i * ef + j].id);
        }
        std::sort(search_result.result.begin(), search_result.result.end(), CompLess());
        search_results[i] = search_result;
    }

    return search_results;
}

template <typename T = float>
SearchResults knn_search(
    const Dataset<T>& queries,
    const int& start_node_id,
    const int& ef,
    const std::vector<Layer>& layers,
    const size_t& ds_size,
    const Dataset<float>& dataset
) {
    // Allocate query on device
    T* d_queries;
    cudaMalloc(&d_queries, queries.size() * VEC_DIM * sizeof(T));   

    // Allocate dataset on device
    T* d_dataset;
    cudaMalloc(&d_dataset, ds_size * VEC_DIM * sizeof(T));

    // Allocate Target output lenght
    int* d_ef;
    cudaMalloc(&d_ef, sizeof(int));

    // Allocate search layer target lenght  
    int* d_layer_ef;
    cudaMalloc(&d_layer_ef, sizeof(int));
    int one = 1;

    // COPY VALUES THAT WILL NOT CHANGE ON LAYER INTERATION
    copy_constant_values(
        queries,
        d_queries,
        d_dataset,
        dataset,
        ds_size,
        d_ef,
        ef,
        d_layer_ef,
        one
    );

    // Allocate initial search id of each query on device
    int* d_start_ids;
    cudaMalloc(&d_start_ids, queries.size() * sizeof(int));

    // Initialize start ids array to start_node_id
    int start_ids[queries.size()];
    std::fill(start_ids, start_ids + queries.size(), start_node_id);

    // Allocate fixed degree graph adjency list on device
    int* d_adjaency_list;
    cudaMalloc(&d_adjaency_list, ds_size * K * sizeof(int));

    // Allocate layer search result on device
    d_Neighbor<T> layer_result[queries.size()];
    d_Neighbor<T>* d_layer_result;
    cudaMalloc(&d_layer_result, queries.size() * sizeof(d_Neighbor<T>));

    // Allocate hnsw search result on device
    d_Neighbor<T> hnsw_result[ef * queries.size()];
    d_Neighbor<T>* d_hnsw_result;
    cudaMalloc(&d_hnsw_result, ef * queries.size() * sizeof(d_Neighbor<T>));
    
    for (int l_c = layers.size() - 1; l_c >= 0; l_c--) {

        // Initialize current layer parameters
        int* d_current_ef;
        int current_ef;
        d_Neighbor<T>* d_current_result;
        d_Neighbor<T>* current_result;

        // Set current layer parameters
        if (l_c > 0) { // if on upper layers
            d_current_ef = d_layer_ef;
            current_ef = one;
            d_current_result = d_layer_result;
            current_result = layer_result;
        } else { // if on base layer
            d_current_ef = d_ef;
            current_ef = ef;
            d_current_result = d_hnsw_result;
            current_result = hnsw_result;
        }
        
        // copy layer-specific data to device memory
        copy_layer_values(
            start_ids,
            queries.size(),
            d_start_ids,
            layers,
            l_c,
            d_adjaency_list
        );
        
        // Launch kernel
        search_layer_kernel<<<queries.size(), VEC_DIM>>>(
            d_queries,
            d_start_ids,
            d_adjaency_list,
            d_dataset,
            d_current_ef,
            d_current_result
        );

        // Fetch layer results
        cudaMemcpy(current_result, d_current_result, current_ef * queries.size() * sizeof(d_Neighbor<T>), cudaMemcpyDeviceToHost);

        if (l_c == 0) break;

        // Update start_ids for next layer
        for (size_t i = 0; i < current_ef * queries.size(); i++) {
            start_ids[i] = current_result[i].id;
        }
    }

    // Free memory
    cudaFree(d_start_ids);
    cudaFree(d_queries);
    cudaFree(d_adjaency_list);
    cudaFree(d_dataset);
    cudaFree(d_ef);
    cudaFree(d_layer_ef);
    cudaFree(d_layer_result);
    cudaFree(d_hnsw_result);

    return prepare_output(hnsw_result, ef, queries.size());
}

#endif // HNSW_KNN_SEARCH_CUH