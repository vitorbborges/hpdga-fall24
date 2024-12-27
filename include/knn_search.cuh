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
void constant_mallocs(
    T*& d_queries,
    const Dataset<T>& queries,
    int*& d_ef,
    const int& ef,
    T*& d_dataset,
    const Dataset<float>& dataset,
    const size_t& ds_size
) {
    // Allocate query on device
    size_t query_size = queries.size() * VEC_DIM * sizeof(T);
    CHECK_CUDA_CALL(cudaMalloc(&d_queries, query_size));
    for (size_t i = 0; i < queries.size(); i++) {
        CHECK_CUDA_CALL(cudaMemcpy(d_queries + i * VEC_DIM, queries[i].data(), VEC_DIM * sizeof(T), cudaMemcpyHostToDevice));
    }

    // Allocate Target output length
    CHECK_CUDA_CALL(cudaMalloc(&d_ef, sizeof(int)));
    CHECK_CUDA_CALL(cudaMemcpy(d_ef, &ef, sizeof(int), cudaMemcpyHostToDevice));

    // Allocate dataset on device
    size_t dataset_size = ds_size * VEC_DIM * sizeof(T);
    CHECK_CUDA_CALL(cudaMalloc(&d_dataset, dataset_size));
    for (size_t i = 0; i < ds_size; i++) {
        CHECK_CUDA_CALL(cudaMemcpy(d_dataset + i * VEC_DIM, dataset[i].data(), VEC_DIM * sizeof(T), cudaMemcpyHostToDevice));
    }
}

void copy_layer_values(
    int* d_start_ids,
    const int* start_ids,
    int* d_adjaency_list,
    const Layer& layer,
    const int& ef,
    int* d_ef,
    const size_t& ds_size,
    size_t queries_size
) {
    // Copy start_ids to device
    CHECK_CUDA_CALL(cudaMemcpy(d_start_ids, start_ids, queries_size * sizeof(int), cudaMemcpyHostToDevice));

    // Copy adjency list to device
    for (Node node: layer) {
        int neighbors_ids[K];
        std::fill(neighbors_ids, neighbors_ids + K, -1);
        int node_neighbors_size = node.neighbors.size();
        if (node_neighbors_size > 0) {
            for (size_t i = 0; i < node_neighbors_size; i++) {
                neighbors_ids[i] = node.neighbors[i].id;
            }
            CHECK_CUDA_CALL(cudaMemcpy(d_adjaency_list + node.data.id() * K, neighbors_ids, K * sizeof(int), cudaMemcpyHostToDevice));
        }
    }

    // Copy ef to device
    CHECK_CUDA_CALL(cudaMemcpy(d_ef, &ef, sizeof(int), cudaMemcpyHostToDevice));
}

template <typename T = float>
SearchResults fetch_results(
    d_Neighbor<T>* d_result,
    d_Neighbor<T>* h_results,
    const int& ef,
    const size_t& queries_size
) {
    // Copy results from device to host
    CHECK_CUDA_CALL(cudaMemcpy(h_results, d_result, ef * queries_size * sizeof(d_Neighbor<T>), cudaMemcpyDeviceToHost));

    // Prepare output
    SearchResults search_results(queries_size);
    for (size_t i = 0; i < queries_size; ++i) {
        SearchResult search_result;
        for (size_t j = 0; j < ef; ++j) {
            const auto& neighbor = h_results[i * ef + j];
            search_result.add_neighbor(neighbor.dist, neighbor.id);
        }
        std::sort(search_result.result.begin(), search_result.result.end(), CompLess());
        search_results.push_back(std::move(search_result));
    }

    return search_results;
}

// Debug memory info
void log_gpu_memory() {
    size_t free_mem, total_mem;
    CHECK_CUDA_CALL(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU Memory - Free: " << free_mem / (1024.0 * 1024) << " MB, Total: " << total_mem / (1024.0 * 1024) << " MB" << std::endl;
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
    // Log initial GPU memory
    log_gpu_memory();

    // Create device pointers
    T* d_queries = nullptr;
    int* d_ef = nullptr;
    T* d_dataset = nullptr;

    // Constant allocations
    constant_mallocs(
        d_queries,
        queries,
        d_ef,
        ef,
        d_dataset,
        dataset,
        ds_size
    );

    // Allocate initial entry node id for each query
    int start_ids[queries.size()];
    std::fill(start_ids, start_ids + queries.size(), start_node_id);

    // Layer-specific device pointers
    int* d_start_ids = nullptr;
    int* d_adjaency_list = nullptr;
    int* d_search_ef = nullptr;
    d_Neighbor<T>* d_layer_results = nullptr;

    // Layer specific variables allocations
    CHECK_CUDA_CALL(cudaMalloc(&d_start_ids, queries.size() * sizeof(int)));
    CHECK_CUDA_CALL(cudaMalloc(&d_adjaency_list, ds_size * K * sizeof(int)));
    CHECK_CUDA_CALL(cudaMalloc(&d_search_ef, sizeof(int)));
    CHECK_CUDA_CALL(cudaMalloc(&d_layer_results, queries.size() * sizeof(d_Neighbor<T>)));

    for (int l_c = layers.size() - 1; l_c >= 0; l_c--) {

        // copy layer-specific data to device memory
        copy_layer_values(
            d_start_ids,
            start_ids,
            d_adjaency_list,
            layers[l_c],
            1,
            d_search_ef,
            ds_size,
            queries.size()
        );
        
        // Launch kernel
        search_layer_kernel<<<queries.size(), VEC_DIM>>>(
            d_queries,
            d_start_ids,
            d_adjaency_list,
            d_dataset,
            d_search_ef,
            d_layer_results
        );

        // Fetch layer results
        d_Neighbor<T> results_array[queries.size()];
        CHECK_CUDA_CALL(cudaMemcpy(results_array, d_layer_results, queries.size() * sizeof(d_Neighbor<T>), cudaMemcpyDeviceToHost));

        // Update start_ids for next layer
        for (size_t i = 0; i < queries.size(); i++) {
            start_ids[i] = results_array[i].id;
        }
    }

    // // Base layer search
    // std::cout << "Start searching base layer" << std::endl;

    // copy_layer_values(
    //     d_start_ids,
    //     start_ids,
    //     d_adjaency_list,
    //     layers[0],
    //     1,
    //     d_search_ef,
    //     ds_size,
    //     queries.size()
    // );

    // // Allocate result on device
    // d_Neighbor<T>* d_search_results = nullptr;
    // CHECK_CUDA_CALL(cudaMalloc(&d_search_results, queries.size() * sizeof(d_Neighbor<T>)));

    // // Launch kernel
    // search_layer_kernel<<<queries.size(), VEC_DIM>>>(
    //     d_queries,
    //     d_start_ids,
    //     d_adjaency_list,
    //     d_dataset,
    //     d_search_ef,
    //     d_search_results
    // );

    // std::cout << "Finished searching base layer" << std::endl;


    // // Fetch layer results
    // d_Neighbor<T> search_results_array[queries.size()];
    SearchResults search_results(queries.size());
    // CHECK_CUDA_CALL(cudaMemcpy(search_results_array, d_search_results, queries.size() * sizeof(d_Neighbor<T>), cudaMemcpyDeviceToHost));


    // Free constant memory
    CHECK_CUDA_CALL(cudaFree(d_queries));
    CHECK_CUDA_CALL(cudaFree(d_ef));
    CHECK_CUDA_CALL(cudaFree(d_dataset));
    CHECK_CUDA_CALL(cudaFree(d_start_ids));
    CHECK_CUDA_CALL(cudaFree(d_adjaency_list));
    // CHECK_CUDA_CALL(cudaFree(d_search_results));
    CHECK_CUDA_CALL(cudaFree(d_layer_results));

    return search_results;
}

#endif // HNSW_KNN_SEARCH_CUH