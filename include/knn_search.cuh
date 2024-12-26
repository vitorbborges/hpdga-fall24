#ifndef HNSW_KNN_SEARCH_CUH
#define HNSW_KNN_SEARCH_CUH

#include <cuda_runtime.h>
#include "search_layer.cuh"

template <typename T = float>
SearchResults knn_search(
    const Dataset<T>& queries,
    const int& start_node_id,
    const int& ef,
    const std::vector<Layer>& layers,
    const size_t& ds_size,
    const Dataset<float>& dataset
) {
    // Create device pointers
    T* d_queries;
    int* d_start_node_id;
    int* d_ef;
    T* d_dataset;


    // constant allocations occur independently of each layer
    constant_mallocs(
        d_queries,
        queries,
        d_ef,
        ef,
        d_dataset,
        dataset,
        ds_size
    );

    // Allocate initial id of entry node of each query on device
    int start_ids[queries.size()];
    std::fill(start_ids, start_ids + queries.size(), start_node_id);


    // iterate upper layers
    for (int l_c = layers.size() - 1; l_c >= 0; l_c--) {

        // create device pointers
        int* d_start_ids;
        int* d_adjaency_list;
        d_Neighbor<T>* d_results;

        // layer allocations accure for each layer
        layer_mallocs(
            d_start_ids,
            start_ids,
            d_adjaency_list,
            layers[l_c],
            ds_size,
            d_results,
            1,
            queries.size()
        )

        // Launch kernel
        search_layer_kernel<<<queries.size(), VEC_DIM>>>(
            d_queries,
            d_start_ids,
            d_adjaency_list,
            d_dataset,
            1,
            d_results
        );

        // Fetch results
        SearchResults search_results = fetch_results(
            d_results,
            1,
            queries.size()
        );

        // Update start_ids for next layer
        for (size_t i = 0; i < queries.size(); i++) {
            start_ids[i] = search_results[i].result[0].id;
        }

        // Free memory
        cudaFree(d_start_ids);
        cudaFree(d_adjaency_list);
        cudaFree(d_results);
    }

    // Search base layer

    // create device pointers
    int* d_adjaency_list;
    d_Neighbor<T>* d_results;

    // layer allocations accure for each layer
    layer_mallocs(
        d_adjaency_list,
        layers[0],
        ds_size,
        d_results,
        ef,
        queries.size()
    )

    // Launch kernel
    search_layer_kernel<<<queries.size(), VEC_DIM>>>(
        d_queries,
        d_start_node_id,
        d_adjaency_list,
        d_dataset,
        d_ef,
        d_results
    );

    // Fetch results
    SearchResults search_results = fetch_results(
        d_results,
        ef,
        queries.size()
    );

    // Free memory
    cudaFree(d_queries);
    cudaFree(d_start_node_id);
    cudaFree(d_ef);
    cudaFree(d_dataset);
    cudaFree(d_adjaency_list);
    cudaFree(d_results);

    return search_results;

}

template <typename T = float>
void constant_mallocs(
    T* d_queries,
    const Dataset<T>& queries,
    int* d_ef,
    const int& ef,
    T* d_dataset,
    const Dataset<float>& dataset,
    const size_t& ds_size,
) {
    // Allocate query on device
    cudaMalloc(&d_queries, queries.size() * VEC_DIM * sizeof(T));
    for (size_t i = 0; i < queries.size(); i++) {
        cudaMemcpy(d_queries + i * VEC_DIM, queries[i].data(), VEC_DIM * sizeof(T), cudaMemcpyHostToDevice);
    }

    // Allocate Target output lenght
    cudaMalloc(&d_ef, sizeof(int));
    cudaMemcpy(d_ef, &ef, sizeof(int), cudaMemcpyHostToDevice);

    // Allocate dataset on device
    cudaMalloc(&d_dataset, ds_size * VEC_DIM * sizeof(T));
    for (size_t i = 0; i < ds_size; i++) {
        cudaMemcpy(d_dataset + i * VEC_DIM, dataset[i].data(), VEC_DIM * sizeof(T), cudaMemcpyHostToDevice);
    }
}

template <typename T = float>
void layer_mallocs(
    int* d_start_ids,
    const int* start_ids,
    int* d_adjaency_list,
    const Layer& layer,
    const size_t& ds_size,
    d_Neighbor<T>* d_result,
    const int& ef,
    const size_t& queries_size
) {
    // Allocate start_ids on device
    cudaMalloc(&d_start_ids, queries_size * sizeof(int));
    cudaMemcpy(d_start_ids, start_ids, queries_size * sizeof(int), cudaMemcpyHostToDevice);

    // Allocate fixed degree graph adjency list on device
    int* d_adjaency_list;
    cudaMalloc(&d_adjaency_list, ds_size * K * sizeof(int));
    for (Node node: layer) {
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

    // Allocate results on device
    cudaMalloc(&d_result, ef * queries_size * sizeof(d_Neighbor<T>));

}

template <typename T = float>
SearchResults fetch_results(
    d_Neighbor<T>* d_result,
    const int& ef,
    const size_t& queries_size,
) {
    // Copy results from device to host
    d_Neighbor<T> result[ef * queries_size];
    cudaMemcpy(result, d_result, ef * queries_size * sizeof(d_Neighbor<T>), cudaMemcpyDeviceToHost);

    // Prepare output
    SearchResults search_results(queries_size);
    for (size_t i = 0; i < queries_size; i++) {
        SearchResult search_result = SearchResult();
        for (size_t j = 0; j < ef; j++) {
            search_result.result.emplace_back(result[i * ef + j].dist, result[i * ef + j].id);
        }
        std::sort(search_result.result.begin(), search_result.result.end(), CompLess());
        search_results.push_back(search_result);
    }

    return search_results;
}

#endif // HNSW_KNN_SEARCH_CUH