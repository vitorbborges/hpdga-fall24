#ifndef HNSW_KNN_SEARCH_CUH
#define HNSW_KNN_SEARCH_CUH

#include <cuda_runtime.h>
#include "search_layer.cuh"

#include <memory>
#include <vector>
#include <string>

#include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvToolsExtCudaRt.h>


template <typename T = float>
void copy_constant_values(
    const Dataset<T>& queries,
    T*& d_queries,
    const int& start_node_id,
    d_Neighbor<T>* d_start_ids,
    T* d_dataset,
    const Dataset<float>& dataset,
    const size_t& ds_size,
    int* d_ef,
    const int& ef,
    int* d_layer_ef,
    const size_t& one
) {

    nvtxRangeId_t constcopyId = nvtxRangeStartA("Constant copies");

    // copy queries to device
    std::vector<T> queries_host(queries.size() * VEC_DIM);
    for (size_t i = 0; i < queries.size(); i++) {
        std::copy(queries[i].data(), queries[i].data() + VEC_DIM, queries_host.data() + i * VEC_DIM);
    }
    cudaMemcpy(d_queries, queries_host.data(), queries.size() * VEC_DIM * sizeof(T), cudaMemcpyHostToDevice);

    // Copy start ids to device
    d_Neighbor<T> start_ids[queries.size()];
    for (size_t i = 0; i < queries.size(); i++) {
        start_ids[i] = {0.0f, start_node_id};
    }
    cudaMemcpy(d_start_ids, start_ids, queries.size() * sizeof(d_Neighbor<T>), cudaMemcpyHostToDevice);

    // Copy dataset to device
    std::vector<T> dataset_host(ds_size * VEC_DIM);
    for (size_t i = 0; i < ds_size; i++) {
        std::copy(dataset[i].data(), dataset[i].data() + VEC_DIM, dataset_host.data() + i * VEC_DIM);
    }
    cudaMemcpy(d_dataset, dataset_host.data(), ds_size * VEC_DIM * sizeof(T), cudaMemcpyHostToDevice);

    // copy ef to device
    cudaMemcpy(d_ef, &ef, sizeof(int), cudaMemcpyHostToDevice);

    // copy layer ef to device
    cudaMemcpy(d_layer_ef, &one, sizeof(int), cudaMemcpyHostToDevice);

    nvtxRangeEnd(constcopyId);
    
}

void fill_adjacency_list(
    std::vector<int>& adjacency_host,
    const std::vector<Layer>& layers,
    int l_c
) {
    for (const Node& node : layers[l_c]) {
        int offset = node.data.id() * K;
        for (size_t i = 0; i < node.neighbors.size(); i++) {
            adjacency_host[offset + i] = node.neighbors[i].id;
        }
    }
}

void copy_layer_values(
    const std::vector<Layer>& layers,
    const int& l_c,
    const size_t& ds_size,
    int* d_adjacency_list,
    cudaStream_t stream
) {
    std::string layercopyMessage = "Layer copies at layer " + std::to_string(l_c);
    nvtxRangeId_t layercopyId = nvtxRangeStartA(layercopyMessage.c_str());

    // Allocate and initialize adjacency_host
    auto adjacency_host = std::make_unique<std::vector<int>>(ds_size * K, -1);

    // HostFunctionData struct
    struct HostFunctionData {
        std::vector<int>* adjacency_host;
        const std::vector<Layer>* layers;
        int l_c;
    };

    // Allocate HostFunctionData
    HostFunctionData* hostData = new HostFunctionData{adjacency_host.get(), &layers, l_c};

    // Define host function
    auto myHostFunction = [](void* userData) {
        auto* data = static_cast<HostFunctionData*>(userData);
        fill_adjacency_list(*(data->adjacency_host), *(data->layers), data->l_c);
        delete data; // Clean up
    };

    // Launch host function
    cudaLaunchHostFunc(stream, myHostFunction, hostData);

    // Synchronize to ensure the host function finishes
    cudaStreamSynchronize(stream);

    // Copy adjacency list to device after host computation completes
    cudaMemcpyAsync(
        d_adjacency_list,
        adjacency_host->data(),
        ds_size * K * sizeof(int),
        cudaMemcpyHostToDevice,
        stream
    );

    // Final stream synchronization to ensure the copy completes
    // cudaStreamSynchronize(stream);

    nvtxRangeEnd(layercopyId);
}


template <typename T = float>
void fetch_layer_results(
    d_Neighbor<T>* d_layer_result,
    d_Neighbor<T>* layer_result,
    const int& ef,
    const size_t& queries_size,
    cudaStream_t stream
) {
    // Fetch layer results concurrently
    cudaMemcpyAsync(layer_result, d_layer_result, ef * queries_size * sizeof(d_Neighbor<T>), cudaMemcpyDeviceToHost, stream);
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

__global__ void warmup_kernel() {}

void warmup(cudaStream_t stream) {
    warmup_kernel<<<1, 1, 0, stream>>>();
    cudaDeviceSynchronize(); // Ensure everything is initialized
}

template <typename T = float>
SearchResults knn_search(
    const Dataset<T>& queries,
    const int& start_node_id,
    const int& ef,
    const std::vector<Layer>& layers,
    const size_t& ds_size,
    const Dataset<float>& dataset,
    std::chrono::system_clock::time_point& q_start,
    std::chrono::system_clock::time_point& q_end
) {
    cudaStream_t warmupStream;
    cudaStreamCreate(&warmupStream);
    warmup(warmupStream);

    size_t shared_query_size = VEC_DIM * sizeof(T); // For shared_query
    size_t candidates_array_size = MAX_HEAP_SIZE * sizeof(d_Neighbor<T>); // For candidates_array
    size_t top_candidates_array_size = MAX_HEAP_SIZE * sizeof(d_Neighbor<T>); // For top_candidates_array
    size_t shared_visited_size = DATASET_SIZE * sizeof(bool); // For shared_visited
    size_t shared_distances_size = MAX_HEAP_SIZE * sizeof(T); // For shared_distances

    // Total shared memory size
    size_t sharedMemSize = shared_query_size +
                                   candidates_array_size +
                                   top_candidates_array_size +
                                   shared_visited_size +
                                   shared_distances_size;


    // CONSTANT ALLOCATIONS

    nvtxRangeId_t constallocId = nvtxRangeStartA("Constant allocations");

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

    nvtxRangeEnd(constallocId);

    //////////////////////////////////////////////////////////////////////////////

    // LAYER-SPECIFIC ALLOCATIONS

    nvtxRangeId_t layerallocId = nvtxRangeStartA("Layer-specific allocations");

    // Allocate initial search id of each query on device
    d_Neighbor<T>* d_start_ids;
    cudaMalloc(&d_start_ids, queries.size() * sizeof(d_Neighbor<T>));

    // Allocate fixed degree graph adjency list on device
    int* d_adjacency_list;
    cudaMalloc(&d_adjacency_list, ds_size * K * sizeof(int));

    // Allocate layer search result on device
    d_Neighbor<T> layer_result[queries.size()];
    d_Neighbor<T>* d_layer_result;
    cudaMalloc(&d_layer_result, queries.size() * sizeof(d_Neighbor<T>));

    // Allocate hnsw search result on device
    d_Neighbor<T> hnsw_result[ef * queries.size()];
    d_Neighbor<T>* d_hnsw_result;
    cudaMalloc(&d_hnsw_result, ef * queries.size() * sizeof(d_Neighbor<T>));

    nvtxRangeEnd(layerallocId);

    //////////////////////////////////////////////////////////////////////////////

    // Create streams
    cudaStream_t streams[layers.size() + 1];
    for (int i = 0; i <= layers.size(); i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Create end of copying events
    cudaEvent_t endcpy_events[layers.size() + 1];
    for (int i = 0; i <= layers.size(); i++) {
        cudaEventCreate(&endcpy_events[i]);
    }

    // Create end of fetching results events
    cudaEvent_t endfetch_events[layers.size() + 1];
    for (int i = 0; i <= layers.size(); i++) {
        cudaEventCreate(&endfetch_events[i]);
    }

    q_start = get_now();

    // COPY VALUES THAT WILL NOT CHANGE ON LAYER INTERATION
    copy_constant_values(
        queries,
        d_queries,
        start_node_id,
        d_start_ids,
        d_dataset,
        dataset,
        ds_size,
        d_ef,
        ef,
        d_layer_ef,
        one
    );

    cudaEventRecord(endcpy_events[0], streams[0]);
    cudaEventRecord(endfetch_events[0], streams[0]);

    
    for (int l_c = layers.size() - 1; l_c >= 0; l_c--) {

        int streamIdx = layers.size() - l_c;

        std::string streamName = "Layer " + std::to_string(l_c);
        nvtxNameCudaStreamA(streams[streamIdx], streamName.c_str());

        cudaStreamWaitEvent(streams[streamIdx - 1], endcpy_events[streamIdx - 1]);

        // Initialize current layer parameters
        d_Neighbor<T>* d_current_start_ids;
        int* d_current_ef;
        int current_ef;
        d_Neighbor<T>* d_current_result;
        d_Neighbor<T>* current_result;

        // Set current layer parameters
        if (l_c == layers.size() - 1) { // if on first layer
            d_current_start_ids = d_start_ids;
        } else { // use previous layer's result as start_ids
            d_current_start_ids = d_layer_result;
        }

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
            layers,
            l_c,
            ds_size,
            d_adjacency_list,
            streams[streamIdx]
        );

        cudaEventRecord(endcpy_events[streamIdx], streams[streamIdx]);

        cudaStreamWaitEvent(streams[streamIdx - 1], endfetch_events[streamIdx - 1]);

        std::string kernelMessage = "Kernel execution on layer " + std::to_string(l_c);
        nvtxRangeId_t kernelId = nvtxRangeStartA(kernelMessage.c_str());
        // Launch kernel
        search_layer_kernel<<<queries.size(), VEC_DIM, sharedMemSize, streams[streamIdx]>>>(
            d_queries,
            d_current_start_ids,
            d_adjacency_list,
            d_dataset,
            d_current_ef,
            d_current_result
        );
        nvtxRangeEnd(kernelId);

        // Fetch layer results
        fetch_layer_results(
            d_current_result,
            current_result,
            current_ef,
            queries.size(),
            streams[streamIdx]
        );
        cudaEventRecord(endfetch_events[streamIdx], streams[streamIdx]);
    }

    q_end = get_now();

    // Free memory
    cudaFree(d_start_ids);
    cudaFree(d_queries);
    cudaFree(d_adjacency_list);
    cudaFree(d_dataset);
    cudaFree(d_ef);
    cudaFree(d_layer_ef);
    cudaFree(d_layer_result);
    cudaFree(d_hnsw_result);
    for (int i = 0; i <= layers.size(); i++) {
        cudaStreamDestroy(streams[i]);
        cudaEventDestroy(endcpy_events[i]);
        cudaEventDestroy(endfetch_events[i]);
    }

    return prepare_output(hnsw_result, ef, queries.size());
}

#endif // HNSW_KNN_SEARCH_CUH