#ifndef HNSW_SEARCH_LAYER_CUH
#define HNSW_SEARCH_LAYER_CUH

#include <cuda_runtime.h>
#include "priority_queue.cuh"
#include "euclidean_distance.cuh"
#include "data_structures.cuh"
#include "device_data_structures.cuh"
#include "utils.cuh"

#define EUCDIST_THREADS 32

template <typename T = float>
__global__ void search_layer_kernel(
    T* query_data,
    int* start_node_id,
    int* ef,
    bool* visited,
    int* vec_dim,
    size_t* layer_len,
    d_Node<T>* layer_data
) {

    __shared__ d_Neighbor<T> candidates_array[MAX_HEAP_SIZE];
    __shared__ int candidates_size;
    __shared__ d_Neighbor<T> top_candidates_array[MAX_HEAP_SIZE];
    __shared__ int top_candidates_size;
    // TODO: solve warning #20054-D: dynamic initialization is not supported for a function-scope 
    // static __shared__ variable within a __device__/__global__ function

    __syncthreads();

    PriorityQueue<T> q(candidates_array, &candidates_size, MIN_HEAP);
    PriorityQueue<T> topk(top_candidates_array, &top_candidates_size, MAX_HEAP);
    
    T start_dist = euclidean_distance(
        query_data,
        layer_data[*start_node_id].data.x,
        *vec_dim
    );
    // TODO: use optimized euclidean distance kernel

    q.insert({start_dist, *start_node_id});

    while (q.get_size() > 0) {
        d_Neighbor<T> now = q.pop();
        printf("now: %d\n", now.id);

        if (topk.top().dist < now.dist) {
            break;
        } else {
            topk.insert(now);
        }

        d_Node<T> now_node = layer_data[now.id];

        d_Neighbor<T>* now_neighbors = now_node.neighbors;
        // TODO: optimize removing visited neighbors from this array.

        process_neighbors<<<1, now_node.n_neighbors>>>(
            query_data,
            layer_data,
            now_neighbors,
            vec_dim
        );

        for (size_t i = 0; i < now_node.n_neighbors; i++) {
            if (!visited[now_neighbors[i].id]) {
                visited[now_neighbors[i].id] = true;
                q.insert(now_neighbors[i]);
            }
        }
        q.print_heap();
    }
}

template <typename T = float>
__global__ void process_neighbors(
    const T* query,
    const d_Node<T>* layer_data,
    d_Neighbor<T>* neighbors,
    const int* vec_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    T dist = euclidean_distance(
        query,
        layer_data[neighbors[idx].id].data.x,
        *vec_dim
    );
    // TODO: use optimized euclidean distance kernel

    __syncthreads();

    neighbors[idx].dist = dist;
}

template <typename T = float>
auto search_layer_launch(
    const Data<T>& query,
    const int& start_node_id,
    const int& ef,
    Layer& layer,
    const size_t& ds_size
) {
    auto results = SearchResult();

    int vec_dim = query.size();
    size_t layer_len = layer.size();

    d_Node<T>* layer_data = new d_Node<T>[layer_len];
    for (size_t i = 0; i < layer_len; i++) {
        T* data = layer[i].data.data();
        int num_neighbors = layer[i].neighbors.size();
        d_Neighbor<T>* neighbors = nullptr;

        if (num_neighbors > 0) {
            neighbors = new d_Neighbor<T>[num_neighbors];
            for (size_t j = 0; j < num_neighbors; j++) {
                neighbors[j] = d_Neighbor<T>(layer[i].neighbors[j].dist, layer[i].neighbors[j].id);
            }
        }

        layer_data[i] = d_Node<T>(data, layer[i].data.id(), neighbors, num_neighbors);
    }

    bool visited[ds_size] = {false};
    visited[start_node_id] = true;

    d_Neighbor<T> h_results[ef]; // TODO: check if here is ef or k
    
    // Create device pointers
    T* d_query_data;
    int* d_start_node_id;
    int* d_ef;
    bool* d_visited;
    int* d_vec_dim;
    size_t* d_layer_len;
    d_Node<T>* d_layer_data;
    d_Neighbor<T>* d_results;


    // Allocate memory on the device
    cudaMalloc(&d_query_data, vec_dim * sizeof(T));
    cudaMalloc(&d_start_node_id, sizeof(int));
    cudaMalloc(&d_ef, sizeof(int));
    cudaMalloc(&d_visited, ds_size * sizeof(bool));
    cudaMalloc(&d_vec_dim, sizeof(int));
    cudaMalloc(&d_layer_len, sizeof(size_t));
    cudaMalloc(&d_layer_data, layer_len * sizeof(d_Node<T>));

    // Copy data to the device
    cudaMemcpy(d_query_data, query.data(), vec_dim * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_start_node_id, &start_node_id, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ef, &ef, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, visited, ds_size * sizeof(bool), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vec_dim, &vec_dim, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_len, &layer_len, sizeof(size_t), cudaMemcpyHostToDevice);

    for (size_t i = 0; i < layer_len; ++i) {
        // Allocate memory for the data array (x) on the device
        T* d_data_array;
        cudaMalloc(&d_data_array, vec_dim * sizeof(T)); 
        cudaMemcpy(d_data_array, layer[i].data.data(), vec_dim * sizeof(T), cudaMemcpyHostToDevice);
        
        // Update the host-side layer_data with the device pointer
        layer_data[i].set_data(d_data_array);
    
        // Allocate memory for the neighbors array on the device
        int neighbors_size = layer[i].neighbors.size();
        d_Neighbor<T>* d_neighbors_array = nullptr;
        if (neighbors_size > 0) {
            cudaMalloc(&d_neighbors_array, neighbors_size * sizeof(d_Neighbor<T>));
            
            // Copy each neighbor's data to the device
            cudaMemcpy(d_neighbors_array, layer[i].neighbors.data(), 
                       neighbors_size * sizeof(d_Neighbor<T>), cudaMemcpyHostToDevice);
        }
    
        // Update the host-side layer_data with the device neighbors pointer
        layer_data[i].neighbors = d_neighbors_array;
        layer_data[i].n_neighbors = neighbors_size;
    }
    
    // Copy the host-side layer_data array to the device memory (d_layer_data)
    cudaMemcpy(d_layer_data, layer_data, layer_len * sizeof(d_Node<T>), cudaMemcpyHostToDevice);
    

    // Launch kernel
    search_layer_kernel<<<1, 1>>>(
        d_query_data,
        d_start_node_id,
        d_ef,
        d_visited,
        d_vec_dim,
        d_layer_len,
        d_layer_data
    );
    // catch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in search_layer_kernel: " << cudaGetErrorString(error) << std::endl;
    }

    // Copy results back to host
    // cudaMemcpy(visited, d_visited, ds_size * sizeof(bool), cudaMemcpyDeviceToHost);
    
    // for (int i = 0; i < ef; i++) {
    //     results.result.emplace_back(top_candidates[i].dist, top_candidates[i].id);
    // }

    // Free memory
    cudaFree(d_query_data);
    cudaFree(d_start_node_id);
    cudaFree(d_ef);
    cudaFree(d_visited);
    cudaFree(d_vec_dim);
    cudaFree(d_layer_len);
    cudaFree(d_layer_data);
    delete[] layer_data;

    return results;
}

#endif // HNSW_SEARCH_LAYER_CUH