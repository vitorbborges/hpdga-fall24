#ifndef HNSW_SEARCH_LAYER_CUH
#define HNSW_SEARCH_LAYER_CUH

#include <cuda_runtime.h>
#include "priority_queue.cuh"
#include "euclidean_distance.cuh"
#include "data_structures.cuh"
#include "device_data_structures.cuh"
#include "utils.cuh"

#define EUCDIST_THREADS 32

__global__ void euclidean_distance_simple(
    const float* vec1,
    const float* vec2,
    float* distance_block,
    int num_dims
) {
    static __shared__ float shared[EUCDIST_THREADS];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float diff = vec1[idx] - vec2[idx];
    shared[threadIdx.x] = diff * diff;

    __syncthreads();

    if (threadIdx.x == 0) {
        float block_sum = 0;
        for (int i = 0; i < blockDim.x; i++) {
            block_sum += shared[i];
        }
        distance_block[blockIdx.x + 1] = block_sum;
    }
    
}

template <typename T = float>
__global__ void search_layer_kernel(
    T* query_data,
    int* start_node_id,
    int* ef,
    bool* visited,
    int* vec_dim,
    size_t* layer_len,
    d_Node<T>* layer_data,
    d_Neighbor<T>* candidates_array,
    d_Neighbor<T>* top_candidates_array
) {
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;

    PriorityQueue<T> candidates_pq(candidates_array);
    PriorityQueue<T> top_candidates_pq(top_candidates_array);

    size_t numBlocks = (*vec_dim + EUCDIST_THREADS - 1) / EUCDIST_THREADS;
    T* dist_block = (T*)malloc((numBlocks + 1) * sizeof(T));
    memset(dist_block, 0, (numBlocks + 1) * sizeof(T));
    // TODO: for some reason it is impossible to write in the first position of the array, this emplains the +1

    euclidean_distance_simple<<<numBlocks, EUCDIST_THREADS>>>(
        query_data,
        layer_data[*start_node_id].data.x,
        dist_block,
        *vec_dim
    );
    // TODO: use optimized euclidean distance kernel

    T dist_from_en = 0;
    for (size_t i = 0; i < numBlocks + 1; i++) {
        printf("Dist block[%d]: %f\n", i, dist_block[i]);
        dist_from_en += dist_block[i];
    }
    printf("Dist from entry node: %f\n", dist_from_en);
    dist_from_en = sqrt(dist_from_en);
    printf("Dist from entry node: %f\n", dist_from_en);

    d_Neighbor<T> start_node(dist_from_en, *start_node_id);
    candidates_pq.insert(start_node);
    top_candidates_pq.insert(start_node);

    top_candidates_pq.print_heap();

    d_Neighbor<T> example(0.0, 0);
    top_candidates_pq.insert(example);
    top_candidates_pq.print_heap();

    d_Neighbor<T> example2(0.5, 1);
    top_candidates_pq.insert(example2);
    top_candidates_pq.print_heap();

    d_Neighbor<T> example3(0.3, 2);
    top_candidates_pq.insert(example3);
    top_candidates_pq.print_heap();    

    top_candidates_pq.pop_min();
    top_candidates_pq.print_heap();

    top_candidates_pq.pop_max();
    top_candidates_pq.print_heap();

    int count = 0;
    while (candidates_pq.get_size() > 0) {
        // printf("Iteration: %d :", count++);
        // top_candidates_pq.print_heap();
        d_Neighbor<T> nearest_candidate = candidates_pq.top();
        d_Node<T> nearest_candidate_node = layer_data[nearest_candidate.id];
        candidates_pq.pop_min();

        if (nearest_candidate.dist > top_candidates_pq.top().dist) break;

        int n_neighbors = nearest_candidate_node.n_neighbors;
        if (n_neighbors > 0) {
            d_Neighbor<T>* nearest_candidate_neighbors = nearest_candidate_node.neighbors;
            process_neighbors<<<1, n_neighbors>>>(
                query_data,
                layer_data,
                nearest_candidate_neighbors,
                vec_dim
            );

            for (size_t i = 0; i < n_neighbors; i++) {
                if (visited[nearest_candidate_neighbors[i].id]) continue;
                if (nearest_candidate_neighbors[i].dist < top_candidates_pq.top().dist ||
                    top_candidates_pq.get_size() < *ef) {
                    d_Neighbor<T> new_candidate(nearest_candidate_neighbors[i].dist, nearest_candidate_neighbors[i].id);
                    candidates_pq.insert(new_candidate);
                    top_candidates_pq.insert(new_candidate);

                    if (top_candidates_pq.get_size() > *ef) {
                        top_candidates_pq.pop_min();
                    }
                }
            }
        }

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

    const size_t numBlocks = (*vec_dim + EUCDIST_THREADS - 1) / EUCDIST_THREADS;
    T* dist_block = (T*)malloc((numBlocks + 1) * sizeof(T));
    memset(dist_block, 0, (numBlocks + 1) * sizeof(T));

    euclidean_distance_simple<<<numBlocks, EUCDIST_THREADS>>>(
        query,
        layer_data[neighbors[idx].id].data.x,
        dist_block,
        *vec_dim
    );

    T dist = 0;
    for (size_t i = 0; i < numBlocks + 1; i++) {
        dist += dist_block[i];
    }

    neighbors[idx].dist = sqrtf(dist);
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

    d_Neighbor<T> top_candidates[ef]; // TODO: check if here is ef or k
    
    // Create device pointers
    T* d_query_data;
    int* d_start_node_id;
    int* d_ef;
    bool* d_visited;
    int* d_vec_dim;
    size_t* d_layer_len;
    d_Node<T>* d_layer_data;
    d_Neighbor<T>* d_candidates;
    d_Neighbor<T>* d_top_candidates;

    // Allocate memory on the device
    cudaMalloc(&d_query_data, vec_dim * sizeof(T));
    cudaMalloc(&d_start_node_id, sizeof(int));
    cudaMalloc(&d_ef, sizeof(int));
    cudaMalloc(&d_visited, ds_size * sizeof(bool));
    cudaMalloc(&d_vec_dim, sizeof(int));
    cudaMalloc(&d_layer_len, sizeof(size_t));
    cudaMalloc(&d_layer_data, layer_len * sizeof(d_Node<T>));
    cudaMalloc(&d_candidates, ef * sizeof(d_Neighbor<T>)); // TODO: check if here is ef or k
    cudaMalloc(&d_top_candidates, ef * sizeof(d_Neighbor<T>)); // TODO: check if here is ef or k


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
        layer_data[i].data.x = d_data_array;
    
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
        d_layer_data,
        d_candidates,
        d_top_candidates
    );
    // catch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error in search_layer_kernel: " << cudaGetErrorString(error) << std::endl;
    }

    // Copy results back to host
    // cudaMemcpy(visited, d_visited, ds_size * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaMemcpy(top_candidates, d_top_candidates, ef * sizeof(d_Neighbor<T>), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < ef; i++) {
        results.result.emplace_back(top_candidates[i].dist, top_candidates[i].id);
    }

    // Free memory
    cudaFree(d_query_data);
    cudaFree(d_start_node_id);
    cudaFree(d_ef);
    cudaFree(d_visited);
    cudaFree(d_vec_dim);
    cudaFree(d_layer_len);
    cudaFree(d_layer_data);
    delete[] layer_data;

    std::vector<int> top_candidates_ids;
    for (int i = 0; i < ef; i++) {
        top_candidates_ids.push_back(top_candidates[i].id);
    }


    return top_candidates_ids;
}

#endif // HNSW_SEARCH_LAYER_CUH