#ifndef HNSW_PROCESS_NEIGHBORS_CUH
#define HNSW_PROCESS_NEIGHBORS_CUH

#include <cuda_runtime.h>
#include <euclidean_distance.cuh>
#include <utils.cuh>

using namespace utils;

_global_ void process_neighbors(
    const Neighbor* neighbors, 
    int num_neighbors,
    const Node* layer, 
    const float* query,
    int* visited, 
    Neighbor* candidates, 
    int* candidate_count,
    Neighbor* top_candidates, 
    int* top_count, 
    float top_dist, 
    int ef
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_neighbors) return;

    const auto neighbor = neighbors[idx];

    if (atomicExch(&visited[neighbor.id], 1) == 1) return;

    const auto& neighbor_node = layer[neighbor.id];
    float dist_from_neighbor = euclidean_distance_cuda(query, neighbor_node.data, query.size());

    bool should_add = dist_from_neighbor < top_dist || *top_count < ef;
    if (should_add) {
        int position = atomicAdd(candidate_count, 1);
        candidates[position] = Neighbor(dist_from_neighbor, neighbor.id);

        position = atomicAdd(top_count, 1);
        if (position < ef) {
            top_candidates[position] = Neighbor(dist_from_neighbor, neighbor.id);
        } else {
            atomicSub(top_count, 1);
        }
    }
}

void launch_process_neighbors(
    const Neighbor* h_neighbors, 
    int num_neighbors,
    const Node* h_layer, 
    const float* h_query, 
    int query_size,
    int* h_visited, 
    Neighbor* h_candidates, 
    int* h_candidate_count,
    Neighbor* h_top_candidates, 
    int* h_top_count, 
    float top_dist, 
    int ef
) {
    Neighbor* d_neighbors;
    Node* d_layer;
    float* d_query;
    int* d_visited;
    Neighbor* d_candidates;
    Neighbor* d_top_candidates;
    int* d_candidate_count;
    int* d_top_count;

    cudaMalloc(&d_neighbors, num_neighbors * sizeof(Neighbor));
    cudaMalloc(&d_layer, num_neighbors * sizeof(Node));
    cudaMalloc(&d_query, query_size * sizeof(float));
    cudaMalloc(&d_visited, num_neighbors * sizeof(int));
    cudaMalloc(&d_candidates, ef * sizeof(Neighbor));
    cudaMalloc(&d_top_candidates, ef * sizeof(Neighbor));
    cudaMalloc(&d_candidate_count, sizeof(int));
    cudaMalloc(&d_top_count, sizeof(int));

    cudaMemcpy(d_neighbors, h_neighbors, num_neighbors * sizeof(Neighbor), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer, h_layer, num_neighbors * sizeof(Node), cudaMemcpyHostToDevice);
    cudaMemcpy(d_query, h_query, query_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_visited, h_visited, num_neighbors * sizeof(int), cudaMemcpyHostToDevice);

    int h_initial_candidate_count = 0;
    int h_initial_top_count = 0;
    cudaMemcpy(d_candidate_count, &h_initial_candidate_count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_top_count, &h_initial_top_count, sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (num_neighbors + blockSize - 1) / blockSize;

    process_neighbors<<<numBlocks, blockSize>>>(
        d_neighbors, num_neighbors, d_layer, d_query,
        d_visited, d_candidates, d_candidate_count,
        d_top_candidates, d_top_count, top_dist, ef
    );

    cudaMemcpy(h_candidate_count, d_candidate_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_top_count, d_top_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_candidates, d_candidates, (*h_candidate_count) * sizeof(Neighbor), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_top_candidates, d_top_candidates, (*h_top_count) * sizeof(Neighbor), cudaMemcpyDeviceToHost);

    cudaFree(d_neighbors);
    cudaFree(d_layer);
    cudaFree(d_query);
    cudaFree(d_visited);
    cudaFree(d_candidates);
    cudaFree(d_top_candidates);
    cudaFree(d_candidate_count);
    cudaFree(d_top_count);
}

#endif // HNSW_PROCESS_NEIGHBORS_CUH