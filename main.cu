#include <cuda_runtime.h>
#include <iostream>
#include "priority_queue.cuh"

__global__ void testPriorityQueue(d_Neighbor<float>* neighbors, int neighborCount) {
    // Initialize priority queue
    PriorityQueue<float> pq(neighbors);

    // Insert elements into the priority queue
    for (int i = 0; i < neighborCount; ++i) {
        d_Neighbor<float> entry((float)(i), i); // Example: decreasing distances
        pq.insert(entry);
        printf("Iteration %d: Inserted (%f, %d)\n", i, entry.dist, entry.id);
        pq.print_heap();
    }
}

int main() {
    const int neighborCount = 10;

    // Allocate memory for neighbors on device
    d_Neighbor<float>* d_neighbors;
    cudaMalloc(&d_neighbors, neighborCount * sizeof(d_Neighbor<float>));

    // Launch kernel to test the PriorityQueue
    testPriorityQueue<<<1, 1>>>(d_neighbors, neighborCount);

    // Synchronize and free memory
    cudaDeviceSynchronize();
    cudaFree(d_neighbors);

    return 0;
}
