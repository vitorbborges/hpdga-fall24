#include "priority_queue.cuh"

__global__ void test_priority_queue() {
    __shared__ Neighbor shared_heap[MAX_HEAP_SIZE];
    __shared__ int shared_size;
    __shared__ Neighbor values[10];

    PriorityQueue pq(shared_heap, &shared_size, false);

    if (threadIdx.x == 0) {
        printf("Initializing shared memory in block %d\n", blockIdx.x);
        shared_size = 0; // Initialize size in shared memory
    }
    __syncthreads();

    // Each thread writes its value to shared memory
    if (threadIdx.x < 10) {
        values[threadIdx.x] = {static_cast<float>(threadIdx.x * 10 + 5), static_cast<int>(threadIdx.x)};
        printf("Thread %d generated value (%f, %d)\n", threadIdx.x, values[threadIdx.x].distance, values[threadIdx.x].id);
    }
    __syncthreads();

    // Only one thread performs all insertions
    if (threadIdx.x == 0) {
        for (int i = 0; i < 10; ++i) {
            printf("Inserting value (%f, %d) into the heap\n", values[i].distance, values[i].id);
            pq.insert(values[i]); // Insert values sequentially
        }
    }
    __syncthreads();

    // Validation phase: Print the heap
    if (threadIdx.x == 0) {
        printf("Heap after insertions (block %d):\n", blockIdx.x);
        pq.print_heap();
    }
    __syncthreads();

    // Popping phase: Only one thread performs min/max pops
    if (threadIdx.x == 0) {
        for (int i = 0; i < 5 && shared_size > 0; ++i) {
            Neighbor top = pq.extract_top();
            printf("Iteration %d (block %d): Top = (%f, %d)\n", i, blockIdx.x, top.distance, top.id);
        }
    }
    __syncthreads();

    // Final validation: Print the heap after popping
    if (threadIdx.x == 0) {
        printf("Heap after popping elements (block %d):\n", blockIdx.x);
        pq.print_heap();
    }
}

int main() {
    test_priority_queue<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
