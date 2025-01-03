#include "priority_queue.cuh"
#include "symmetric_mmh.cuh"


__global__ void testSMMH() {
    // Shared memory for the heap and size
    extern __shared__ d_Neighbor<float> sharedHeap[];
     __shared__ int sharedSize;

    // Initialize the heap
    SymmetricMinMaxHeap<float> heap;
    heap.init(sharedHeap, &sharedSize);

    int tid = threadIdx.x;

    // Insert elements into the heap
    if (tid == 0) {
        heap.insert(d_Neighbor<float>(5.0f, 1));
        heap.insert(d_Neighbor<float>(3.0f, 2));
        heap.insert(d_Neighbor<float>(8.0f, 3));
        heap.insert(d_Neighbor<float>(1.0f, 4));
        heap.insert(d_Neighbor<float>(10.0f, 5));
    }

    __syncthreads();

    // Print the heap after insertions
    if (tid == 0) {
        printf("Heap after insertions:\n");
        heap.print();
    }

    __syncthreads();

    // Remove the minimum element
    if (tid == 0) {
        d_Neighbor<float> min = heap.popMin();
        printf("Removed min: dist = %.2f, id = %d\n", min.dist, min.id);

        printf("Heap after removing min:\n");
        heap.print();
    }

    __syncthreads();

    // Remove the maximum element
    if (tid == 0) {
        d_Neighbor<float> max = heap.popMax();
        printf("Removed max: dist = %.2f, id = %d\n", max.dist, max.id);

        printf("Heap after removing max:\n");
        heap.print();
    }
}



// __global__ void test_priority_queue(Neighbor* data_points, int num_points, HeapType heapType) {
//     __shared__ Neighbor shared_heap[MAX_HEAP_SIZE];
//     __shared__ int shared_size;

//     PriorityQueue pq(shared_heap, &shared_size, heapType);

//     if (threadIdx.x == 0) {
//         // Insert and extract elements in unusual orders
//         pq.insert({50.0, 1});
//         pq.insert({10.0, 2});
//         pq.insert({70.0, 3});
//         pq.print_heap();

//         Neighbor top = pq.top();
//         printf("Accessed Top (without extraction): (%f, %d)\n", top.distance, top.id);

//         pq.insert({30.0, 4});
//         pq.print_heap();
//         pq.insert({90.0, 5});
//         pq.print_heap();

//         top = pq.top();
//         printf("Accessed Top (without extraction): (%f, %d)\n", top.distance, top.id);

//         top = pq.pop();
//         printf("Extracted Top: (%f, %d)\n", top.distance, top.id);
//         pq.print_heap();

//         pq.insert({20.0, 6});
//         pq.insert({40.0, 7});
//         pq.print_heap();

//         while (shared_size > 0) {
//             top = pq.top();
//             printf("Accessed Top (without extraction): (%f, %d)\n", top.distance, top.id);
//             top = pq.pop();
//             printf("Extracted Top: (%f, %d)\n", top.distance, top.id);
//             pq.print_heap();
//         }
//     }
// }

// int main() {
//     const int num_points = 0; // No initial points needed for this test

//     Neighbor* d_data_points;
//     cudaMalloc(&d_data_points, num_points * sizeof(Neighbor));

//     test_priority_queue<<<1, 32>>>(d_data_points, num_points, MAX_HEAP);
//     cudaDeviceSynchronize();

//     cudaFree(d_data_points);
//     return 0;
// }

int main() {
    // Launch a single CUDA block with one thread
    int sharedMemSize = HEAP_SIZE * sizeof(d_Neighbor<float>);

    testSMMH<<<1, 32, sharedMemSize>>>();

    // Synchronize to wait for kernel completion
    cudaDeviceSynchronize();

    return 0;
}
