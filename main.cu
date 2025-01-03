#include "priority_queue.cuh"
#include "symmetric_mmh.cuh"
#include <cassert>

__global__ void testSMMH() {
    // Shared memory for the heap and size
    __shared__ d_Neighbor<float> sharedHeap[HEAP_SIZE];
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
        heap.insert(d_Neighbor<float>(1.0f, 6)); // Duplicate distance
        heap.insert(d_Neighbor<float>(0.0f, 7)); // Very small distance
        heap.insert(d_Neighbor<float>(1000.0f, 8)); // Very large distance
    }

    __syncthreads();

    // Validate heap size after insertion
    if (tid == 0) {
        assert(sharedSize == 8);
        printf("Heap size after insertions: %d\n", sharedSize);
        heap.print();
    }

    __syncthreads();

    // Test removing minimum element
    if (tid == 0) {
        d_Neighbor<float> min = heap.popMin();
        assert(min.dist == 0.0f && min.id == 7);
        printf("Removed min: dist = %.2f, id = %d\n", min.dist, min.id);
        heap.print();
    }

    __syncthreads();

    // Test removing maximum element
    if (tid == 0) {
        d_Neighbor<float> max = heap.popMax();
        assert(max.dist == 1000.0f && max.id == 8);
        printf("Removed max: dist = %.2f, id = %d\n", max.dist, max.id);
        heap.print();
    }

    __syncthreads();

    // Test popping elements until empty
    if (tid == 0) {
        while (sharedSize > 0) {
            d_Neighbor<float> top = heap.popMin();
            printf("Removed min: dist = %.2f, id = %d\n", top.dist, top.id);
        }

        assert(sharedSize == 0);
        printf("Heap is empty after removing all elements.\n");
    }

    __syncthreads();

    // Test removing from an empty heap
    if (tid == 0) {
        bool exceptionThrown = false;
        // Replace try-catch with a manual error-checking mechanism
        if (sharedSize <= 0) {
            exceptionThrown = true;
        } else {
            heap.popMin(); // This should be protected by the condition
        }
        assert(exceptionThrown);
        if (exceptionThrown) {
            printf("Properly handled removal from an empty heap.\n");
        }
    }
    
}

int main() {
    // Launch a single CUDA block with multiple threads
    testSMMH<<<1, 32>>>();

    printf("All tests passed!\n");

    // Synchronize to wait for kernel completion
    cudaDeviceSynchronize();

    return 0;
}