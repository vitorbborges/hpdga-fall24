#include "priority_queue.cuh"
#include <cassert>

__global__ void testPriorityQueue() {
  __shared__ d_Neighbor<float> sharedHeap[MAX_HEAP_SIZE];
  __shared__ int sharedSize;

  int tid = threadIdx.x;

  if (tid == 0) {
    // Initialize Priority Queue as a Min-Heap
    PriorityQueue<float> minHeap(sharedHeap, &sharedSize, MIN_HEAP);
    sharedSize = 0;

    // Insert elements
    minHeap.insert(d_Neighbor<float>(5.0f, 1));
    minHeap.insert(d_Neighbor<float>(3.0f, 2));
    minHeap.insert(d_Neighbor<float>(8.0f, 3));
    minHeap.insert(d_Neighbor<float>(1.0f, 4));
    minHeap.insert(d_Neighbor<float>(10.0f, 5));

    // Validate size
    assert(minHeap.get_size() == 5);
    printf("Min-Heap size after insertions: %d\n", minHeap.get_size());
    minHeap.print_heap();

    // Check top element
    d_Neighbor<float> top = minHeap.top();
    assert(top.dist == 1.0f && top.id == 4);
    printf("Min-Heap top element: dist = %.2f, id = %d\n", top.dist, top.id);

    // Pop top element
    d_Neighbor<float> removed = minHeap.pop();
    assert(removed.dist == 1.0f && removed.id == 4);
    printf("Min-Heap removed element: dist = %.2f, id = %d\n", removed.dist,
           removed.id);
    minHeap.print_heap();

    // Validate size after removal
    assert(minHeap.get_size() == 4);

    // Test Max-Heap
    PriorityQueue<float> maxHeap(sharedHeap, &sharedSize, MAX_HEAP);
    sharedSize = 0;

    // Insert elements
    maxHeap.insert(d_Neighbor<float>(5.0f, 1));
    maxHeap.insert(d_Neighbor<float>(3.0f, 2));
    maxHeap.insert(d_Neighbor<float>(8.0f, 3));
    maxHeap.insert(d_Neighbor<float>(1.0f, 4));
    maxHeap.insert(d_Neighbor<float>(10.0f, 5));

    // Validate size
    assert(maxHeap.get_size() == 5);
    printf("Max-Heap size after insertions: %d\n", maxHeap.get_size());
    maxHeap.print_heap();

    // Check top element
    top = maxHeap.top();
    assert(top.dist == 10.0f && top.id == 5);
    printf("Max-Heap top element: dist = %.2f, id = %d\n", top.dist, top.id);

    // Pop top element
    removed = maxHeap.pop();
    assert(removed.dist == 10.0f && removed.id == 5);
    printf("Max-Heap removed element: dist = %.2f, id = %d\n", removed.dist,
           removed.id);
    maxHeap.print_heap();

    // Validate size after removal
    assert(maxHeap.get_size() == 4);

    // Pop remaining elements from Max-Heap
    while (maxHeap.get_size() > 0) {
      removed = maxHeap.pop();
      printf("Max-Heap removed element: dist = %.2f, id = %d\n", removed.dist,
             removed.id);
    }
    assert(maxHeap.get_size() == 0);
    printf("Max-Heap is empty after all removals.\n");

    // Attempt to pop from an empty heap
    d_Neighbor<float> empty = maxHeap.pop();
    assert(empty.dist == -1.0f && empty.id == -1);
    printf("Handled popping from empty heap correctly.\n");
  }
}

int main() {
  // Launch kernel to test PriorityQueue
  testPriorityQueue<<<1, 32>>>();

  // Synchronize to wait for kernel completion
  cudaDeviceSynchronize();

  // Print final test confirmation
  printf("All PriorityQueue tests passed!\n");

  return 0;
}
