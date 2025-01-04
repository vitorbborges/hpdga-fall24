#include "symmetric_mmh.cuh"
#include <cassert>

// Kernel to test the SymmetricMinMaxHeap implementation
__global__ void testSMMH() {
  // Shared memory for the heap
  extern __shared__ d_Neighbor<float> sharedHeap[];

  // Initialize the heap with MIN_HEAP type and maximum size of 7
  SymmetricMinMaxHeap<float> heap(sharedHeap, MIN_HEAP, 7);

  int tid = threadIdx.x;

  // Insert elements into the heap (only performed by thread 0)
  if (tid == 0) {
    heap.insert(d_Neighbor<float>(5.0f, 1));
    heap.print();
    heap.insert(d_Neighbor<float>(3.0f, 2));
    heap.print();
    heap.insert(d_Neighbor<float>(1.0f, 4));
    heap.print();
    heap.insert(d_Neighbor<float>(10.0f, 5));
    heap.print();
    heap.insert(d_Neighbor<float>(1.0f, 6)); // Duplicate distance
    heap.print();
    heap.insert(d_Neighbor<float>(0.0f, 7)); // Very small distance
    heap.print();
    heap.insert(d_Neighbor<float>(1000.0f, 8)); // Very large distance
    heap.print();
  }

  __syncthreads(); // Ensure all threads synchronize before validation

  // Validate heap size after insertion
  if (tid == 0) {
    printf("Heap size after insertions: %d\n", heap.getSize());
    assert(heap.getSize() == 7); // Assert correct size
    heap.print();
  }

  __syncthreads();

  // Test removing the minimum element
  if (tid == 0) {
    d_Neighbor<float> min = heap.popMin();
    printf("Removed min: dist = %.2f, id = %d\n", min.dist, min.id);
    assert(min.dist == 0.0f && min.id == 7); // Validate min element
    heap.print();
  }

  __syncthreads();

  // Test removing the maximum element
  if (tid == 0) {
    d_Neighbor<float> max = heap.popMax();
    printf("Removed max: dist = %.2f, id = %d\n", max.dist, max.id);
    assert(max.dist == 1000.0f && max.id == 8); // Validate max element
    heap.print();
  }

  __syncthreads();

  // Test inserting a new element and popping min/max
  if (tid == 0) {
    heap.insert(d_Neighbor<float>(2.0f, 9));
    heap.print();
    d_Neighbor<float> min = heap.popMin();
    printf("Removed min: dist = %.2f, id = %d\n", min.dist, min.id);
    assert(min.dist == 1.0f && min.id == 4); // Validate min
    heap.print();
    d_Neighbor<float> max = heap.popMax();
    printf("Removed max: dist = %.2f, id = %d\n", max.dist, max.id);
    assert(max.dist == 10.0f && max.id == 5); // Validate max
    heap.print();
  }

  __syncthreads();

  // Test inserting a new minimum and popping min/max
  if (tid == 0) {
    heap.insert(d_Neighbor<float>(0.5f, 10));
    heap.print();
    d_Neighbor<float> min = heap.popMin();
    printf("Removed min: dist = %.2f, id = %d\n", min.dist, min.id);
    assert(min.dist == 0.5f && min.id == 10); // Validate new min
    heap.print();
    d_Neighbor<float> max = heap.popMax();
    printf("Removed max: dist = %.2f, id = %d\n", max.dist, max.id);
    assert(max.dist == 5.0f && max.id == 1); // Validate new max
    heap.print();
  }

  // Test inserting a new maximum and popping min/max
  if (tid == 0) {
    heap.insert(d_Neighbor<float>(1001.0f, 11));
    heap.print();
    d_Neighbor<float> min = heap.popMin();
    printf("Removed min: dist = %.2f, id = %d\n", min.dist, min.id);
    assert(min.dist == 1.0f && min.id == 6); // Validate min after insertion
    heap.print();
    d_Neighbor<float> max = heap.popMax();
    printf("Removed max: dist = %.2f, id = %d\n", max.dist, max.id);
    assert(max.dist == 1001.0f && max.id == 11); // Validate new max
    heap.print();
  }

  if (tid == 0) {
    printf("All tests passed!\n"); // Indicate test completion
  }
}

int main() {
  // Launch the kernel with one block of 32 threads
  testSMMH<<<1, 32>>>();

  // Catch and handle potential errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    fprintf(stderr, "ERROR: %s\n", cudaGetErrorString(error));
    return 1;
  }

  // Synchronize to wait for kernel completion
  cudaDeviceSynchronize();
  return 0;
}
