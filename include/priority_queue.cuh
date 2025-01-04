#ifndef HNSW_PRIORITY_QUEUE_CUH
#define HNSW_PRIORITY_QUEUE_CUH

#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "data_structures.cuh"

using namespace ds;

template <typename T = float> class PriorityQueue {
private:
  d_Neighbor<T> *heap; // Pointer to the heap array in shared memory
  int *size;           // Pointer to the size of the heap in shared memory
  HeapType heapType;   // Specifies if this is a min-heap or max-heap

  // Comparison function to maintain heap order based on the type (min or max
  // heap)
  __device__ bool compare(const d_Neighbor<T> &a, const d_Neighbor<T> &b) {
    return heapType == MIN_HEAP ? a < b : a > b;
  }

  // Moves an element up the heap to maintain heap order
  __device__ void heapify_up(int index) {
    while (index > 0) {
      int parent = (index - 1) / 2;
      if (compare(heap[index], heap[parent])) {
        // Swap the current element with its parent
        d_Neighbor<T> temp = heap[index];
        heap[index] = heap[parent];
        heap[parent] = temp;
        index = parent;
      } else {
        break;
      }
    }
  }

  // Moves an element down the heap to maintain heap order
  __device__ void heapify_down(int index) {
    while (index < *size) {
      int left = 2 * index + 1;
      int right = 2 * index + 2;
      int target = index;

      // Compare with left child
      if (left < *size && compare(heap[left], heap[target])) {
        target = left;
      }

      // Compare with right child
      if (right < *size && compare(heap[right], heap[target])) {
        target = right;
      }

      // If the current element is not in the correct position, swap it
      if (target != index) {
        d_Neighbor<T> temp = heap[index];
        heap[index] = heap[target];
        heap[target] = temp;
        index = target;
      } else {
        break;
      }
    }
  }

public:
  // Constructor to initialize the priority queue
  __device__ PriorityQueue(d_Neighbor<T> *shared_heap, int *shared_size,
                           HeapType type)
      : heap(shared_heap), size(shared_size), heapType(type) {}

  // Inserts a new element into the heap
  __device__ void insert(const d_Neighbor<T> &value) {
    heap[*size] = value;
    heapify_up((*size)++);
  }

  // Removes and returns the top element (min or max depending on heap type)
  __device__ d_Neighbor<T> pop() {
    if (*size == 0)
      return {-1.0f, -1}; // Heap underflow
    d_Neighbor<T> top_value = heap[0];
    heap[0] = heap[--(*size)];
    heapify_down(0);
    return top_value;
  }

  // Returns the top element without removing it
  __device__ d_Neighbor<T> top() {
    if (*size == 0)
      return {-1.0f, -1}; // Heap underflow
    return heap[0];
  }

  // Prints the contents of the heap (for debugging purposes)
  __device__ void print_heap() {
    printf("Heap: ");
    for (int i = 0; i < *size; i++) {
      printf("(%f, %d) ", heap[i].dist, heap[i].id);
    }
    printf("top_id: [%d], size: [%d]\n", top().id, *size);
  }

  // Returns the current size of the heap
  __device__ int get_size() { return *size; }
};

#endif // HNSW_PRIORITY_QUEUE_CUH
