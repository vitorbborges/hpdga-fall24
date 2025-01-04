#ifndef HNSW_SYMMETRIC_MMH_CUH
#define HNSW_SYMMETRIC_MMH_CUH

#include "data_structures.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace ds;

template <typename T> class SymmetricMinMaxHeap {
private:
  d_Neighbor<T> *heap; // Shared memory array for the heap
  int size;            // Current number of elements in the heap
  HeapType type;       // Type of heap: MIN_HEAP or MAX_HEAP
  size_t capacity;     // Maximum allowed size of the heap

  __device__ int parent(int x) const { return ((x - 1) >> 1); }
  __device__ int grandparent(int x) const { return ((x - 3) >> 2); }
  __device__ int leftChild(int x) const { return ((x << 1) + 1); }
  __device__ int rightChild(int x) const { return ((x << 1) + 2); }
  __device__ bool isLeaf(unsigned int x) const { return leftChild(x) >= size; }

  // Swaps two heap elements at given indices
  __device__ void swap(int idx1, int idx2) {
    d_Neighbor<T> temp = heap[idx1];
    heap[idx1] = heap[idx2];
    heap[idx2] = temp;
  }

  // Adjusts sibling nodes to maintain order
  __device__ int adjustSibling(int idx) {
    int s;
    if (idx & 1) { // Left child case
      s = idx + 1;
      if (s >= size)
        return idx; // No sibling
      if (heap[idx].dist > heap[s].dist) {
        swap(idx, s);
        return s;
      }
    } else { // Right child case
      s = idx - 1;
      if (heap[idx].dist < heap[s].dist) {
        swap(idx, s);
        return s;
      }
    }
    return idx;
  }

  // Adjusts based on grandparent relationships for heap property
  __device__ int adjustGrandparent(int idx) {
    if (idx <= 2)
      return idx; // Root level, no grandparent
    int G = grandparent(idx);
    int GL = leftChild(G), GR = rightChild(G);
    if (heap[GL].dist > heap[idx].dist) {
      swap(GL, idx);
      return GL;
    } else if (heap[GR].dist < heap[idx].dist) {
      swap(GR, idx);
      return GR;
    }
    return idx;
  }

  // Adjusts grandchild nodes to maintain heap order
  __device__ int adjustGrandchild(int idx) {
    if (idx & 1) { // Left child case
      if (isLeaf(idx))
        return idx;
      int CL = leftChild(idx), CR = leftChild(idx + 1);
      int C = CL;
      if (CR < size && heap[CR].dist < heap[CL].dist)
        C = CR;
      if (heap[C].dist < heap[idx].dist) {
        swap(C, idx);
        return C;
      }
    } else { // Right child case
      int CL = rightChild(idx - 1), CR = rightChild(idx);
      if (CL >= size)
        return idx; // No children
      int C = CL;
      if (CR < size && heap[CR].dist > heap[CL].dist)
        C = CR;
      if (heap[C].dist > heap[idx].dist) {
        swap(C, idx);
        return C;
      }
    }
    return idx;
  }

public:
  // Constructor initializes heap in shared memory
  __device__ SymmetricMinMaxHeap(d_Neighbor<T> *sharedHeap,
                                 HeapType type,
                                 int maximumSize = 1) {
    heap = sharedHeap;
    size = 1; // Start size at 1 for consistent indexing
    type = type;
    capacity = maximumSize;
  }

  // Inserts a new element into the heap
  __device__ void insert(d_Neighbor<T> value) {
    if (getSize() >= capacity) { // Handle capacity limit
      d_Neighbor<T> insertValue;
      d_Neighbor<T> bott = bottom();
      if (type == MIN_HEAP) {
        popMax(); // Remove max for MIN_HEAP
        insertValue = (value.dist < bott.dist) ? value : bott;
      } else {
        popMin(); // Remove min for MAX_HEAP
        insertValue = (value.dist > bott.dist) ? value : bott;
      }
      insert(insertValue);
      return;
    }

    int Y = size;
    heap[size++] = value;

    // Adjust heap properties iteratively
    while (true) {
      Y = adjustSibling(Y);
      int X = adjustGrandparent(Y);
      if (X == Y)
        break; // Stop when no adjustments needed
      Y = X;
    }
  }

  // Removes the element at a specified index
  __device__ void deletion(int idx) {
    heap[idx] = heap[--size]; // Replace with last element
    int Y = idx;
    while (true) {
      Y = adjustSibling(Y);
      int X = adjustGrandchild(Y);
      if (X == Y)
        break; // Stop when no adjustments needed
      Y = X;
    }
  }

  // Removes and returns the smallest element
  __device__ d_Neighbor<T> popMin() {
    if (size == 1) {
      return d_Neighbor<T>(); // Empty heap case
    }
    d_Neighbor<T> ret = heap[1];
    deletion(1);
    return ret;
  }

  // Removes and returns the largest element
  __device__ d_Neighbor<T> popMax() {
    if (size <= 1)
      return popMin(); // Fallback to popMin for single element

    d_Neighbor<T> ret = heap[2];
    deletion(2);
    return ret;
  }

  // Prints the heap contents (for debugging)
  __device__ void print() {
    printf("Heap (size = %d):\n", getSize());
    for (int i = 0; i < size - 1; ++i) {
      printf("Index %d: dist = %.2f, id = %d\n", i, heap[i + 1].dist,
             heap[i + 1].id);
    }
    printf("\n");
  }

  // Returns the number of elements in the heap
  __device__ int getSize() const { return size - 1; }

  // Checks if the heap is empty
  __device__ bool isEmpty() const { return size == 1; }

  // Returns the top (smallest or largest) element
  __device__ d_Neighbor<T> top() const {
    if (type == MIN_HEAP) {
      return heap[1];
    } else {
      return heap[2];
    }
  }

  // Returns the bottom element
  __device__ d_Neighbor<T> bottom() const {
    if (type == MIN_HEAP && getSize() > 1) {
      return heap[2];
    } else if (getSize() > 1) {
      return heap[1];
    } else {
      return heap[1];
    }
  }

  // Pops the top element according to heap type
    __device__ d_Neighbor<T> pop() {
        if (type == MIN_HEAP) {
        return popMin();
        } else {
        return popMax();
        }
    }
};

#endif // HNSW_SYMMETRIC_MMH_CUH
