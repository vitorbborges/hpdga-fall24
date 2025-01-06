#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

#include "data_structures.cuh"

using namespace ds;

#define HEAP_SIZE 128  

template <typename T>
struct SymmetricMinMaxHeap {
    d_Neighbor<T>* heap;  
    int* size;            

    __device__ void init(d_Neighbor<T>* sharedHeap, int* sharedSize) {
        heap = sharedHeap;  
        size = sharedSize;  // Assign shared memory to size
        *size = 0;          // Initialize heap size
    }

    // Get parent and child indices
    __device__ int parent(int idx) { return (idx - 1) / 2; }
    __device__ int leftChild(int idx) { return 2 * idx + 1; }
    __device__ int rightChild(int idx) { return 2 * idx + 2; }

    // Check if a level is min or max
    __device__ bool isMinLevel(int idx) { return __ffs(idx + 1) % 2 == 1; }

    // Insert an element into the heap
    __device__ void insert(d_Neighbor<T> value) {
        int idx = atomicAdd(size, 1);  // Get the next available position atomically
        if (idx >= HEAP_SIZE) {
            printf("Heap overflow!\n");
            atomicSub(size, 1);  // Revert the size increment
            return;
        }

        heap[idx] = value;  // Place the element
        heapifyUp(idx);     // Reheapify up
    }

    // Heapify up to maintain the heap property
    __device__ void heapifyUp(int idx) {
        if (idx == 0) return;  // Root node, no parent

        int p = parent(idx);
        if (isMinLevel(idx)) {
            if (heap[idx].dist > heap[p].dist) {
                swap(idx, p);
                heapifyUpMax(p);
            } else {
                heapifyUpMin(idx);
            }
        } else {
            if (heap[idx].dist < heap[p].dist) {
                swap(idx, p);
                heapifyUpMin(p);
            } else {
                heapifyUpMax(idx);
            }
        }
    }

    __device__ void heapifyUpMin(int idx) {
        int gp = parent(parent(idx));
        if (idx > 2 && heap[idx].dist < heap[gp].dist) {
            swap(idx, gp);
            heapifyUpMin(gp);
        }
    }

    __device__ void heapifyUpMax(int idx) {
        int gp = parent(parent(idx));
        if (idx > 2 && heap[idx].dist > heap[gp].dist) {
            swap(idx, gp);
            heapifyUpMax(gp);
        }
    }

    __device__ d_Neighbor<T> popMin() {
        if (*size == 0) {
            printf("Heap underflow!\n");
            return d_Neighbor<T>();  // Return default value
        }

        d_Neighbor<T> minVal = heap[0];
        heap[0] = heap[--(*size)];  // Replace root with last element
        heapifyDown(0);            // Reheapify down
        return minVal;
    }

    __device__ d_Neighbor<T> popMax() {
        if (*size <= 1) {
            return popMin();  // If only one element, popMin is equivalent
        }

        int maxIdx = (*size == 2 || heap[1].dist > heap[2].dist) ? 1 : 2;
        d_Neighbor<T> maxVal = heap[maxIdx];
        heap[maxIdx] = heap[--(*size)];  // Replace max with last element
        heapifyDown(maxIdx);            // Reheapify down
        return maxVal;
    }

    // Heapify down to maintain the heap property
    __device__ void heapifyDown(int idx) {
        if (isMinLevel(idx)) {
            heapifyDownMin(idx);
        } else {
            heapifyDownMax(idx);
        }
    }

    __device__ void heapifyDownMin(int idx) {
        int left = leftChild(idx), right = rightChild(idx);
        if (left >= *size) return;

        int smallest = left;
        if (right < *size && heap[right].dist < heap[left].dist) {
            smallest = right;
        }

        if (heap[idx].dist > heap[smallest].dist) {
            swap(idx, smallest);
            if (leftChild(smallest) < *size) {
                int l = leftChild(smallest), r = rightChild(smallest);
                int gSmallest = l;
                if (r < *size && heap[r].dist < heap[l].dist) {
                    gSmallest = r;
                }
                if (heap[smallest].dist > heap[gSmallest].dist) {
                    swap(smallest, gSmallest);
                    heapifyDownMin(gSmallest);
                }
            }
        }
    }

    __device__ void heapifyDownMax(int idx) {
        int left = leftChild(idx), right = rightChild(idx);
        if (left >= *size) return;

        int largest = left;
        if (right < *size && heap[right].dist > heap[left].dist) {
            largest = right;
        }

        if (heap[idx].dist < heap[largest].dist) {
            swap(idx, largest);
            if (leftChild(largest) < *size) {
                int l = leftChild(largest), r = rightChild(largest);
                int gLargest = l;
                if (r < *size && heap[r].dist > heap[l].dist) {
                    gLargest = r;
                }
                if (heap[largest].dist < heap[gLargest].dist) {
                    swap(largest, gLargest);
                    heapifyDownMax(gLargest);
                }
            }
        }
    }

    __device__ void swap(int idx1, int idx2) {
        d_Neighbor<T> temp = heap[idx1];
        heap[idx1] = heap[idx2];
        heap[idx2] = temp;
    }

    __device__ void print() {
        printf("Heap (size = %d):\n", *size);
        for (int i = 0; i < *size; ++i) {
            printf("Index %d: dist = %.2f, id = %d\n", i, heap[i].dist, heap[i].id);
        }
        printf("\n");
    }
};