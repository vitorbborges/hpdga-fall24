#ifndef HNSW_PRIORITY_QUEUE_CUH
#define HNSW_PRIORITY_QUEUE_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

#include "device_data_structures.cuh"

/*
    gpu priority queue implementation based on the paper
    Symmetric Min-Max heap: A simpler data structure for double-ended priority queue
    by A. Arvind, C.Pandu Rangan

    this is done in order to prune the pq up to k
*/

#define MAX_HEAP_SIZE 1024

// Define an enum for heap type
enum HeapType {
    MIN_HEAP,
    MAX_HEAP
};


template <typename T = float>
class PriorityQueue {
private:
    d_Neighbor<T>* heap;
    int* size;
    HeapType heapType;

    __device__ bool compare(const d_Neighbor<T>& a, const d_Neighbor<T>& b) {
        return heapType == MIN_HEAP ? a < b : a > b;
    }

    __device__ void heapify_up(int index) {
        while (index > 0) {
            int parent = (index - 1) / 2;
            if (compare(heap[index], heap[parent])) {
                d_Neighbor<T> temp = heap[index];
                heap[index] = heap[parent];
                heap[parent] = temp;
                index = parent;
            } else {
                break;
            }
        }
    }

    __device__ void heapify_down(int index) {
        while (index < *size) {
            int left = 2 * index + 1;
            int right = 2 * index + 2;
            int target = index;

            if (left < *size && compare(heap[left], heap[target])) {
                target = left;
            }
            if (right < *size && compare(heap[right], heap[target])) {
                target = right;
            }

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
    __device__ PriorityQueue(d_Neighbor<T>* shared_heap, int* shared_size, HeapType type)
        : heap(shared_heap), size(shared_size), heapType(type) {}

    __device__ void insert(const d_Neighbor<T>& value) {
        heap[*size] = value;
        heapify_up((*size)++);
    }

    __device__ d_Neighbor<T> pop() {
        if (*size == 0) return { -1.0f, -1 }; // Heap underflow
        d_Neighbor<T> top_value = heap[0];
        heap[0] = heap[--(*size)];
        heapify_down(0);
        return top_value;
    }

    __device__ d_Neighbor<T> top() {
        if (*size == 0) return { -1.0f, -1 }; // Heap underflow
        return heap[0];
    }

    __device__ void print_heap() {
        if (threadIdx.x == 0) {
            printf("Heap: ");
            for (int i = 0; i < *size; i++) {
                printf("(%f, %d) ", heap[i].dist, heap[i].id);
            }
            printf("top_id: [%d], size: [%d]\n", top().id, *size);
        }
        __syncthreads();
    }

    __device__ int get_size() {
        return *size;
    }
};

#endif // HNSW_PRIORITY_QUEUE_CUH