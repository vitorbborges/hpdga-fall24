#ifndef HNSW_PRIORITY_QUEUE_CUH
#define HNSW_PRIORITY_QUEUE_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>

/*
    gpu priority queue implementation based on the paper
    Symmetric Min-Max heap: A simpler data structure for double-ended priority queue
    by A. Arvind, C.Pandu Rangan

    this is done in order to prune the pq up to k
*/

#define MAX_HEAP_SIZE 10

// Define the Neighbor struct
struct Neighbor {
    float distance;
    int id;

    __host__ __device__ bool operator<(const Neighbor& other) const {
        return distance < other.distance; // Max-heap based on distance
    }

    __host__ __device__ bool operator>(const Neighbor& other) const {
        return distance > other.distance; // Min-heap based on distance
    }
};

// Define an enum for heap type
enum HeapType {
    MIN_HEAP,
    MAX_HEAP
};

class PriorityQueue {
private:
    Neighbor* heap;
    int* size;
    HeapType heapType;

    __device__ bool compare(const Neighbor& a, const Neighbor& b) {
        return heapType == MIN_HEAP ? a < b : a > b;
    }

    __device__ void heapify_up(int index) {
        while (index > 0) {
            int parent = (index - 1) / 2;
            if (compare(heap[index], heap[parent])) {
                Neighbor temp = heap[index];
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
                Neighbor temp = heap[index];
                heap[index] = heap[target];
                heap[target] = temp;
                index = target;
            } else {
                break;
            }
        }
    }

public:
    __device__ PriorityQueue(Neighbor* shared_heap, int* shared_size, HeapType type)
        : heap(shared_heap), size(shared_size), heapType(type) {
        if (threadIdx.x == 0) {
            *size = 0; // Initialize size in shared memory
        }
        __syncthreads();
    }

    __device__ void insert(const Neighbor& value) {
        if (*size >= MAX_HEAP_SIZE) {
            // Remove the bottom element
            Neighbor bottom_value = heap[--(*size)];
            heapify_down(0);
            printf("Removed Bottom: (%f, %d)\n", bottom_value.distance, bottom_value.id);
        }
        heap[*size] = value;
        heapify_up((*size)++);
    }

    __device__ Neighbor pop() {
        if (*size == 0) return { -1.0f, -1 }; // Heap underflow
        Neighbor top_value = heap[0];
        heap[0] = heap[--(*size)];
        heapify_down(0);
        return top_value;
    }

    __device__ Neighbor top() {
        if (*size == 0) return { -1.0f, -1 }; // Heap underflow
        return heap[0];
    }

    __device__ void print_heap() {
        if (threadIdx.x == 0) {
            printf("Heap: ");
            for (int i = 0; i < *size; i++) {
                printf("(%f, %d) ", heap[i].distance, heap[i].id);
            }
            printf("\n");
        }
        __syncthreads();
    }
};

