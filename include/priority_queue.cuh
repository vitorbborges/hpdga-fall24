#ifndef HNSW_PRIORITY_QUEUE_CUH
#define HNSW_PRIORITY_QUEUE_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "device_data_structures.cuh"

/*
    gpu priority queue implementation based on the paper
    Symmetric Min-Max heap: A simpler data structure for double-ended priority queue
    by A. Arvind, C.Pandu Rangan

    this is done in order to prune the pq up to k
*/

;template <typename T = float>
class PriorityQueue {
private:
    d_Neighbor<T>* neighbors;
    int size;

public:
    __device__ PriorityQueue(d_Neighbor<T>* neighbors) : 
        neighbors(neighbors),
        size(0) {}

    __device__ ~PriorityQueue() {}

    __device__ void inline swap(d_Neighbor<T>& a, d_Neighbor<T>& b) {
        d_Neighbor<T> c(a);
        a = b;
        b = c;
    }

    __device__ void insert(d_Neighbor<T>& entry) {
        int idx = size;
        neighbors[size].dist = entry.dist;
        neighbors[size].id = entry.id;
        size++;

        while (idx > 0) {
            int parent = (idx - 1) / 2;

            // Adjust based on level: even = min-level, odd = max-level
            if ((idx & 1) == 0) { // Right child (max-level)
                if (neighbors[idx].dist > neighbors[parent].dist) {
                    swap(neighbors[idx], neighbors[parent]);
                    idx = parent;
                    continue;
                }
            } else { // Left child (min-level)
                if (neighbors[idx].dist < neighbors[parent].dist) {
                    swap(neighbors[idx], neighbors[parent]);
                    idx = parent;
                    continue;
                }
            }

        // Grandparent adjustment
        int grandparent = (idx - 3) / 4;
        if (grandparent >= 0) {
            if (neighbors[idx].dist < neighbors[grandparent].dist) {
                swap(neighbors[idx], neighbors[grandparent]);
                idx = grandparent;
            } else if (neighbors[idx].dist > neighbors[grandparent + 1].dist) {
                swap(neighbors[idx], neighbors[grandparent + 1]);
                idx = grandparent + 1;
            } else {
                break;
            }
        } else {
                break;
        }

        }
    }

    __device__ void deleteAt(int idx) {
        neighbors[idx] = neighbors[--size];
        int current = idx;

        while (current < size) {
            int left = 2 * current + 1;
            int right = 2 * current + 2;

            if (left >= size) break;

            int swapIdx = left;

            // Determine whether to adjust for min or max levels
            if ((current & 1) == 0) { // Even level -> Max
                if (right < size && neighbors[right].dist > neighbors[left].dist) swapIdx = right;
                if (neighbors[current].dist < neighbors[swapIdx].dist) {
                    swap(neighbors[current], neighbors[swapIdx]);
                    current = swapIdx;
                } else break;
            } else { // Odd level -> Min
                if (right < size && neighbors[right].dist < neighbors[left].dist) swapIdx = right;
                if (neighbors[current].dist > neighbors[swapIdx].dist) {
                    swap(neighbors[current], neighbors[swapIdx]);
                    current = swapIdx;
                } else break;
            }
        }
    }

    __device__ d_Neighbor<T> pop_max() {
        if (size <= 2) return neighbors[0]; // Return root if only one element exists
        d_Neighbor<T> ret = neighbors[2];        // Index 2 contains the maximum
        deleteAt(2);
        return ret;
    }

    __device__ d_Neighbor<T> pop_min() {
        if (size <= 1) return neighbors[0]; // Return root if only one element exists
        d_Neighbor<T> ret = neighbors[1];        // Index 1 contains the minimum
        deleteAt(1);
        return ret;
    }

    __device__ void print_heap() {
        int level = 1, count = 0;
        for (int i = 0; i < size; i++) {
            printf("(id: %d, dist:%f) :", neighbors[i].id, neighbors[i].dist);
            if (++count == level) {
                printf("\n");
                level <<= 1;
                count = 0;
            }
        }
        printf("top_id : %d", neighbors[0].id);
    }

    __device__ T get_size() {
        return size;
    }

    __device__ d_Neighbor<T>* get_neighbors() {
        return neighbors;
    }

    __device__ d_Neighbor<T> operator [] (int idx) {
        return neighbors[idx];
    }

    __device__ d_Neighbor<T> top() {
        return neighbors[0];
    }
}

#endif // HNSW_PRIORITY_QUEUE_CUH