#pragma once
#ifndef PRIORITY_QUEUE_CUH
#define PRIORITY_QUEUE_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

/*
    gpu priority queue implementation based on the paper
    Symmetric Min-Max heap: A simpler data structure for double-ended priority queue
    by A. Arvind, C.Pandu Rangan

    this is done in order to prune the pq up to k
*/

// TODO: Implement priority queue as a class (or struct) hold Neighbor objects
// TODO: Implement Neighbor object compatible with device code

template<class T>
__device__ void inline swap(T& a, T& b){
    T c(a); 
    a=b; 
    b=c;
}

template<class T>
__device__ void insert(T* smmh, int& size, T& entry) {
    int idx = size;
    smmh[size++] = entry;

    while (idx > 0) {
        int parent = (idx - 1) / 2;

        // Adjust based on level: even = min-level, odd = max-level
        if ((idx & 1) == 0) { // Right child (max-level)
            if (smmh[idx] > smmh[parent]) { 
                swap(smmh[idx], smmh[parent]);
                idx = parent; 
                continue; 
            }
        } else { // Left child (min-level)
            if (smmh[idx] < smmh[parent]) { 
                swap(smmh[idx], smmh[parent]); 
                idx = parent; 
                continue; 
            }
        }

        // Grandparent adjustment
        int grandparent = (idx - 3) / 4;
        if (grandparent >= 0) {
            if (smmh[idx] < smmh[grandparent]) {
                swap(smmh[idx], smmh[grandparent]);
                idx = grandparent;
            } else if (smmh[idx] > smmh[grandparent + 1]) {
                swap(smmh[idx], smmh[grandparent + 1]);
                idx = grandparent + 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }
}

template<class T>
__device__ void deleteAt(T* smmh, int idx, int& size) {
    smmh[idx] = smmh[--size];
    int current = idx;

    while (current < size) {
        int left = 2 * current + 1;
        int right = 2 * current + 2;

        if (left >= size) break;

        int swapIdx = left;

        // Determine whether to adjust for min or max levels
        if ((current & 1) == 0) { // Even level -> Max
            if (right < size && smmh[right] > smmh[left]) swapIdx = right;
            if (smmh[current] < smmh[swapIdx]) {
                swap(smmh[current], smmh[swapIdx]);
                current = swapIdx;
            } else break;
        } else { // Odd level -> Min
            if (right < size && smmh[right] < smmh[left]) swapIdx = right;
            if (smmh[current] > smmh[swapIdx]) {
                swap(smmh[current], smmh[swapIdx]);
                current = swapIdx;
            } else break;
        }
    }
}

template<class T>
__device__ T pop_max(T* smmh, int& size) {
    if (size <= 2) return smmh[0]; // Return root if only one element exists
    T ret = smmh[2];               // Index 2 contains the maximum
    deleteAt(smmh, 2, size);
    return ret;
}

template<class T>
__device__ T pop_min(T* smmh, int& size) {
    if (size <= 1) return smmh[0]; // Return root if only one element exists
    T ret = smmh[1];               // Index 1 contains the minimum
    deleteAt(smmh, 1, size);
    return ret;
}

template<class T>
__device__ void print_heap(T* smmh, int size) {
    int level = 1, count = 0;
    for (int i = 0; i < size; i++) {
        printf("%d\t", smmh[i]);
        if (++count == level) {
            printf("\n");
            level <<= 1;
            count = 0;
        }
    }
    printf("\n");
}


#endif // PRIORITY_QUEUE_CUH