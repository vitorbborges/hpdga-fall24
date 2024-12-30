#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

struct SharedBloomFilter {
    static const int NUM_BITS = 104000;  // Total number of bits
    static const int NUM_HASH = 7;      // Number of hash functions
    static const int NUM_WORDS = NUM_BITS / 64; // Number of 64-bit words
    uint64_t data[NUM_WORDS];           // Bit array

    __device__ void init() {
        for (int i = 0; i < NUM_WORDS; ++i) {
            data[i] = 0; // Initialize all bits to 0
        }
        printf("Bloom Filter initialized with %d bits\n", NUM_BITS);
    }

    // First hash function
    __device__ int hash1(int key) {
        return (key * 31) % NUM_BITS;
    }

    // Second hash function
    __device__ int hash2(int key) {
        return (key * 17 + 7) % NUM_BITS;
    }

    // Combined hash function
    __device__ int hash(int h, int key) {
        int result = (hash1(key) + h * hash2(key)) % NUM_BITS;
        return (result < 0) ? result + NUM_BITS : result; // Handle negative results
    }

    // Set a specific bit in the Bloom Filter
 __device__ void set(int key) {
    for (int i = 0; i < NUM_HASH; ++i) {
        int bit_index = hash(i, key);
        if (bit_index < 0 || bit_index >= NUM_BITS) {
            printf("Set Error: Key=%d, Hash=%d, BitIndex=%d out of bounds\n", key, i, bit_index);
            return; // Exit if out of bounds
        }

        int word_index = bit_index / 64;       // Locate the 64-bit word
        int bit_position = bit_index % 64;    // Locate the bit position
        if (word_index < 0 || word_index >= NUM_WORDS) {
            printf("Set Error: Key=%d, Hash=%d, WordIndex=%d out of bounds\n", key, i, word_index);
            return; // Exit if out of bounds
        }

        // Safely set the bit using atomicOr
        atomicOr(reinterpret_cast<unsigned long long*>(&data[word_index]),
                 (1ULL << bit_position));
    }
}

__device__ bool test(int key) {
    for (int i = 0; i < NUM_HASH; ++i) {
        int bit_index = hash(i, key);
        if (bit_index < 0 || bit_index >= NUM_BITS) {
            printf("Test Error: Key=%d, Hash=%d, BitIndex=%d out of bounds\n", key, i, bit_index);
            return false; // Out of bounds
        }

        int word_index = bit_index / 64;       // Locate the 64-bit word
        int bit_position = bit_index % 64;    // Locate the bit position
        if (word_index < 0 || word_index >= NUM_WORDS) {
            printf("Test Error: Key=%d, Hash=%d, WordIndex=%d out of bounds\n", key, i, word_index);
            return false; // Out of bounds
        }

        if (!((data[word_index] >> bit_position) & 1ULL)) {
            return false; // If any bit is not set, the key is not present
        }
    }
    return true; // All bits are set, the key is probably present
}

};
