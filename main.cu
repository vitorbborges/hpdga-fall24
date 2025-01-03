#include "bloom_filter.cuh"
#include <cassert>
#include <cstdio>

__global__ void testBloomFilter() {
  __shared__ SharedBloomFilter bloomFilter;

  // Initialize the Bloom filter (only one thread handles this)
  if (threadIdx.x == 0) {
    bloomFilter.init();
  }
  __syncthreads(); // Ensure initialization is complete before other threads
                   // proceed

  // Test initialization (verify all bits are zero)
  if (threadIdx.x == 0) {
    for (int i = 0; i < SharedBloomFilter::NUM_WORDS; ++i) {
      assert(bloomFilter.data[i] == 0);
    }
  }
  __syncthreads();

  // Define test keys to be added to the Bloom filter
  int keysToTest[] = {0, 1, 123456, 1048576, 2147483647, -1};
  for (int i = 0; i < 6; ++i) {
    if (threadIdx.x == 0) {
      bloomFilter.set(keysToTest[i]);
    }
    __syncthreads(); // Ensure all threads are synchronized after setting a key
  }

  // Verify keys are present in the Bloom filter
  for (int i = 0; i < 6; ++i) {
    if (threadIdx.x == 0) {
      assert(bloomFilter.test(keysToTest[i]) == true);
    }
    __syncthreads();
  }

  // Define non-existent keys to test absence in the Bloom filter
  int nonExistentKeys[] = {999, 5000, 999999, -9999};
  for (int i = 0; i < 4; ++i) {
    if (threadIdx.x == 0) {
      assert(bloomFilter.test(nonExistentKeys[i]) == false);
    }
    __syncthreads();
  }

  // Test edge cases
  if (threadIdx.x == 0) {
    unsigned int largeKey = 4294967295u; // Maximum value for an unsigned int
    bloomFilter.set(
        static_cast<int>(largeKey)); // Explicitly cast to signed int
    assert(bloomFilter.test(static_cast<int>(largeKey)) == true);

    // Negative numbers
    bloomFilter.set(-12345);
    assert(bloomFilter.test(-12345) == true);

    // Simulated hash collisions
    int collisionKey1 = 42;
    int collisionKey2 = 42 + SharedBloomFilter::NUM_BITS; // Deliberate overlap
    bloomFilter.set(collisionKey1);
    assert(bloomFilter.test(collisionKey2) ==
           true); // Likely true due to overlap
  }
}

int main() {
  printf("Starting Bloom Filter Tests\n");

  // Launch the kernel
  testBloomFilter<<<1, 32>>>();

  // Wait for GPU to finish and check for errors
  cudaDeviceSynchronize();

  // Check for CUDA errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
    return -1;
  }

  printf("All tests passed!\n");
  return 0;
}
