#include "euclidean_distance.cuh"

int main() {
  const int num_vectors = 100000; // Number of vectors
  const int dimensions = 128;     // Dimensions of each vector

  // Allocate and initialize host memory
  std::vector<float> h_vec1(num_vectors * dimensions);
  std::vector<float> h_vec2(num_vectors * dimensions);
  std::vector<float> h_distances_simple(
      num_vectors); // For the simple GPU kernel
  std::vector<float> h_distances_kepler(
      num_vectors); // For the optimized GPU kernel
  std::vector<float> h_distances_cpu(num_vectors); // For CPU results
  std::vector<float> h_distances_atomic(
      num_vectors); // For the GPU kernel with atomics

  // Initialize random vectors
  for (int i = 0; i < num_vectors * dimensions; ++i) {
    h_vec1[i] = static_cast<float>(rand()) / RAND_MAX;
    h_vec2[i] = static_cast<float>(rand()) / RAND_MAX;
  }

  // CPU computation for baseline comparison
  auto start_cpu = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < num_vectors; ++i) {
    h_distances_cpu[i] = euclidean_distance_cpu<float>(
        &h_vec1[i * dimensions], &h_vec2[i * dimensions], dimensions);
  }
  auto end_cpu = std::chrono::high_resolution_clock::now();

  // Allocate device memory
  float *d_vec1, *d_vec2, *d_distances;
  cudaMalloc(&d_vec1, num_vectors * dimensions * sizeof(float));
  cudaMalloc(&d_vec2, num_vectors * dimensions * sizeof(float));
  cudaMalloc(&d_distances, num_vectors * sizeof(float));

  // Copy data to device
  cudaMemcpy(d_vec1, h_vec1.data(), num_vectors * dimensions * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_vec2, h_vec2.data(), num_vectors * dimensions * sizeof(float),
             cudaMemcpyHostToDevice);

  // Kernel launch configuration
  const int threads_per_block = 256;
  const int blocks_per_grid =
      (num_vectors + threads_per_block - 1) / threads_per_block;

  // Run non-optimized GPU kernel
  auto start_simple = std::chrono::high_resolution_clock::now();
  batch_euclidean_distance<<<blocks_per_grid, threads_per_block>>>(
      d_vec1, d_vec2, d_distances, num_vectors, dimensions);
  cudaDeviceSynchronize();
  auto end_simple = std::chrono::high_resolution_clock::now();

  // Copy results from device
  cudaMemcpy(h_distances_simple.data(), d_distances,
             num_vectors * sizeof(float), cudaMemcpyDeviceToHost);

  // Run optimized GPU kernel
  auto start_kepler = std::chrono::high_resolution_clock::now();
  batch_gpu<<<blocks_per_grid, threads_per_block>>>(d_vec1, d_vec2, d_distances,
                                                    num_vectors, dimensions);
  cudaDeviceSynchronize();
  auto end_kepler = std::chrono::high_resolution_clock::now();

  // Copy results from device
  cudaMemcpy(h_distances_kepler.data(), d_distances,
             num_vectors * sizeof(float), cudaMemcpyDeviceToHost);

  // Run GPU kernel with atomic operations
  auto start_atomic = std::chrono::high_resolution_clock::now();
  batch_euclidean_distance_atomic<<<blocks_per_grid, threads_per_block>>>(
      d_vec1, d_vec2, d_distances, num_vectors, dimensions);
  cudaDeviceSynchronize();
  auto end_atomic = std::chrono::high_resolution_clock::now();

  // Copy results from device
  cudaMemcpy(h_distances_atomic.data(), d_distances,
             num_vectors * sizeof(float), cudaMemcpyDeviceToHost);

  // Print execution times for each method
  std::cout << "CPU computation time: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu -
                                                                    start_cpu)
                   .count()
            << " ns" << std::endl;
  std::cout << "Simple GPU kernel time: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   end_simple - start_simple)
                   .count()
            << " ns" << std::endl;
  std::cout << "Optimized GPU kernel time: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   end_kepler - start_kepler)
                   .count()
            << " ns" << std::endl;
  std::cout << "Optimized GPU with atomic kernel time: "
            << std::chrono::duration_cast<std::chrono::nanoseconds>(
                   end_atomic - start_atomic)
                   .count()
            << " ns" << std::endl;

  // Compare results between CPU and GPU implementations
  bool valid = true;
  for (int i = 0; i < num_vectors; ++i) {
    if (fabs(h_distances_cpu[i] - h_distances_simple[i]) > 1e-5 ||
        fabs(h_distances_cpu[i] - h_distances_kepler[i]) > 1e-5 ||
        fabs(h_distances_cpu[i] - h_distances_atomic[i]) > 1e-5) {
      valid = false;
      break;
    }
  }

  // Output result comparison status
  if (valid) {
    printf("All results match!\n");
  } else {
    printf("Results do not match!\n");
  }

  // Print the first 50 distances for debugging and comparison
  std::cout << "\nFirst 50 results comparison:\n";
  std::cout << "Index\tCPU\tSimple GPU\tOptimized GPU\tAtomic GPU\n";
  for (int i = 0; i < 50; ++i) {
    std::cout << i << "\t" << h_distances_cpu[i] << "\t"
              << h_distances_simple[i] << "\t" << h_distances_kepler[i] << "\t"
              << h_distances_atomic[i] << std::endl;
  }

  // Free allocated device memory
  cudaFree(d_vec1);
  cudaFree(d_vec2);
  cudaFree(d_distances);

  return 0;
}
