#ifndef HNSW_UTILS_HPP
#define HNSW_UTILS_HPP

#include "data_structures.cuh"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>

using namespace std;
using namespace ds;

// Macro to check CUDA calls and handle errors
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at "         \
                << __FILE__ << ":" << __LINE__ << " in " << #call              \
                << std::endl;                                                  \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace utils {

// Type alias for a distance function
template <typename T = float>
using DistanceFunction = function<float(Data<T>, Data<T>)>;

// Euclidean distance calculation between two data points
template <typename T = float>
auto euclidean_distance(const Data<T> &p1, const Data<T> &p2) {
  float result = 0;
  for (size_t i = 0; i < p1.size(); i++) {
    result += std::pow(p1[i] - p2[i], 2);
  }
  result = std::sqrt(result);
  return result;
}

// Read fvecs file and construct a dataset
Dataset<float> fvecs_read(const std::string &filename, int size) {
  std::ifstream file(filename, std::ios::binary);

  if (!file.is_open()) {
    throw std::runtime_error("I/O error: Unable to open the file " + filename);
  }

  // Read the vector size (dimension)
  int d;
  file.read(reinterpret_cast<char *>(&d), sizeof(int));
  int vecsizeof = sizeof(int) + d * sizeof(float);

  // Calculate number of vectors in the file
  file.seekg(0, std::ios::end);
  int bmax = file.tellg() / vecsizeof;
  int b = size;

  if (b > bmax) {
    b = bmax;
  }
  if (b == 0) {
    return {};
  }

  int n = b;
  file.seekg(0, std::ios::beg);

  // Read the vectors
  std::vector<float> buffer((d + 1) * n);
  file.read(reinterpret_cast<char *>(buffer.data()),
            (d + 1) * n * sizeof(float));

  // Reshape and store in dataset
  Dataset<float> dataset;
  for (int i = 0; i < n; ++i) {
    float values[d];
    for (int j = 0; j < d; ++j) {
      values[j] = buffer[i * (d + 1) + j + 1]; // Skip dimension value
    }
    dataset.push_back(Data<float>(i, values));
  }

  file.close();
  return dataset;
}

// Get current time point
auto get_now() { return chrono::system_clock::now(); }

// Calculate duration in microseconds
auto get_duration(chrono::system_clock::time_point start,
                  chrono::system_clock::time_point end) {
  return chrono::duration_cast<chrono::microseconds>(end - start).count();
}

// Brute-force k-NN search
template <typename T>
auto scan_knn_search(const Data<T> &query, int k, const Dataset<T> &dataset) {
  const auto df = euclidean_distance<float>;
  auto threshold = float_max;

  multimap<float, int> result_map; // Sorted map to store neighbors
  for (const auto &data : dataset) {
    const auto dist = df(query, data);

    if (result_map.size() < k || dist < threshold) {
      result_map.emplace(dist, data.id);
      threshold = (--result_map.cend())->first;
      if (result_map.size() > k)
        result_map.erase(--result_map.cend());
    }
  }

  vector<Neighbor> result;
  for (const auto &result_pair : result_map) {
    result.emplace_back(result_pair.first, result_pair.second);
  }

  return result;
}

// Calculate recall as a fraction of correct neighbors found
auto calc_recall(const Neighbors &actual, const Neighbors &expect, int k) {
  float recall = 0;

  for (int i = 0; i < k; ++i) {
    const auto n1 = actual[i];
    int match = 0;
    for (int j = 0; j < k; ++j) {
      const auto n2 = expect[j];
      if (n1.id != n2.id)
        continue;
      match = 1;
      break;
    }
    recall += match;
  }

  recall /= actual.size();
  return recall;
}

// Load neighbors from .ivec file
vector<Neighbors> load_ivec(const string &neighbor_path, int n, int k) {
  ifstream ifs(neighbor_path, ios::binary); // Open in binary mode
  if (!ifs)
    throw runtime_error("Can't open file: " + neighbor_path);

  vector<Neighbors> neighbors_list(n); // Vector of Neighbors for each head_id

  // Read neighbors for each query
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < k; ++j) {
      int neigh;
      ifs.read(reinterpret_cast<char *>(&neigh),
               sizeof(int));                     // Read neighbor ID
      neighbors_list[i].emplace_back(42, neigh); // Dummy distance
    }
  }

  return neighbors_list;
}

// Dummy CUDA kernel for warmup
__global__ void warmup_kernel() {}

// Initialize CUDA resources by running a dummy kernel
void warmup(cudaStream_t stream) {
  warmup_kernel<<<1, 1, 0, stream>>>();
  cudaDeviceSynchronize();
}
} // namespace utils

#endif // HNSW_UTILS_HPP
