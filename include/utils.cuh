#ifndef HNSW_UTILS_HPP
#define HNSW_UTILS_HPP

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <cassert>
#include <random>
#include <map>
#include <functional>
#include <string>
#include "data_structures.cuh"

using namespace std;
using namespace ds;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)              \
                      << " at " << __FILE__ << ":" << __LINE__                  \
                      << " in " << #call << std::endl;                          \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

namespace utils {
    template <typename T = float>
    using DistanceFunction = function<float(Data<T>, Data<T>)>;

    template <typename T = float>
    auto euclidean_distance(const Data<T>& p1, const Data<T>& p2) {
        float result = 0;
        for (size_t i = 0; i < p1.size(); i++) {
            result += std::pow(p1[i] - p2[i], 2);
        }
        result = std::sqrt(result);
        return result;
    }
        
    Dataset<float> fvecs_read(const std::string& filename, int size) {
        std::ifstream file(filename, std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("I/O error: Unable to open the file " + filename);
        }

        // Read the vector size
        int d;
        file.read(reinterpret_cast<char*>(&d), sizeof(int));
        int vecsizeof = sizeof(int) + d * sizeof(float);

        // Get the number of vectors
        file.seekg(0, std::ios::end);
        int a = 1;
        int bmax = file.tellg() / vecsizeof;
        int b = bmax;

        b = size;

        assert(a >= 1);
        if (b > bmax) {
            b = bmax;
        }

        if (b == 0 || b < a) {
            return {};
        }

        // Compute the number of vectors that are really read and go to starting positions
        int n = b - a + 1;
        file.seekg((a - 1) * vecsizeof, std::ios::beg);

        // Read n vectors
        std::vector<float> buffer((d + 1) * n);
        file.read(reinterpret_cast<char*>(buffer.data()), (d + 1) * n * sizeof(float));

        // Reshape the vectors
        Dataset<float> dataset;
        for (int i = 0; i < n; ++i) {
            std::vector<float> values(d);  // Use std::vector for dynamic size
            for (int j = 0; j < d; ++j) {
                values[j] = buffer[i * (d + 1) + j + 1]; // Start from 1 to skip the dimension value
            }
            // Pass the vector pointer to the Data constructor
            dataset.push_back(Data<float>(a + i - 1, values.data(), d)); // Index starts from 0
        }

        file.close();

        return dataset;
    }

    auto get_now() { return chrono::system_clock::now(); }

    auto get_duration(chrono::system_clock::time_point start,
                      chrono::system_clock::time_point end) {
        return chrono::duration_cast<chrono::microseconds>(end - start).count();
    }

    template <typename T>
    auto scan_knn_search(const Data<T>& query, int k, const Dataset<T>& dataset) {
        const auto df = euclidean_distance<float>;
        auto threshold = float_max;

        multimap<float, int> result_map;
        for (const auto& data : dataset) {
            const auto dist = df(query, data);

            if (result_map.size() < k || dist < threshold) {
                result_map.emplace(dist, data.id);
                threshold = (--result_map.cend())->first;
                if (result_map.size() > k) result_map.erase(--result_map.cend());
            }
        }

        vector<Neighbor> result;
        for (const auto& result_pair : result_map) {
            result.emplace_back(result_pair.first, result_pair.second);
        }

        return result;
    }

    auto calc_recall(const Neighbors& actual, const Neighbors& expect, int k) {
        float recall = 0;

        for (int i = 0; i < k; ++i) {
            const auto n1 = actual[i];
            int match = 0;
            for (int j = 0; j < k; ++j) {
                const auto n2 = expect[j];
                if (n1.id != n2.id) continue;
                match = 1;
                break;
            }
            recall += match;
        }

        recall /= actual.size();
        return recall;
    }

    /*
        the .ivec is a raw binary file (little endian)
        where you have consecutive values in the following format
        vector0_size/vec0(0)/vec0(1)/vec0(2)/.../vec0(vec_size-1)/vector1_size/vec1(0) ...
        where each value is an int (4 bytes)
        in the case of ground truth, the vectorX_size is the K of the ANNS procedure
        and each value is the index inside the original dataset of the KNN of the queryX
    */
    vector<Neighbors> load_ivec(const string& neighbor_path, int n, int K) {
        ifstream ifs(neighbor_path, ios::binary); // Open in binary mode
        if (!ifs) throw runtime_error("Can't open file: " + neighbor_path);

        vector<Neighbors> neighbors_list(n); // Vector of Neighbors for each head_id

        // Read data
        for (int i = 0; i < n; ++i) {
            int head_id = i;
            // cout << head_id << ": ";
            // Read K elements for each head_id
            for (int j = 0; j < K; ++j) {
                int neigh;
                ifs.read(reinterpret_cast<char*>(&neigh), sizeof(int)); // Read tail_id

                // cout << neigh << " / ";
                neighbors_list[head_id].emplace_back(42, neigh); // Add Neighbor to corresponding head_id
            }
            // cout << endl;
        }

        return neighbors_list;
    }
}

#endif // HNSW_UTILS_HPP
