#include "cpu_eucdist.cuh"

// Explicit template instantiation
template class EuclideanDistanceCPU<float>;

template <typename T>
EuclideanDistanceCPU<T>::EuclideanDistanceCPU(
    const std::vector<std::vector<T>>& v1,
    const std::vector<std::vector<T>>& v2
) : EuclideanDistance<T>(v1, v2) {}

template <typename T>
std::vector<std::vector<T>> EuclideanDistanceCPU<T>::compute() {
    this->start = std::chrono::high_resolution_clock::now();
    int num_vectors1 = get_vec1().size();
    int num_vectors2 = get_vec2().size();
    int dimensions = get_vec1()[0].size();
    std::vector<std::vector<T>> distances;

    for (int i = 0; i < num_vectors1; ++i) {
        std::vector<T> row;
        for (int j = 0; j < num_vectors2; ++j) {
            T dist = 0.0;
            for (int k = 0; k < dimensions; ++k) {
                float diff = get_vec1()[i][k] - get_vec2()[j][k];
                dist += diff * diff;
            }
            row.push_back(sqrtf(dist));
        }
        distances.push_back(row);
    }

    this->end = std::chrono::high_resolution_clock::now();

    return distances;
}