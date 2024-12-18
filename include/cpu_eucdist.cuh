#ifndef EUCLIDEAN_DISTANCE_CPU_CUH
#define EUCLIDEAN_DISTANCE_CPU_CUH

#include "euclidean_distance_ABC.cuh"

template <typename T = float>
class EuclideanDistanceCPU : public EuclideanDistance<T> {
public:
    EuclideanDistanceCPU(
        const std::vector<std::vector<T>>& v1,
        const std::vector<std::vector<T>>& v2
    );

    ~EuclideanDistanceCPU() override = default;

    std::vector<std::vector<T>> compute() override;

    std::vector<std::vector<T>> get_vec1() {
        return this->vec1;
    }

    std::vector<std::vector<T>> get_vec2() {
        return this->vec2;
    }
};

#endif // EUCLIDEAN_DISTANCE_CPU_CUH