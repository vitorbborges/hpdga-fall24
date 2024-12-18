#ifndef EUCLIDEAN_DISTANCE_ABC_CUH
#define EUCLIDEAN_DISTANCE_ABC_CUH

// Abstract base class for Euclidean distance computation

#include <vector>
#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>

template <typename T = float>
class EuclideanDistance {
protected:
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    std::vector<std::vector<T>> vec1;
    std::vector<std::vector<T>> vec2;

public:
    EuclideanDistance() = default;
    EuclideanDistance(
        const std::vector<std::vector<T>>& v1,
        const std::vector<std::vector<T>>& v2
    ) : vec1(v1), vec2(v2) {}

    virtual ~EuclideanDistance() = default;

    virtual std::vector<std::vector<T>> compute() = 0;

    auto get_duration() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
};

#endif // EUCLIDEAN_DISTANCE_ABC_CUH