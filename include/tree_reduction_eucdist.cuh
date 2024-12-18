#ifndef EUCLIDEAN_DISTANCE_TREE_CUH
#define EUCLIDEAN_DISTANCE_TREE_CUH

#include "euclidean_distance_CUDA_ABC.cuh"

using namespace std;

template <typename T = float>
class EuclideanDistanceTree : public EuclideanDistanceCUDA<T> {
public:
    // Constructor and Destructor
    EuclideanDistanceTree(
        const vector<vector<T>>& v1,
        const vector<vector<T>>& v2,
        size_t block_size
    );

    std::vector<std::vector<T>> compute() override;

    T* get_h_matrix1() {
        return this->h_matrix1;
    }

    T* get_h_matrix2() {
        return this->h_matrix2;
    }
};


#endif // EUCLIDEAN_DISTANCE_TREE_CUH
