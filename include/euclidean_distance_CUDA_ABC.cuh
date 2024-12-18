#include "euclidean_distance_ABC.cuh"

template <typename T = float>
class EuclideanDistanceCUDA : public EuclideanDistance<T> {
protected:
    std::vector<std::vector<T>> vec1, vec2;

    const dim3 blockSize;
    dim3 gridDim;

    // host data
    T* h_matrix1;
    T* h_matrix2;
    T* h_distances;
    size_t matrix1_n, matrix2_n, dimension;

    // device data
    T* d_matrix1;
    T* d_matrix2;
    T* d_distances;
    size_t* d_matrix1_n;
    size_t* d_matrix2_n;
    size_t* d_dimension;
    
public:
    EuclideanDistanceCUDA(
        const std::vector<std::vector<T>>& v1,
        const std::vector<std::vector<T>>& v2,
        size_t& block_size
    ) : vec1(v1), vec2(v2), blockSize(block_size, 1, 1) {
        dimension = vec1[0].size();
        matrix1_n = vec1.size();
        matrix2_n = vec2.size();

        h_matrix1 = new T[matrix1_n * dimension];
        for (size_t i = 0; i < matrix1_n; ++i) {
            for (size_t j = 0; j < dimension; ++j) {
                h_matrix1[i * dimension + j] = vec1[i][j];
            }
        }

        h_matrix2 = new T[matrix2_n * dimension];
        for (size_t i = 0; i < matrix2_n; ++i) {
            for (size_t j = 0; j < dimension; ++j) {
                h_matrix2[i * dimension + j] = vec2[i][j];
            }
        }

        h_distances = new T[matrix1_n * matrix2_n];
        for (size_t i = 0; i < matrix1_n; ++i) {
            for (size_t j = 0; j < matrix2_n; ++j) {
                h_distances[i * matrix2_n + j] = 0.0;
            }
        }

        // Calculate grid dimensions
        gridDim = dim3(
            (dimension + blockSize.x - 1) / blockSize.x,
            matrix1_n,
            matrix2_n
        );
    }

    // Destructor to free dynamically allocated memory
    ~EuclideanDistanceCUDA() {  
        delete[] h_matrix1;
        delete[] h_matrix2;
        delete[] h_distances;
    }
};