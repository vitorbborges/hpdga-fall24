#ifndef HNSW_DATA_STRUCTURES_CUH
#define HNSW_DATA_STRUCTURES_CUH

#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <iostream>

using namespace std;

namespace ds {
    template <typename T = float>
    class Data {
    private:
        size_t unique_id;
        size_t dim; // TODO: remove dim from this class to save memory.
        T* x;
    public:
        // Constructor with a pointer to an array and size
        Data(size_t i, T* arr, size_t d) : unique_id(i), dim(d) {
            x = new T[dim];
            std::copy(arr, arr + dim, x); // Copy array into dynamic memory
        }

        // Destructor
        ~Data() {
            delete[] x;
        }

        // Copy constructor
        Data(const Data& other) : unique_id(other.unique_id), dim(other.dim) {
            x = new T[dim];
            std::copy(other.x, other.x + dim, x);
        }

        // Copy assignment operator
        Data& operator=(const Data& other) {
            if (this == &other) return *this; // Self-assignment check
            
            delete[] x; // Free existing memory
            dim = other.dim;
            x = new T[dim];
            std::copy(other.x, other.x + dim, x);
            return *this;
        }

        // Move constructor
        Data(Data&& other) noexcept : unique_id(other.unique_id), dim(other.dim), x(other.x) {
            other.x = nullptr;  // Null out the original pointer
        }

        // Move assignment operator
        Data& operator=(Data&& other) noexcept {
            if (this == &other) return *this;
            delete[] x;  // Free existing memory
            unique_id = other.unique_id;
            dim = other.dim;
            x = other.x;
            other.x = nullptr;  // Null out the original pointer
            return *this;
        }

        // Access operator (non-const)
        T& operator[](size_t i) {
            return x[i];
        }

        // Access operator (const)
        const T& operator[](size_t i) const {
            return x[i];
        }

        // Equality operator
        bool operator==(const Data& o) const {
            if (unique_id != o.unique_id) return false;
            if (dim != o.dim) return false;
            for (size_t i = 0; i < dim; ++i) {
                if (x[i] != o.x[i]) return false;
            }
            return true;
        }
        // Inequality operator
        bool operator!=(const Data& o) const {
            return !(*this == o);
        }
        // Size function
        size_t size() const {
            return dim;
        }
        // Getter for the ID
        size_t id() const {
            return unique_id;
        }
        // Getter for the pointer to the underlying array
        T* data() {
            return x;
        }
        // Getter for the pointer to the underlying array (const)
        const T* data() const {
            return x;
        }
    };
    
    template <typename T = float>
    using Dataset = vector<Data<T>>;

    constexpr auto double_max = numeric_limits<double>::max();
    constexpr auto double_min = numeric_limits<double>::min();

    constexpr auto float_max = numeric_limits<float>::max();
    constexpr auto float_min = numeric_limits<float>::min();

    struct Neighbor {
        float dist;
        int id;

        Neighbor() : dist(float_max), id(-1) {}
        Neighbor(float dist, int id) : dist(dist), id(id) {}
    };

    using Neighbors = vector<Neighbor>;

    struct CompLess {
        constexpr bool operator()(const Neighbor& n1, const Neighbor& n2) const noexcept {
            return n1.dist < n2.dist;
        }
    };

    struct CompGreater {
        constexpr bool operator()(const Neighbor& n1, const Neighbor& n2) const noexcept {
            return n1.dist > n2.dist;
        }
    };

    struct Node {
        Data<> data;
        Neighbors neighbors; // TODO: implement this as an array for search stage

        explicit Node(const Data<>& data_) : data(data_) {}
    };

    using Layer = vector<Node>;

    struct SearchResult {
        std::vector<Neighbor> result;
        float recall = 0.0f; // Initialize with a default value
    
        void add_neighbor(float dist, int id) {
            result.emplace_back(dist, id);
        }
    };

    struct SearchResults {
        std::vector<SearchResult> results;
    
        explicit SearchResults(size_t size) : results(size) {}
    
        void push_back(const SearchResult& result) {
            results.emplace_back(result);
        }
    
        void push_back(SearchResult&& result) {
            results.emplace_back(std::move(result));
        }
    
        const SearchResult& operator[](size_t i) const {
            if (i >= results.size()) {
                throw std::out_of_range("Index out of bounds");
            }
            return results[i];
        }
    
        SearchResult& operator[](size_t i) {
            if (i >= results.size()) {
                throw std::out_of_range("Index out of bounds");
            }
            return results[i];
        }
    
        void save(const std::string& log_path, const std::string& result_path) const {
            std::ofstream log_ofs(log_path);
            if (!log_ofs) {
                throw std::ios_base::failure("Failed to open log file");
            }
            log_ofs << "query_id,recall\n";
    
            std::ofstream result_ofs(result_path);
            if (!result_ofs) {
                throw std::ios_base::failure("Failed to open result file");
            }
            result_ofs << "query_id,data_id,dist\n";
    
            int query_id = 0;
            for (const auto& result : results) {
                log_ofs << query_id << "," << result.recall << "\n";
                for (const auto& neighbor : result.result) {
                    result_ofs << query_id << "," << neighbor.id << "," << neighbor.dist << "\n";
                }
                ++query_id;
            }
        }

        void print_results() const {
            int query_id = 0;
            for (const auto& result : results) {
                // std::cout << "Query ID " << query_id << ": ";
                for (const auto& neighbor : result.result) {
                    // std::cout << "(" << neighbor.dist << ", " << neighbor.id << ") ";
                }
                // std::cout << "\n";
                ++query_id;
            }
        }
    };

    template <typename T = float>
    struct d_Neighbor {
        T dist;
        int id;

        __host__ __device__ d_Neighbor() : dist(0), id(-1) {}

        __host__ __device__ d_Neighbor(T dist, int id) : dist(dist), id(id) {}

        // Explicit copy constructor for CUDA
        __host__ __device__ d_Neighbor(const d_Neighbor<T>& other) : dist(other.dist), id(other.id) {}

        // Comparison operators
        __host__ __device__ bool operator<(const d_Neighbor& other) const {
            return dist < other.dist; // Max-heap based on distance
        }
        __host__ __device__ bool operator>(const d_Neighbor& other) const {
            return dist > other.dist; // Min-heap based on distance
        }

        // Copy assignment operator
        __host__ __device__ d_Neighbor<T>& operator=(const d_Neighbor<T>& other) {
            if (this != &other) {
                dist = other.dist;
                id = other.id;
            }
            return *this;
        }

        // Move assignment operator
        __host__ __device__ d_Neighbor<T>& operator=(d_Neighbor<T>&& other) noexcept {
            if (this != &other) {
                dist = std::move(other.dist);
                id = std::move(other.id);
            }
            return *this;
        }

        // Explicit move constructor
        __host__ __device__ d_Neighbor(d_Neighbor<T>&& other) noexcept : dist(std::move(other.dist)), id(std::move(other.id)) {}
    };

}

#endif // HNSW_DATA_STRUCTURES_CUH