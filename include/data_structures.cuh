#ifndef HNSW_DATA_STRUCTURES_CUH
#define HNSW_DATA_STRUCTURES_CUH

#include <vector>

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
        Neighbors result;
        double recall = 0;
    };

    struct SearchResults {
        vector<SearchResult> results;

        SearchResults(size_t size) : results(size) {}
        void push_back(const SearchResult& result) { results.emplace_back(result); }
        void push_back(SearchResult&& result) { results.emplace_back(move(result)); }
        decltype(auto) operator [] (int i) { return results[i]; }

        void save(const string& log_path, const string& result_path) {
            ofstream log_ofs(log_path);
            string line = "query_id,recall";
            log_ofs << line << endl;

            ofstream result_ofs(result_path);
            line = "query_id,data_id,dist";
            result_ofs << line << endl;

            int query_id = 0;
            for (const auto& result : results) {
                log_ofs << query_id << ","<< result.recall << endl;

                for (const auto& neighbor : result.result) {
                    result_ofs << query_id << ","
                               << neighbor.id << ","
                               << neighbor.dist << endl;
                }

                query_id++;
            }
        }
    };
}

#endif // HNSW_DATA_STRUCTURES_CUH