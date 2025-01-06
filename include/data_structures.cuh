#ifndef HNSW_DATA_STRUCTURES_CUH
#define HNSW_DATA_STRUCTURES_CUH

#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

namespace ds {

// Template class representing a data point with a unique ID
template <typename T = float> class Data {
private:
  size_t unique_id; // Unique identifier for the data point
  T *x;             // Dynamically allocated array for data values

public:
  // Constructor with a pointer to an array and size
  Data(size_t i, T *arr) : unique_id(i) {
    x = new T[VEC_DIM];               // Allocate memory for VEC_DIM elements
    std::copy(arr, arr + VEC_DIM, x); // Copy array into dynamic memory
  }

  // Destructor to release allocated memory
  ~Data() { delete[] x; }

  // Copy constructor for deep copy of the object
  Data(const Data &other) : unique_id(other.unique_id) {
    x = new T[VEC_DIM];
    std::copy(other.x, other.x + VEC_DIM, x);
  }

  // Copy assignment operator for deep copy
  Data &operator=(const Data &other) {
    if (this == &other)
      return *this; // Self-assignment check
    delete[] x;     // Free existing memory
    x = new T[VEC_DIM];
    std::copy(other.x, other.x + VEC_DIM, x);
    return *this;
  }

  // Move constructor for efficient resource transfer
  Data(Data &&other) noexcept : unique_id(other.unique_id), x(other.x) {
    other.x = nullptr; // Nullify the original pointer to avoid double delete
  }

  // Move assignment operator for efficient resource transfer
  Data &operator=(Data &&other) noexcept {
    if (this == &other)
      return *this; // Self-assignment check
    delete[] x;     // Free existing memory
    unique_id = other.unique_id;
    x = other.x;
    other.x = nullptr; // Nullify the original pointer
    return *this;
  }

  // Non-const access operator
  T &operator[](size_t i) { return x[i]; }

  // Const access operator
  const T &operator[](size_t i) const { return x[i]; }

  // Equality operator to compare two Data objects
  bool operator==(const Data &o) const {
    if (unique_id != o.unique_id)
      return false;
    for (size_t i = 0; i < VEC_DIM; ++i) {
      if (x[i] != o.x[i])
        return false;
    }
    return true;
  }

  // Inequality operator
  bool operator!=(const Data &o) const { return !(*this == o); }

  // Return the size of the data (VEC_DIM)
  size_t size() const { return VEC_DIM; }

  // Getter for the unique ID
  size_t id() const { return unique_id; }

  // Get a pointer to the underlying array (non-const)
  T *data() { return x; }

  // Get a pointer to the underlying array (const)
  const T *data() const { return x; }
};

// Alias for a collection of Data objects
template <typename T = float> using Dataset = vector<Data<T>>;

// Numeric limits for double and float values
constexpr auto double_max = numeric_limits<double>::max();
constexpr auto double_min = numeric_limits<double>::min();

constexpr auto float_max = numeric_limits<float>::max();
constexpr auto float_min = numeric_limits<float>::min();

// Struct representing a neighbor with distance and ID
struct Neighbor {
  float dist; // Distance to the neighbor
  int id;     // Identifier of the neighbor

  // Default constructor initializes to maximum distance and invalid ID
  Neighbor() : dist(float_max), id(-1) {}

  // Constructor with specific distance and ID
  Neighbor(float dist, int id) : dist(dist), id(id) {}
};

// Alias for a collection of neighbors
using Neighbors = vector<Neighbor>;

// Comparison structs for sorting neighbors by distance
struct CompLess {
  constexpr bool operator()(const Neighbor &n1,
                            const Neighbor &n2) const noexcept {
    return n1.dist < n2.dist; // Sort in ascending order
  }
};

struct CompGreater {
  constexpr bool operator()(const Neighbor &n1,
                            const Neighbor &n2) const noexcept {
    return n1.dist > n2.dist; // Sort in descending order
  }
};

// Struct representing a graph node with data and neighbors
struct Node {
  Data<> data;         // The data point stored in the node
  Neighbors neighbors; // List of neighbors (can be optimized as an array)

  // Constructor initializes node with data
  explicit Node(const Data<> &data_) : data(data_) {}
};

// Alias for a collection of nodes (a layer in the graph)
using Layer = vector<Node>;

// Struct representing a search result with recall and neighbors
struct SearchResult {
  std::vector<Neighbor> result; // Neighbors in the search result
  float recall = 0.0f;          // Recall value of the search result

  // Add a neighbor to the search result
  void add_neighbor(float dist, int id) { result.emplace_back(dist, id); }
};

// Experiment parameters structure
struct ExperimentParams {
  int k;       // Number of neighbors to search for
  int n_query; // Number of queries to search for
  std::chrono::system_clock::time_point start_alloc;
  std::chrono::system_clock::time_point end_alloc;
  std::chrono::system_clock::time_point start_calc;
  std::chrono::system_clock::time_point end_calc;
  std::string experiment_name; // Name for saving results

  // Default constructor
  ExperimentParams() : k(0), n_query(0), experiment_name("") {}

  // Constructor with specific parameters
  ExperimentParams(int k_, int n_query_, std::string name)
      : k(k_), n_query(n_query_), experiment_name(name) {}

  long long get_duration(chrono::system_clock::time_point start,
                         chrono::system_clock::time_point end) const {
    return chrono::duration_cast<chrono::microseconds>(end - start).count();
  }

  // search time duration
  long long search_duration() const {
    return get_duration(start_calc, end_calc);
  }

  // total time duration
  long long total_duration() const {
    if (start_alloc != chrono::system_clock::time_point()) {
      return get_duration(start_alloc, end_alloc);
    } else {
      return 0;
    }
  }

  // calculate save name
  std::string save_name() const {
    std::string save_name = experiment_name;
    save_name += "-k" + std::to_string(k);
    save_name += "-query" + std::to_string(n_query);
    return save_name;
  }
};

// Struct representing multiple search results
struct SearchResults {
  std::vector<SearchResult> results; // Collection of search results
  ExperimentParams params;           // Experiment parameters

  // Default constructor
  SearchResults() = default;

  // Initialize with a specified number of search results
  explicit SearchResults(size_t size) : results(size) {}

  // Add a new search result
  void push_back(const SearchResult &result) { results.emplace_back(result); }

  // Add a search result using move semantics
  void push_back(SearchResult &&result) {
    results.emplace_back(std::move(result));
  }

  // Access a search result (const)
  const SearchResult &operator[](size_t i) const {
    if (i >= results.size()) {
      throw std::out_of_range("Index out of bounds");
    }
    return results[i];
  }

  // Access a search result (non-const)
  SearchResult &operator[](size_t i) {
    if (i >= results.size()) {
      throw std::out_of_range("Index out of bounds");
    }
    return results[i];
  }

  auto calc_one_recall(const Neighbors &actual, const Neighbors &expect,
                       int k) {
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

  void calculate_recall(const std::vector<Neighbors> &ground_truth) {
    for (size_t i = 0; i < results.size(); ++i) {
      // Use the calc_recall function to calculate the recall for each query
      results[i].recall =
          calc_one_recall(results[i].result, ground_truth[i], params.k);
    }
  }

  // Calculate average recall of the search results
  float get_avg_recall() const {
    float avg_recall = 0.0f;
    for (const auto &result : results) {
      avg_recall += result.recall;
    }
    avg_recall /= results.size();

    return avg_recall;
  }

  // Save search results to log and result files
  void save(const std::string &result_data_dir,
            std::ofstream &times_ofs) const {

    const string log_path =
        result_data_dir + "log-" + params.save_name() + ".csv";
    const string result_path =
        result_data_dir + "result-" + params.save_name() + ".csv";

    times_ofs << params.save_name() << "," << std::to_string(get_avg_recall())
              << "," << params.search_duration() << ","
              << params.total_duration() << "\n";

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
    for (const auto &result : results) {
      log_ofs << query_id << "," << result.recall << "\n";
      for (const auto &neighbor : result.result) {
        result_ofs << query_id << "," << neighbor.id << "," << neighbor.dist
                   << "\n";
      }
      ++query_id;
    }
  }

  // Print search results to the console
  void print_results() const {
    int query_id = 0;
    for (const auto &result : results) {
      std::cout << "Query ID " << query_id << ": ";
      for (const auto &neighbor : result.result) {
        std::cout << "(" << neighbor.dist << ", " << neighbor.id << ") ";
      }
      std::cout << "\n";
      ++query_id;
    }
  }

  // print search parameters to the console
  void print_params() const {
    std::cout << "experiment_name: " << params.experiment_name << "\n";
    std::cout << "k: " << params.k << "\n";
    std::cout << "n_query: " << params.n_query << "\n";
  }

  // Print average recall and search time to the console
  void print_metrics() const {
    print_params();
    std::cout << "Average recall: " << get_avg_recall() << "\n";
    std::cout << "Search time: " << params.search_duration() / 1000
              << " [ms]\n";
    std::cout << "Total time: " << params.total_duration() / 1000 << " [ms]\n";
  }
};

// Device-friendly neighbor structure for CUDA
template <typename T = float> struct d_Neighbor {
  T dist; // Distance to the neighbor
  int id; // Identifier of the neighbor

  // Default constructor
  __host__ __device__ d_Neighbor() : dist(0), id(-1) {}

  // Constructor with specific distance and ID
  __host__ __device__ d_Neighbor(T dist, int id) : dist(dist), id(id) {}

  // Explicit copy constructor for CUDA
  __host__ __device__ d_Neighbor(const d_Neighbor<T> &other)
      : dist(other.dist), id(other.id) {}

  // Comparison operators for heap usage
  __host__ __device__ bool operator<(const d_Neighbor &other) const {
    return dist < other.dist; // Max-heap based on distance
  }
  __host__ __device__ bool operator>(const d_Neighbor &other) const {
    return dist > other.dist; // Min-heap based on distance
  }

  // Copy assignment operator
  __host__ __device__ d_Neighbor<T> &operator=(const d_Neighbor<T> &other) {
    if (this != &other) {
      dist = other.dist;
      id = other.id;
    }
    return *this;
  }

  // Move assignment operator
  __host__ __device__ d_Neighbor<T> &operator=(d_Neighbor<T> &&other) noexcept {
    if (this != &other) {
      dist = std::move(other.dist);
      id = std::move(other.id);
    }
    return *this;
  }

  // Explicit move constructor
  __host__ __device__ d_Neighbor(d_Neighbor<T> &&other) noexcept
      : dist(std::move(other.dist)), id(std::move(other.id)) {}
};

enum HeapType { MIN_HEAP, MAX_HEAP }; // Defines heap type

} // namespace ds

#endif // HNSW_DATA_STRUCTURES_CUH