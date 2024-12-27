#include <hnsw.cuh>
#include <utils.cuh>

#include "knn_search.cuh"

#define REPETITIONS 1

using namespace utils;
using namespace hnsw;

int main() {
    std::cout << "GPU" << std::endl;

    const string base_dir = "../";

    int k = 100;
    int m = 16;
    int ef_construction = 100;
    int ef = 100;

    // SIFT10k (small) - 10,000	base / 100 query / 128 dim
    const string data_path = base_dir + "datasets/siftsmall/siftsmall_base.fvecs";
    const string query_path =
        base_dir + "datasets/siftsmall/siftsmall_query.fvecs";
    const string ground_truth_path =
        base_dir + "datasets/siftsmall/siftsmall_groundtruth.ivecs";
    const int n = 10000, n_query = 100;

    // // SIFT1M (normal) - 1,000,000	base / 10,000 query / 128 dim
    // const string data_path = base_dir + "datasets/sift/sift_base.fvecs";
    // const string query_path = base_dir + "datasets/sift/sift_query.fvecs";
    // const string ground_truth_path = base_dir +
    // "datasets/sift/sift_groundtruth.ivecs"; const int n = 1000000, n_query =
    // 10000;


    cout << "Start loading data" << endl;
    const auto dataset = fvecs_read(data_path, n);
    const auto queries = fvecs_read(query_path, n_query);
    const auto ground_truth = load_ivec(ground_truth_path, n_query, k);
    cout << "Data loaded" << endl;

    cout << "Start building index" << endl;
    const auto start = get_now();
    auto index = HNSW(m, ef_construction);
    index.build(dataset);
    const auto end = get_now();
    const auto build_time = get_duration(start, end);
    cout << "index_construction: " << build_time / 1000 << " [ms]" << endl;

    // CPU Calculation
    long total_time_cpu = 0;
    SearchResults results(n_query);
    // Simulating REPETITIONS * n_query search procedure (during dev you can
    // reduce REPETITIONS = 1 for fast testing)
    cout << "Start searching CPU" << endl;
    for (int rep = 0; rep < REPETITIONS; rep++) {
        for (int i = 0; i < n_query; i++) {
        const auto& query = queries[i];

        auto q_start = get_now();
        auto result = index.knn_search(query, k, ef);
        auto q_end = get_now();
        total_time_cpu += get_duration(q_start, q_end);

        result.recall = calc_recall(result.result, ground_truth[query.id()], k);
        results[i] = result;
        }
    }
    cout << "time for " << REPETITIONS * n_query
        << " queries: " << total_time_cpu / 1000 << " [ms]" << endl;

    for (SearchResult& result : results.results) {
        for (Neighbor& neighbor : result.result) {
            cout << "(" << neighbor.id << ", " << neighbor.dist << ")";
        }
        cout << endl;

    }
        
    // GPU Calculation
    long total_time_gpu = 0;
    cout << "Start searching GPU" << endl;

    auto q_start = get_now();
    SearchResults results_gpu = knn_search(
        queries,
        index.enter_node_id,
        1,
        index.layers,
        n,
        dataset
    );
    auto q_end = get_now();
    total_time_gpu += get_duration(q_start, q_end);
    cout << "time for " << REPETITIONS * n_query
        << " queries: " << total_time_gpu / 1000 << " [ms]" << endl;

    // for (SearchResult& result : results_gpu.results) {
    //     for (Neighbor& neighbor : result.result) {
    //         cout << "(" << neighbor.id << ", " << neighbor.dist << ")" << endl;
    //     }

    // }
    
}