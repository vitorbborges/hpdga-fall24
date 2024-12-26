#include <hnsw.cuh>
#include <utils.cuh>
#include "search_layer.cuh"

using namespace utils;
using namespace hnsw;

int main() {
    std::cout << "GPU" << std::endl;

    const string base_dir = "../";

    int k = 100;
    int m = 16;
    int ef_construction = 100;
    // int ef = 100;

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

    int test_k = 5;
    int test_layer = 2;

    auto search_results = search_layer_launch(
        queries,
        index.enter_node_id,
        test_k,
        index.layers,
        test_layer,
        dataset.size(),
        dataset
    );

    cout << "cuda results: " << endl;

    for (SearchResult sr: search_results.results) {
        for (Neighbor n: sr.result) {
            std::cout << "(" << n.dist << ", " << n.id << ") ";
        }
        std::cout << std::endl;
    }

    cout << "cpu results: " << endl;

    for (size_t i = 0; i < n_query; i++) {
        auto cpu_result = index.search_layer(
            queries[i],
            index.enter_node_id,
            test_k,
            test_layer
        );

        for (size_t i = 0; i < test_k; i++) {
            cout << "(" << cpu_result.result[i].dist << ", " << cpu_result.result[i].id << ") ";
        }
        cout << endl;
    }

    // // check if results are the same
    // int query_id = 0;
    // for (SearchResult sr: search_results.results) {
    //     auto cpu_result = index.search_layer(
    //         queries[query_id],
    //         index.enter_node_id,
    //         test_k,
    //         test_layer
    //     );

    //     for (size_t i = 0; i < test_k; i++) {
    //         for (Neighbor n: sr.result) {
    //             if (n.dist != cpu_result.result[i].dist || n.id != cpu_result.result[i].id) {
    //                 cout << "Results are not the same on query " << query_id << endl;
    //                 return 1;
    //             }
    //         }
    //     }
    //     query_id++;
    // }

    
}