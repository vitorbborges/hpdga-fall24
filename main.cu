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
    int test_layer = 0;
    Dataset<float> test_queries;
    // test_queries.push_back(queries[85]);
    int test_n_query = test_queries.size();

    auto search_results = search_layer_launch(
        test_queries,
        index.enter_node_id,
        test_k,
        index.layers,
        test_layer,
        dataset.size(),
        dataset
    );

    cout << "cuda results: " << endl;

    search_results.print_results();

    cout << "cpu results: " << endl;

    SearchResults cpu_results(test_n_query);
    for (size_t i = 0; i < test_n_query; i++) {
        auto cpu_result = index.search_layer(
            test_queries[i],
            index.enter_node_id,
            test_k,
            test_layer
        );

        SearchResult sr;
        for (size_t j = 0; j < test_k; j++) {
            Neighbor n;
            n.id = cpu_result.result[j].id;
            n.dist = cpu_result.result[j].dist;
            sr.result.push_back(n);
        }
        cpu_results[i] = sr;
    }

    cpu_results.print_results();

    // Check if results are the same using precomputed CPU results
    bool mismatch_found = false;
    for (size_t i = 0; i < test_n_query; i++) {
        for (size_t j = 0; j < test_k; j++) {
            if (search_results[i].result[j].id != cpu_results[i].result[j].id
                || search_results[i].result[j].dist != cpu_results[i].result[j].dist) {
                cout << "Mismatch found at query_id: " << i << " resuslt_id: " << j << endl;
                cout << "GPU: " << search_results[i].result[j].id << " " << search_results[i].result[j].dist << endl;
                cout << "CPU: " << cpu_results[i].result[j].id << " " << cpu_results[i].result[j].dist << endl;
                cout << "--------------------------------" << endl;
                mismatch_found = true;
            }
        }
    }

    if (!mismatch_found) {
        cout << "Results are the same" << endl;
    }
    return 0;


    
}