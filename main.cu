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

    // SIFT10k (small) - 10,000 base / 100 query / 128 dim
    const string data_path = base_dir + "datasets/siftsmall/siftsmall_base.fvecs";
    const string query_path = base_dir + "datasets/siftsmall/siftsmall_query.fvecs";
    const string ground_truth_path = base_dir + "datasets/siftsmall/siftsmall_groundtruth.ivecs";
    const int n = 10000, n_query = 100;

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

    int test_k = 4;
    int test_layer = 2;
    Dataset<float> test_queries;
    test_queries.push_back(queries[66]);
    int test_n_query = test_queries.size();

    // Measure GPU time
    std::chrono::system_clock::time_point gpu_start;
    std::chrono::system_clock::time_point gpu_end;
    auto results_gpu = search_layer_launch(
        test_queries,
        index.enter_node_id,
        test_k,
        index.layers[test_layer],
        index.layer_map.at(test_layer),
        dataset.size(),
        dataset,
        gpu_start,
        gpu_end
    );
    const auto gpu_duration = get_duration(gpu_start, gpu_end);
    cout << "GPU search time: " << gpu_duration / 1000 << " [ms]" << endl;

    // Calculate recall
    for (int i = 0; i < test_n_query; i++) {
        results_gpu[i].recall = calc_recall(results_gpu[i].result, ground_truth[i], k);
    }
    
    // print average recall
    float avg_recall_gpu = 0;
    for (int i = 0; i < test_n_query; i++) {
        avg_recall_gpu += results_gpu[i].recall;
    }
    avg_recall_gpu /= test_n_query;
    cout << "Average recall: " << avg_recall_gpu << endl;

    results_gpu.print_results();

    // Measure CPU time
    const auto cpu_start = get_now();
    SearchResults results_cpu(test_n_query);
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
            sr.recall = calc_recall(sr.result, ground_truth[i], k);
        }
        results_cpu[i] = sr;
    }
    const auto cpu_end = get_now();
    const auto cpu_duration = get_duration(cpu_start, cpu_end);
    cout << "CPU search time: " << cpu_duration / 1000 << " [ms]" << endl;

    // print average recall
    float avg_recall_cpu = 0;
    for (int i = 0; i < test_n_query; i++) {
        avg_recall_cpu += results_cpu[i].recall;
    }
    avg_recall_cpu /= test_n_query;
    cout << "Average recall: " << avg_recall_cpu << endl;

    results_cpu.print_results();

    // Check if results are the same using precomputed CPU results
    bool mismatch_found = false;
    for (size_t i = 0; i < test_n_query; i++) {
        for (size_t j = 0; j < test_k; j++) {
            if (results_gpu[i].result[j].id != results_cpu[i].result[j].id
                || results_gpu[i].result[j].dist != results_cpu[i].result[j].dist) {
                mismatch_found = true;
            }
        }
    }

    if (!mismatch_found) {
        cout << "Results are the same" << endl;
    } else {
        cout << "Results are different" << endl;
    }
    return 0;
}
