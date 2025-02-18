#include <hnsw.cuh>
#include <utils.cuh>
#include "knn_search.cuh"

#define REPETITIONS 100

using namespace utils;
using namespace hnsw;

int main() {
    std::cout << "GPU" << std::endl;

    const string base_dir = "../";

    int k = 100;
    int m = 16;
    int ef_construction = 100;
    int ef = 100;

    // SIFT10k (small) - 10,000 base / 100 query / 128 dim
    const string data_path = base_dir + "datasets/siftsmall/siftsmall_base.fvecs";
    const string query_path = base_dir + "datasets/siftsmall/siftsmall_query.fvecs";
    const string ground_truth_path = base_dir + "datasets/siftsmall/siftsmall_groundtruth.ivecs";
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

    // Test parameters
    Dataset<float> test_queries = queries;
    int test_n_query = test_queries.size();
    int test_ef = 30;

    long total_time_cpu = 0, total_time_gpu = 0;

    // CPU Calculation
    cout << "Start searching CPU" << endl;
    SearchResults results_cpu(n_query);

    for (int rep = 0; rep < REPETITIONS; rep++) {
        for (int i = 0; i < test_n_query; i++) {
            const auto& query = queries[i];

            auto q_start = get_now();
            auto result = index.knn_search(query, k, test_ef);
            auto q_end = get_now();
            total_time_cpu += get_duration(q_start, q_end);

            result.recall = calc_recall(result.result, ground_truth[query.id()], k);
            results_cpu[i] = result;
        }
    }

    results_cpu.print_results();

    // GPU Calculation
    cout << "Start searching GPU" << endl;
    SearchResults results_gpu(n_query);

    for (int rep = 0; rep < REPETITIONS; rep++) {
        std::chrono::system_clock::time_point q_start;
        std::chrono::system_clock::time_point q_end;

        results_gpu = knn_search(
            test_queries,
            index.enter_node_id,
            test_ef,
            index.layers,
            index.layer_map,
            n,
            dataset,
            q_start,
            q_end
        );
        total_time_gpu += get_duration(q_start, q_end);
    }

    for (int i = 0; i < test_n_query; i++) {
        results_gpu[i].recall = calc_recall(results_gpu[i].result, ground_truth[i], k);
    }

    results_gpu.print_results();

    // Compare Results
    bool mismatch_found = false;
    int mismatch_count = 0;
    for (int i = 0; i < test_n_query; i++) {
        for (int j = 0; j < k; j++) {
            if (results_gpu[i].result[j].id != results_cpu[i].result[j].id ||
                results_gpu[i].result[j].dist != results_cpu[i].result[j].dist) {
                mismatch_found = true;
                mismatch_count++;
            }
        }
    }

    if (!mismatch_found) {
        cout << "Results are the same" << endl;
    } else {
        cout << "Mismatch count: " << mismatch_count << endl;
    }

    // CPU Timing and Recall
    cout << "Time for " << REPETITIONS * n_query << " queries on CPU: " << total_time_cpu / 1000 << " [ms]" << endl;
    float avg_recall_cpu = 0;
    for (int i = 0; i < test_n_query; i++) {
        avg_recall_cpu += results_cpu[i].recall;
    }
    avg_recall_cpu /= test_n_query;
    cout << "Average recall (CPU): " << avg_recall_cpu << endl;

    // GPU Timing and Recall
    cout << "Time for " << REPETITIONS * n_query << " queries on GPU: " << total_time_gpu / 1000 << " [ms]" << endl;
    float avg_recall_gpu = 0;
    for (int i = 0; i < test_n_query; i++) {
        avg_recall_gpu += results_gpu[i].recall;
    }
    avg_recall_gpu /= test_n_query;
    cout << "Average recall (GPU): " << avg_recall_gpu << endl;

    return 0;
}
