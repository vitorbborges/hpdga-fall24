#include <hnsw.cuh>
#include <utils.cuh>
#include "search_layer2.cuh"

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

    auto result_layer = search_layer_launch(
        queries[0],
        index.enter_node_id,
        5,
        index.layers,
        2,
        dataset.size(),
        dataset
    );

    auto result = index.search_layer(
        queries[0],
        index.enter_node_id,
        5,
        2
    );
}