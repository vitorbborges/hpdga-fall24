#include "search_layer.cuh"
#include <hnsw.cuh>
#include <utils.cuh>

using namespace utils;
using namespace hnsw;

int main() {
  std::cout << "GPU" << std::endl;

  const string base_dir = "../";

  int k = 100;
  const int m = 16;
  int ef_construction = 100;

  // SIFT10k (small) - 10,000 base / 100 query / 128 dim
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

  const auto dataset = fvecs_read(data_path, n);
  const auto queries = fvecs_read(query_path, n_query);
  const auto ground_truth = load_ivec(ground_truth_path, n_query, k);

  cout << "Start building index" << endl;
  const auto start = get_now();
  auto index = HNSW(m, ef_construction);
  index.build(dataset);
  const auto end = get_now();
  const auto build_time = get_duration(start, end);
  cout << "index_construction: " << build_time / 1000 << " [ms]" << endl;

  const string result_base_dir = base_dir + "results/";
  const string times_path = result_base_dir + "search_times.csv";
  const string result_data_dir = result_base_dir + "data/";

  std::ofstream times_ofs(times_path);
  if (!times_ofs) {
    throw std::ios_base::failure("Failed to open times file");
  }
  times_ofs << "save_name,avg_recall,search_duration,total_duration\n";

  ////////////////////////////////////////////////////////////////////////////////////////////////////////

  // Warmup Kernel
  cudaStream_t warmupStream;
  cudaStreamCreate(&warmupStream);
  warmup(warmupStream);

  ////////////////////////////////////////////////////////////////////////////////////////////////////////

  int test_layer = 1;

  for (int test_n_query = 1; test_n_query <= 100; test_n_query += 1) {
    Dataset<float> test_queries;
    for (int i = 0; i < test_n_query; i++) {
      test_queries.push_back(queries[i]);
    }
    for (int test_k = 100; test_k <= 100; test_k += 1) {
      // Measure CPU time for search
      ExperimentParams cpu_params(test_k, test_n_query, "cpu");

      cpu_params.start_calc = get_now();
      SearchResults results_cpu(test_n_query);
      for (size_t i = 0; i < test_n_query; i++) {
        auto cpu_result = index.search_layer(
            test_queries[i], index.enter_node_id, test_k, test_layer);

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
      cpu_params.end_calc = get_now();
      results_cpu.params = cpu_params;
      results_cpu.save(result_data_dir, times_ofs);

      // results_cpu.print_metrics();

      ////////////////////////////////////////////////////////////////////////////////////////////////////////

      // GPU search variations
      std::vector<std::string> save_names = {"gpu_non_opt", "gpu_shared_mem",
                                             "gpu_eucdist_opt", "gpu_bloom_pq"};
      std::vector<SearchResults> results(save_names.size());

      for (const auto &save_name : save_names) {
        SearchResults search_result = search_layer_launch(
            test_queries, index.enter_node_id, test_k, index.layers[test_layer],
            index.layer_map.at(test_layer), dataset.size(), dataset, save_name);

        cudaDeviceSynchronize();

        // Calculate recall
        for (int i = 0; i < test_n_query; i++) {
          search_result[i].recall =
              calc_recall(search_result[i].result, ground_truth[i], k);
        };

        search_result.save(result_data_dir, times_ofs);

        // search_result.print_metrics();
      }

      //////////////////////////////////////////////////////////////////////////////////////////////////////////

      // Search Layer using SMMH

      SearchResults results_gpu_smmh = search_layer_bloom_smmh_launch(
          test_queries, index.enter_node_id, test_k, index.layers[test_layer],
          index.layer_map.at(test_layer), dataset.size(), dataset);

      cudaDeviceSynchronize();

      results_gpu_smmh.params.experiment_name = "gpu_bloom_smmh";

      // Calculate recall
      for (int i = 0; i < test_n_query; i++) {
        results_gpu_smmh[i].recall =
            calc_recall(results_gpu_smmh[i].result, ground_truth[i], k);
      };

      results_gpu_smmh.save(result_data_dir, times_ofs);

      // results_gpu_smmh.print_metrics();
    }
  }
  return 0;
}
