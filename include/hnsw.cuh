#ifndef HNSW_HNSW_HPP
#define HNSW_HNSW_HPP

#include <queue>
#include <random>
#include <utils.cuh>

using namespace std;
using namespace utils;

namespace hnsw {
struct HNSW {
  const int m, m_max_0, ef_construction;
  const double m_l;
  const bool extend_candidates, keep_pruned_connections;

  int enter_node_id;
  int enter_node_level;
  vector<Layer> layers;
  map<int, vector<int>> layer_map;
  Dataset<> dataset;
  DistanceFunction<> calc_dist;

  mt19937 engine;
  uniform_real_distribution<double> unif_dist;

  HNSW(int m, int ef_construction = 64, bool extend_candidates = false,
       bool keep_pruned_connections = true)
      : m(m), m_max_0(m * 2), m_l(1 / log(1.0 * m)), enter_node_id(-1),
        enter_node_level(-1), ef_construction(ef_construction),
        extend_candidates(extend_candidates),
        keep_pruned_connections(keep_pruned_connections),
        calc_dist([](const Data<float> &p1, const Data<float> &p2) {
          return utils::euclidean_distance<float>(p1, p2);
        }),
        engine(42), unif_dist(0.0, 1.0) {}

  const Node &get_enter_node() const { return layers.back()[enter_node_id]; }

  int get_new_node_level() {
    return static_cast<int>(-log(unif_dist(engine)) * m_l);
  }

  auto search_layer(const Data<> &query, int start_node_id, int ef, int l_c) {
    auto result = SearchResult();

    vector<bool> visited(dataset.size());
    visited[start_node_id] = true;

    priority_queue<Neighbor, vector<Neighbor>, CompGreater> candidates;
    priority_queue<Neighbor, vector<Neighbor>, CompLess> top_candidates;

    const auto &start_node = layers[l_c][start_node_id];
    const auto dist_from_en = calc_dist(query, start_node.data);

    candidates.emplace(dist_from_en, start_node_id);
    top_candidates.emplace(dist_from_en, start_node_id);

    while (!candidates.empty()) {
      const auto nearest_candidate = candidates.top();
      const auto &nearest_candidate_node = layers[l_c][nearest_candidate.id];
      candidates.pop();
      // cout << "nearest candidate id: " << nearest_candidate.id << endl;

      if (nearest_candidate.dist > top_candidates.top().dist)
        break;

      for (const auto neighbor : nearest_candidate_node.neighbors) {
        if (visited[neighbor.id])
          continue;
        visited[neighbor.id] = true;

        const auto &neighbor_node = layers[l_c][neighbor.id];
        const auto dist_from_neighbor = calc_dist(query, neighbor_node.data);
        // cout << "neighbor id: " << neighbor.id << " dist: " <<
        // dist_from_neighbor << endl;

        if (dist_from_neighbor < top_candidates.top().dist ||
            top_candidates.size() < ef) {
          candidates.emplace(dist_from_neighbor, neighbor.id);
          top_candidates.emplace(dist_from_neighbor, neighbor.id);

          if (top_candidates.size() > ef)
            top_candidates.pop();
        }
      }
    }

    while (!top_candidates.empty()) {
      result.result.emplace_back(top_candidates.top());
      top_candidates.pop();
    }

    reverse(result.result.begin(), result.result.end());

    return result;
  }

  auto select_neighbors_heuristic(const Data<> &query,
                                  vector<Neighbor> initial_candidates,
                                  int n_neighbors, int l_c) {
    const auto &layer = layers[l_c];
    priority_queue<Neighbor, vector<Neighbor>, CompGreater> candidates,
        discarded_candidates;

    vector<bool> added(dataset.size());
    added[query.id()] = true;

    // init candidates
    for (const auto &candidate : initial_candidates) {
      if (added[candidate.id])
        continue;
      added[candidate.id] = true;
      candidates.emplace(candidate);
    }

    if (extend_candidates) {
      for (const auto &candidate : initial_candidates) {
        if (added[candidate.id])
          continue;
        added[candidate.id] = true;

        const auto &candidate_node = layer[candidate.id];
        for (const auto &neighbor : candidate_node.neighbors) {
          const auto &neighbor_node = layer[neighbor.id];
          const auto dist_from_neighbor = calc_dist(query, neighbor_node.data);
          candidates.emplace(dist_from_neighbor, neighbor.id);
        }
      }
    }

    // init neighbors
    vector<Neighbor> neighbors;
    neighbors.emplace_back(candidates.top());
    candidates.pop();

    // select edge
    while (!candidates.empty() && neighbors.size() < n_neighbors) {
      const auto candidate = candidates.top();
      candidates.pop();
      const auto &candidate_node = layer[candidate.id];

      bool good = true;
      for (const auto &neighbor : neighbors) {
        const auto &neighbor_node = layer[neighbor.id];
        const auto dist = calc_dist(candidate_node.data, neighbor_node.data);

        if (dist < candidate.dist) {
          good = false;
          break;
        }
      }

      if (good)
        neighbors.emplace_back(candidate);
      else
        discarded_candidates.emplace(candidate);
    }

    if (keep_pruned_connections) {
      while (!discarded_candidates.empty() && neighbors.size() < n_neighbors) {
        neighbors.emplace_back(discarded_candidates.top());
        discarded_candidates.pop();
      }
    }

    return neighbors;
  }

  void insert(const Data<> &new_data) {
    auto l_new_node =
        get_new_node_level(); // all levels below that will have the node in it
    for (int l_c = l_new_node; l_c >= 0; --l_c)
      layer_map[l_c].emplace_back(
          new_data.id()); // register the ID of that node in the map for all the
                          // lower layers

    // navigate the layers from the start_node_layer down to the layer where the
    // new node needs to be added
    auto start_node_id = enter_node_id;
    for (int l_c = enter_node_level; l_c > l_new_node; --l_c) {
      // get the nearest neighbour node to the new node in the current layer
      const auto nn_layer =
          search_layer(new_data, start_node_id, 1, l_c).result[0];
      start_node_id = nn_layer.id;
    }
    // when the previous cycle ends, we have that start_node_id == entry point
    // node in the layer where we want to add the new node i.e. we have the
    // entry point to the layer of the new node

    // now we iterate from this layer down to layer zero
    for (int l_c = min(enter_node_level, l_new_node); l_c >= 0; --l_c) {
      // compute the NN nodes in the current layer with respect to the node that
      // i want to insert neighbors will be populated with the nearest neighbors
      // nodes (and corresponding distances) to the new_data node the
      // entry_point for computing the NN nodes is start_node_id which is the
      // result of the previous greedy search from the top layer to the current
      // one
      auto neighbors =
          search_layer(new_data, start_node_id, ef_construction, l_c).result;

      // if the NN list is greater than M, filter them out with an heuristic
      if (neighbors.size() > m)
        neighbors = select_neighbors_heuristic(new_data, neighbors, m, l_c);

      auto &layer = layers[l_c];
      // for each neighbor
      for (const auto neighbor : neighbors) {
        if (neighbor.id == new_data.id())
          continue;
        // get the Node class (the list of neighbors contains only IDs)
        auto &neighbor_node = layer[neighbor.id];

        // add a bidirectional edge
        layer[new_data.id()].neighbors.emplace_back(
            neighbor); // link the new node to its NN
        neighbor_node.neighbors.emplace_back(
            neighbor.dist,
            new_data.id()); // add the link from the neighbour to the new node

        const auto m_max =
            l_c ? m : m_max_0; // if we are in layer 0 m_max should be equal to
                               // m_max of layer 0 (which is different from the
                               // normal M)

        // if the neighbor now has more than the allowed maximum of neighbours
        // (we have added new node to its list)
        if (neighbor_node.neighbors.size() > m_max) {
          // compute a restricted set and update the list
          const auto new_neighbor_neighbors = select_neighbors_heuristic(
              neighbor_node.data, neighbor_node.neighbors, m_max, l_c);
          neighbor_node.neighbors = new_neighbor_neighbors;
        }
      }

      if (l_c == 0) // if we arrived at layer 0 we can skip the following step
        break;

      // repeat the operation taking into consideration the nearest neighbour of
      // new_data
      start_node_id = neighbors[0].id;
    }

    // when we finish the previous loop, we have added the new Node to the graph

    // if new node is top (this can either happen at the first element added or
    // if we are adding a new layer)
    if (layers.empty() || l_new_node > enter_node_level) {
      // change enter node
      enter_node_id = new_data.id();

      // add new layer
      layers.resize(l_new_node + 1);
      for (int l_c = max(enter_node_level, 0); l_c <= l_new_node; ++l_c) {
        for (const auto &data : dataset) {
          layers[l_c].emplace_back(data);
        }
      }
      enter_node_level = l_new_node;
    }
  }

  void build(const Dataset<> &dataset_) {
    dataset = dataset_;
    for (const auto &data : dataset) {
      insert(data);
    }
    cout << "Index construction completed." << endl;
  }

  auto knn_search(const Data<> &query, int k, int ef) {
    SearchResult result;
    // search in upper layers
    auto start_id_layer = enter_node_id;
    for (int l_c = enter_node_level; l_c >= 1; --l_c) {
      const auto result_layer = search_layer(query, start_id_layer, 1, l_c);
      const auto &nn_id_layer = result_layer.result[0].id;
      start_id_layer = nn_id_layer;
    }

    const auto &nn_upper_layer = layers[1][start_id_layer];

    // search in base layer
    const auto result_layer = search_layer(query, start_id_layer, ef, 0);
    const auto candidates = result_layer.result;
    for (const auto &candidate : candidates) {
      result.result.emplace_back(candidate);
      if (result.result.size() >= k)
        break;
    }

    return result;
  }
};
} // namespace hnsw

#endif // HNSW_HNSW_HPP
