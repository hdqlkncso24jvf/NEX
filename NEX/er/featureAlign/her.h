#ifndef HER_HER_H_
#define HER_HER_H_
#include <boost/mpi.hpp>
#include <ostream>
#include <queue>
#include <thread>

#include "her/apair_parallel.h"
#include "her/config.h"
#include "her/flags.h"
#include "her/graph_loader.h"
#include "her/inverted_index.h"
#include "her/processing_utils.h"
#include "her/timer.h"
#include "her/vpair.h"
#include "her/incremental.h"

namespace her
{

  int GetParallelism(boost::mpi::communicator &comm)
  {
    int parallelism = FLAGS_parallelism;

    if (parallelism == -1)
    {
      auto local_pnum = local_process_num(comm);

      parallelism =
          (std::thread::hardware_concurrency() + local_pnum - 1) / local_pnum;
    }
    return parallelism;
  }

  template <typename GRAPH_T, typename coord_t>
  void LoadData(
      boost::mpi::communicator &comm, GRAPH_T &gd, GRAPH_T &g, GRAPH_T &g_updated, GRAPH_T &gd_updated,
      std::unordered_map<std::string, dense_vector_t<coord_t>> &word_embeddings,
      std::unordered_set<std::string> &gd_source_labels,
      std::unordered_set<std::string> &g_source_labels,
      std::unordered_map<std::pair<std::string, std::string>, coord_t> &synonym,
      std::vector<std::vector<std::pair<typename GRAPH_T::vertex_t, depth_t>>> &
          g_descendants,
      std::unordered_map<
          typename GRAPH_T::vertex_t,
          std::unordered_map<typename GRAPH_T::vertex_t, std::string>> &g_path)
  {

    using oid_t = typename GRAPH_T::oid_t;
    using vertex_t = typename GRAPH_T::vertex_t;

    GraphLoader<GRAPH_T> loader;
    std::string gd_vfile = FLAGS_gd_vfile;
    std::string gd_efile = FLAGS_gd_efile;
    std::string g_vfile = FLAGS_g_vfile;
    std::string g_efile = FLAGS_g_efile;
    std::string g_updated_efile = FLAGS_g_updated_efile;
    std::string gd_updated_efile = FLAGS_gd_updated_efile;
    std::string word_embedding_file = FLAGS_embedding_file;
    std::string synonym_file = FLAGS_synonym_file;
    std::string gd_slabel_file = FLAGS_gd_slabel_file;
    std::string g_slabel_file = FLAGS_g_slabel_file;
    std::string desc_file = FLAGS_desc_file;
    std::string path_file = FLAGS_path_file;

    if (access(gd_vfile.c_str(), 0) != 0)
    {
      LOG(FATAL) << "Invalid param: -gd_vfile = " << gd_vfile;
    }

    if (access(gd_efile.c_str(), 0) != 0)
    {
      LOG(FATAL) << "Invalid param: -gd_efile = " << gd_efile;
    }
    if (access(g_vfile.c_str(), 0) != 0)
    {
      LOG(FATAL) << "Invalid param: -g_vfile = " << g_vfile;
    }
    if (access(g_efile.c_str(), 0) != 0)
    {
      LOG(FATAL) << "Invalid param: -g_efile = " << g_efile;
    }

    if (FLAGS_query_type == "incremental" || FLAGS_query_type == "batch_spair")
    {
      if (access(g_updated_efile.c_str(), 0) != 0)
      {
        LOG(FATAL) << "Invalid param: -g_updated_efile = " << g_efile;
      }

      if (access(gd_updated_efile.c_str(), 0) != 0)
      {
        LOG(FATAL) << "Invalid param: -gd_updated_efile = " << gd_efile;
      }
    }

    if (access(word_embedding_file.c_str(), 0) != 0)
    {
      LOG(FATAL) << "Invalid param: -word_embedding_file = "
                 << word_embedding_file;
    }
    if (access(gd_slabel_file.c_str(), 0) != 0)
    {
      LOG(FATAL) << "Invalid param: -gd_slabel_file = " << gd_slabel_file;
    }
    if (access(g_slabel_file.c_str(), 0) != 0)
    {
      LOG(FATAL) << "Invalid param: -g_slabel_file = " << g_slabel_file;
    }
    if (!synonym_file.empty() && access(synonym_file.c_str(), 0) != 0)
    {
      LOG(FATAL) << "Invalid param: -synonym_file = " << synonym_file;
    }
    if (!desc_file.empty() && access(desc_file.c_str(), 0) != 0)
    {
      LOG(FATAL) << "Invalid param: -desc_file = " << desc_file;
    }
    if (!path_file.empty() && access(path_file.c_str(), 0) != 0)
    {
      LOG(FATAL) << "Invalid param: -path_file = " << path_file;
    }

    std::thread load_gd_thread(
        [&loader, &gd](const std::string &vfile, const std::string &efile)
        {
          gd = loader.LoadGraph(vfile, efile);

          for (auto v : gd.Vertices())
          {
            boost::to_lower(gd[v]);

            for (auto &e : gd.GetOutgoingAdjList(v))
            {
              boost::to_lower(gd[e]);
            }
          }
        },
        gd_vfile, gd_efile);

    std::thread load_g_thread(
        [&loader, &g](const std::string &vfile, const std::string &efile)
        {
          g = loader.LoadGraph(vfile, efile);

          for (auto v : g.Vertices())
          {
            boost::to_lower(g[v]);

            for (auto &e : g.GetOutgoingAdjList(v))
            {
              boost::to_lower(g[e]);
            }
          }
        },
        g_vfile, g_efile);

    std::thread load_embedding_thread(
        [&comm, &word_embeddings](const std::string &embedding_file)
        {
          std::ifstream fi(embedding_file);
          std::string line;
          size_t last_dim = 0;

          while (getline(fi, line))
          {
            std::istringstream iss(line);
            std::string word;

            iss >> word;

            boost::to_lower(word);

            coord_t val;
            auto &vec = word_embeddings[word];
            std::vector<coord_t> tmp;

            while (iss >> val)
            {
              tmp.push_back(val);
            }

            if (last_dim == 0)
            {
              last_dim = tmp.size();
            }
            vec.resize(last_dim);

            for (size_t i = 0; i < last_dim; i++)
            {
              vec[i] = tmp[i];
            }
            if (last_dim == 0)
            {
              last_dim = vec.size();
            }
            else
            {
              CHECK_EQ(last_dim, vec.size())
                  << "Bad word vector: seen unmatched dim " << last_dim << " vs "
                  << vec.size();
            }
          }

          fi.close();
          if (comm.rank() == 0)
          {
            // LOG(INFO) << "Rank: " << comm.rank() << " Loaded " << embedding_file
            //           << ": " << word_embeddings.size() << " words";
          }
        },
        word_embedding_file);

    if (!synonym_file.empty())
    {
      std::ifstream fi(synonym_file);
      std::string line;

      while (getline(fi, line))
      {
        std::string word_a, word_b;
        coord_t score;

        std::vector<std::string> words;
        boost::split(words, line, boost::is_any_of(","),
                     boost::token_compress_on);

        CHECK_EQ(words.size(), 3) << "Bad line: " << line;

        word_a = words[0];
        word_b = words[1];
        score = std::stof(words[2]);

        CHECK_GT(score, 0) << "Bad line: " << line;
        CHECK_LE(score, 1) << "Bad line: " << line;

        boost::to_lower(word_a);
        boost::to_lower(word_b);
        synonym.template emplace(std::make_pair(word_a, word_b), score);
        synonym.template emplace(std::make_pair(word_b, word_a), score);
      }

      fi.close();
      if (comm.rank() == 0)
      {
        // LOG(INFO) << "Loaded " << synonym_file << ": " << synonym.size()
        //           << " words";
      }
    }

    if (FLAGS_query_type == "incremental" || FLAGS_query_type == "batch_spair")
    {
      std::thread load_g_updated_thread(
          [&loader, &g_updated](const std::string &vfile, const std::string &efile)
          {
            g_updated = loader.LoadGraph(vfile, efile);

            for (auto v : g_updated.Vertices())
            {
              boost::to_lower(g_updated[v]);

              for (auto &e : g_updated.GetOutgoingAdjList(v))
              {
                boost::to_lower(g_updated[e]);
              }
            }
          },
          g_vfile, g_updated_efile);

      std::thread load_gd_updated_thread(
          [&loader, &gd_updated](const std::string &vfile, const std::string &efile)
          {
            gd_updated = loader.LoadGraph(vfile, efile);

            for (auto v : gd_updated.Vertices())
            {
              boost::to_lower(gd_updated[v]);

              for (auto &e : gd_updated.GetOutgoingAdjList(v))
              {
                boost::to_lower(gd_updated[e]);
              }
            }
          },
          gd_vfile, gd_updated_efile);

      load_gd_updated_thread.join();
      load_g_updated_thread.join();
    }

    load_gd_thread.join();
    load_g_thread.join();
    load_embedding_thread.join();

    {
      std::ifstream fi(gd_slabel_file);
      std::string line;

      while (getline(fi, line))
      {
        boost::to_lower(line);

        gd_source_labels.insert(line);
      }
      fi.close();
    }
    {
      std::ifstream fi(g_slabel_file);
      std::string line;

      while (getline(fi, line))
      {
        boost::to_lower(line);

        g_source_labels.insert(line);
      }
      fi.close();
    }
    auto vm_ptr = g.vertex_map();

    if (!desc_file.empty())
    {
      std::ifstream fi(desc_file);
      std::string line;
      size_t n_desc = 0;

      g_descendants.resize(g.Vertices().size());

      while (getline(fi, line))
      {
        std::istringstream iss(line);
        oid_t v_oid;
        depth_t depth;
        vertex_t v;

        iss >> v_oid;

        if (vm_ptr->HasOid(v_oid))
        {
          CHECK(g.GetVertex(v_oid, v))
              << "Failed to get vertex with id: " << v_oid;
          auto &descendants = g_descendants[v];

          while (iss >> v_oid >> depth)
          {
            if (vm_ptr->HasOid(v_oid))
            {
              CHECK(g.GetVertex(v_oid, v))
                  << "Failed to get descendant with id: " << v_oid;
              descendants.push_back({v, depth});
              n_desc++;
            }
          }
        }
      }

      if (comm.rank() == 0)
      {
        LOG(INFO) << "Read " << n_desc << " descendants";
      }
      fi.close();
    }

    if (!path_file.empty())
    {
      std::ifstream fi(path_file);
      std::string line;
      size_t n_path = 0;

      while (getline(fi, line))
      {
        std::istringstream iss(line);
        oid_t v1_oid, v2_oid;
        vertex_t v1, v2;
        std::string path_labels;

        iss >> v1_oid;
        iss >> v2_oid;
        iss >> path_labels;

        boost::to_lower(path_labels);

        std::replace(path_labels.begin(), path_labels.end(), ';', ' ');
        std::replace(path_labels.begin(), path_labels.end(), ',', ' ');

        if (vm_ptr->HasOid(v1_oid) && vm_ptr->HasOid(v2_oid))
        {
          CHECK(g.GetVertex(v1_oid, v1))
              << "Failed to get vertex v1 with id: " << v1_oid;
          CHECK(g.GetVertex(v2_oid, v2))
              << "Failed to get vertex v2 with id: " << v2_oid;

          g_path[v1][v2] = path_labels;
          n_path++;
        }
      }

      if (comm.rank() == 0)
      {
        LOG(INFO) << "Read " << n_path << " paths";
      }
      fi.close();
    }
  }

  template <typename GRAPH_T, typename H_V, typename H_P, typename H_R>
  bool SPairQuery(GRAPH_T &gd, GRAPH_T &g, H_V &h_v, H_P &h_p, H_R &h_r)
  {
    using vertex_t = typename GRAPH_T::vertex_t;
    using oid_t = typename GRAPH_T::oid_t;

    double sigma = FLAGS_sigma;
    double delta = FLAGS_delta;
    int k = FLAGS_k;
    oid_t u_oid = FLAGS_vertex_u, v_oid = FLAGS_vertex_v;
    vertex_t u, v;

    CHECK(gd.GetVertex(u_oid, u))
        << "Can not found vertex " << u_oid << " from graph GD";
    CHECK(g.GetVertex(v_oid, v))
        << "Can not found vertex " << v_oid << " from graph G";

    SPair<GRAPH_T, H_V, H_P, H_R> s_pair(gd, g, h_v, h_p, h_r);

    s_pair.InitParams(sigma, delta, k);

    return s_pair.Query(u, v);
  }

  template <typename GRAPH_T, typename H_V, typename H_P, typename H_R>
  void BatchSPairQuery(GRAPH_T &gd, GRAPH_T &g, GRAPH_T &gd_updated, GRAPH_T &g_updated,
                       H_V &h_v, H_P &h_p, H_R &h_r, std::unordered_set<VertexPair<typename GRAPH_T::vertex_t>> &candidates)
  {

    auto RunBatchSPair = [&candidates, &h_v, &h_p, &h_r](GRAPH_T &gd, GRAPH_T &g) -> double
    {
      SPair<GRAPH_T, H_V, H_P, H_R> s_pair(gd, g, h_v, h_p, h_r);
      s_pair.InitParams(FLAGS_sigma, FLAGS_delta, FLAGS_k);
      s_pair.InitVcache(candidates);
      auto cache = s_pair.cache();
      auto beginTime = GetCurrentTime();

      for (auto pairs : candidates)
      {
        auto u = pairs.u(), v = pairs.v();
        bool match;
        // issue a query if not hit cache
        if (!cache.SetMatchIfHit(pairs, match))
        {
          match = s_pair.Query(u, v);
          // Whether the Boolean value is true or false determines whether the pair matches.
          // You can use two sets to these matches separately.
        }
      }
      auto spair_time = GetCurrentTime() - beginTime;
      return spair_time;
    };

    std::cout << "Batch varying g: " << RunBatchSPair(gd, g_updated) << std::endl;
    std::cout << "Batch varying gd: " << RunBatchSPair(gd_updated, g) << std::endl;

    // std::ofstream sp(FLAGS_tmp_file);
    // sp << RunBatchSPair(gd, g_updated) << std::endl;
    // sp << RunBatchSPair(gd_updated, g) << std::endl;
    // sp << RunBatchSPair(gd_updated, g_updated) << std::endl;
    // sp.close();
  }

  template <typename GRAPH_T, typename H_V, typename H_P, typename H_R>
  std::vector<typename GRAPH_T::vertex_t> VPairQuery(GRAPH_T &gd, GRAPH_T &g,
                                                     H_V &h_v, H_P &h_p,
                                                     H_R &h_r)
  {
    using vertex_t = typename GRAPH_T::vertex_t;
    using oid_t = typename GRAPH_T::oid_t;

    double sigma = FLAGS_sigma;
    double delta = FLAGS_delta;
    int k = FLAGS_k;
    oid_t u_oid = FLAGS_vertex_u;
    vertex_t u;

    CHECK(gd.GetVertex(u_oid, u))
        << "Can not found vertex " << u_oid << " from graph GD";

    VPair<GRAPH_T, H_V, H_P, H_R> v_pair(gd, g, h_v, h_p, h_r);

    v_pair.InitParams(sigma, delta, k);

    return v_pair.Query(u);
  }

  template <typename GRAPH_T, typename H_V, typename H_P, typename H_R,
            typename coord_t>
  std::vector<VertexPair<typename GRAPH_T::vertex_t>> APairQuery(
      GRAPH_T &gd, GRAPH_T &g, H_V &h_v, H_P &h_p, H_R &h_r,
      const VertexArray<dense_vector_t<coord_t>, GRAPH_T> &gd_label_vector,
      const VertexArray<dense_vector_t<coord_t>, GRAPH_T> &g_label_vector,
      const std::unordered_set<std::string> &gd_source_labels,
      const std::unordered_set<std::string> &g_source_labels,
      std::vector<VertexPair<typename GRAPH_T::vertex_t>> &raw_result,
      const InvertedIndex<GRAPH_T> &inverted_index, int parallelism)
  {
    double sigma = FLAGS_sigma;
    double delta = FLAGS_delta;
    int k = FLAGS_k;

    APairParallel<GRAPH_T, coord_t, H_V, H_P, H_R> a_pair(
        gd, g, h_v, h_p, h_r, gd_source_labels, g_source_labels, inverted_index);

    a_pair.InitParams(sigma, delta, k, parallelism);
    auto res = a_pair.Query();
    raw_result = a_pair.getRawRes();
    return res;
  }

  template <typename GRAPH_T, typename H_V, typename H_P, typename H_R>
  void IncMatchProcedure(
      std::unordered_set<VertexPair<typename GRAPH_T::vertex_t>> apair_match,
      std::unordered_set<VertexPair<typename GRAPH_T::vertex_t>> candidates,
      std::unordered_set<typename GRAPH_T::vertex_t> g_affected_vertexes,
      std::unordered_set<typename GRAPH_T::vertex_t> gd_affected_vertexes,
      GRAPH_T &gd, GRAPH_T &gd_updated, GRAPH_T &g, GRAPH_T &g_updated, H_V &h_v, H_P &h_p, H_R &h_r)
  {

    // LOG(INFO) << "IncMatch begin.";
    IncProcedure<GRAPH_T, H_V, H_P, H_R> raw_inc(gd, g, h_v, h_p, h_r);
    raw_inc.InitParams(FLAGS_sigma, FLAGS_delta, FLAGS_k, candidates);

    DepGraph<GRAPH_T> dependencies_graph;
    // Load or get dependencies graph
    {
      std::string filePath = FLAGS_out_prefix + "/dependenciesGraph";
      dependencies_graph.loadDepGraph(candidates, filePath);
      if (dependencies_graph.getEdgesNum() == 0)
      {
        // Not initialized
        dependencies_graph = raw_inc.getDependenciesGraph(candidates);
        dependencies_graph.storeDepGraph(filePath);
      }
    }

    // LOG(INFO) << "Dependencies graph ready.";

    auto RunInc = [&apair_match, &dependencies_graph, &candidates, &h_v, &h_p, &h_r](GRAPH_T &gd, GRAPH_T &g, std::unordered_set<typename GRAPH_T::vertex_t> &affected_vertexes) -> double
    {
      std::unordered_set<std::pair<VertexPair<typename GRAPH_T::vertex_t>,
                                   VertexPair<typename GRAPH_T::vertex_t>>>
          insertedEdges;
      std::unordered_set<std::pair<VertexPair<typename GRAPH_T::vertex_t>,
                                   VertexPair<typename GRAPH_T::vertex_t>>>
          deletedEdges;
      IncProcedure<GRAPH_T, H_V, H_P, H_R> inc(gd, g, h_v, h_p, h_r);
      inc.InitParams(FLAGS_sigma, FLAGS_delta, FLAGS_k, candidates);
      auto beginTime = GetCurrentTime();
      auto matches = dependencies_graph.getAffectedMatches(affected_vertexes);
      auto delta_pair = inc.getDeltaGraph(matches, dependencies_graph);
      insertedEdges = delta_pair.first;
      deletedEdges = delta_pair.second;
      auto result = inc.IncMatch(apair_match, dependencies_graph, deletedEdges, insertedEdges);
      auto plus = result.second, minus = result.first; // this is the result of new or revoked pairs.
      auto inc_time = GetCurrentTime() - beginTime;
      return inc_time;
    };

    std::cout << "Incremental varying g: " << RunInc(gd, g_updated, g_affected_vertexes) << std::endl;
    std::cout << "Incremental varying gd: " << RunInc(gd_updated, g, gd_affected_vertexes) << std::endl;
    // std::ofstream incRes(FLAGS_tmp_file);
    // incRes << RunInc(gd, g_updated, g_affected_vertexes) << std::endl;
    // incRes << RunInc(gd_updated, g, gd_affected_vertexes) << std::endl;
    // incRes.close();
  }

  void RunApp()
  {
    using oid_t = int32_t;
    using vid_t = uint32_t;
    using vdata_t = std::string;
    using edata_t = std::string;
    using graph_t = Graph<oid_t, vid_t, vdata_t, edata_t>;
    using vertex_t = typename graph_t::vertex_t;
    using coord_t = float;
    using point_t = dense_vector_t<coord_t>;

    boost::mpi::communicator comm;
    graph_t gd, g, g_updated, gd_updated;
    std::unordered_map<std::string, dense_vector_t<coord_t>> word_embedding;
    std::unordered_set<std::string> gd_source_labels, g_source_labels;
    std::unordered_map<std::pair<std::string, std::string>, coord_t> synonym;
    std::vector<std::vector<std::pair<vertex_t, depth_t>>> g_descendants;
    std::unordered_map<vertex_t, std::unordered_map<vertex_t, std::string>>
        g_path;
    InvertedIndex<graph_t> inverted_index;
    VertexArray<point_t, graph_t> gd_label_vector;
    VertexArray<point_t, graph_t> gd_updated_label_vector;
    VertexArray<point_t, graph_t> g_label_vector;
    VertexArray<point_t, graph_t> g_updated_label_vector;
    int parallelism = GetParallelism(comm);

    // LOG(INFO) << "Rank: " << comm.rank() << " thread num: " << parallelism;

    timer_start(comm.rank() == 0);
    // timer_next("Load Data");

    LoadData(comm, gd, g, g_updated, gd_updated, word_embedding, gd_source_labels, g_source_labels,
             synonym, g_descendants, g_path);

    // timer_next("Filling word vector");
    FillWordVector(gd, word_embedding, gd_label_vector, parallelism);
    FillWordVector(g, word_embedding, g_label_vector, parallelism);
    if (FLAGS_query_type == "incremental")
    {
      FillWordVector(gd_updated, word_embedding, gd_label_vector, parallelism);
      FillWordVector(g_updated, word_embedding, g_updated_label_vector, parallelism);
    }

    // timer_next("Init inverted index");
    inverted_index.Init(g, g_source_labels);

    comm.barrier();

    auto h_v = [&gd_label_vector, &g_label_vector, &synonym](
                   graph_t &gd, vertex_t u, graph_t &g, vertex_t v) -> coord_t
    {
      auto &u_label = gd[u];
      auto &v_label = g[v];

      if (u_label == v_label)
      {
        return 1.0;
      }

      auto pair = std::make_pair(u_label, v_label);
      auto it = synonym.find(pair);

      // if u_label v_label is a pair of synonym, then return score
      if (it != synonym.end())
      {
        return it->second;
      }

      auto u_vector = gd_label_vector[u];
      auto v_vector = g_label_vector[v];
      return CosineSimilarity(u_vector, v_vector);
    };

    auto h_p = [&word_embedding, &synonym, &g_path](
                   const graph_t &gd, vertex_t u, vertex_t u1, graph_t &g,
                   vertex_t v, vertex_t v1) -> coord_t
    {
      std::string path_u_u1 = ConcatEdgeLabel(gd, u, u1, " ");
      std::string path_v_v1;
      auto desc_it = g_path.find(v);
      // Firstly, finding path with v,v1 endpoints in path file
      if (desc_it != g_path.end())
      {
        auto end_it = desc_it->second.find(v1);

        if (end_it != desc_it->second.end())
        {
          path_v_v1 = end_it->second;
        }
      }

      // Then, if the path can not be found, start a BFS to construct a path
      if (path_v_v1.empty())
      {
        path_v_v1 = ConcatEdgeLabel(g, v, v1, " ");
      }

      // Concat label between u...v
      if (!path_u_u1.empty() && !path_v_v1.empty())
      {
        if (path_u_u1 == path_v_v1)
        {
          return 1.0;
        }

        auto it = synonym.find(std::make_pair(path_u_u1, path_v_v1));

        // if u_label v_label is a pair of synonym, then return score
        if (it != synonym.end())
        {
          return it->second;
        }

        auto e1_vector = TextToVector(word_embedding, path_u_u1);
        auto e2_vector = TextToVector(word_embedding, path_v_v1);

        return CosineSimilarity(e1_vector, e2_vector);
      }

      return 0.0;
    };

    int bfs_depth = FLAGS_bfs_depth;

    auto h_r = [bfs_depth, &g_descendants](const graph_t &g_or_gd,
                                           vertex_t u_or_v, size_t k, bool is_g)
    {
      std::vector<std::pair<vertex_t, depth_t>> top_k_descendants;
      if (is_g && !g_descendants.empty())
      {
        auto &desc = g_descendants[u_or_v];

        k = std::min(k, desc.size());
        top_k_descendants.assign(desc.begin(), desc.begin() + k);
      }
      else
      {
        top_k_descendants = BFS(g_or_gd, u_or_v, bfs_depth, k);
      }
      return top_k_descendants;
    };

    std::string query_type = FLAGS_query_type;
    size_t n_iter = FLAGS_n_iter;

    std::vector<vertex_t> gd_sources;

    if (!FLAGS_vpair_sources_file.empty())
    {
      std::ifstream fi(FLAGS_vpair_sources_file);
      std::string line;

      while (getline(fi, line))
      {
        std::istringstream iss(line);
        oid_t oid;
        vertex_t v;

        iss >> oid;
        CHECK(g.GetVertex(oid, v)) << "Invalid oid: " << oid;
        gd_sources.push_back(v);
      }

      fi.close();
    }

    // timer_next("Query");

    if (query_type == "spair")
    {
      // bool ans = SPairQuery(gd, g, h_v, h_p, h_r);

      // LOG(INFO) << "Query: (" << FLAGS_vertex_u << ", " << FLAGS_vertex_v
      //           << ") = " << (ans ? "True" : "False");
    }
    else if (query_type == "spair_benchmark")
    {
      double sigma = FLAGS_sigma;
      double delta = FLAGS_delta;
      int k = FLAGS_k;

      SPair<graph_t, decltype(h_v), decltype(h_p), decltype(h_r)> s_pair(
          gd, g, h_v, h_p, h_r);

      s_pair.InitParams(sigma, delta, k);

      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> dist_gd(0, gd.Vertices().size() - 1);
      std::uniform_int_distribution<> dist_g(0, g.Vertices().size() - 1);

      auto begin = GetCurrentTime();
      bool result = false;
      for (size_t i = 0; i < n_iter; i++)
      {
        vertex_t u = dist_gd(gen);
        vertex_t v = dist_g(gen);

        // prevent compilter optimize it away
        result |= s_pair.Query(u, v);
      }
      timer_next("Average Query", (GetCurrentTime() - begin) / n_iter);
      VLOG(99) << result;
    }
    else if (query_type == "vpair")
    {
      auto ans = VPairQuery(gd, g, h_v, h_p, h_r);

      timer_next("Output");

      LOG(INFO) << "Matched pairs: ";
      for (auto v : ans)
      {
        oid_t v_oid;

        CHECK(g.GetId(v, v_oid));
        // Output matched
        LOG(INFO) << v_oid << "|" << g[v] << std::endl;
      }
    }
    else if (query_type == "vpair_benchmark")
    {
      double sigma = FLAGS_sigma;
      double delta = FLAGS_delta;
      int k = FLAGS_k;

      VPair<graph_t, decltype(h_v), decltype(h_p), decltype(h_r)> v_pair(
          gd, g, h_v, h_p, h_r);

      v_pair.InitParams(sigma, delta, k);

      if (gd_sources.empty())
      {
        LOG(FATAL) << "Having an empty gd sources";
      }

      size_t size = 0;
      auto begin = GetCurrentTime();
      for (auto u : gd_sources)
      {
        size += v_pair.Query(u).size();
      }
      timer_next("Average Query", (GetCurrentTime() - begin) / gd_sources.size());
      VLOG(99) << size;
    }
    else if (query_type == "apair")
    {
      std::string out_prefix = FLAGS_out_prefix;
      std::vector<VertexPair<vertex_t>> ans, raw_ans;

      ans = APairQuery(gd, g, h_v, h_p, h_r, gd_label_vector, g_label_vector,
                       gd_source_labels, g_source_labels, raw_ans, inverted_index,
                       parallelism);

      comm.barrier();

      if (!out_prefix.empty())
      {
        std::ofstream apair_fo(out_prefix + "/apair_" + std::to_string(comm.rank()));
        std::ofstream candidates_fo(out_prefix + "/candidates_" + std::to_string(comm.rank()));

        for (auto &u_v : ans)
        {
          vertex_t u = u_v.u(), v = u_v.v();
          oid_t u_oid, v_oid;

          CHECK(gd.GetId(u, u_oid));
          CHECK(g.GetId(v, v_oid));

          // fo << u_oid << "|" << v_oid << "|" << gd[u] << "|" << g[v] << std::endl;
          apair_fo << u_oid << " " << v_oid << std::endl;
        }
        apair_fo.close();

        for (auto &u_v : raw_ans)
        {
          vertex_t u = u_v.u(), v = u_v.v();
          oid_t u_oid, v_oid;

          CHECK(gd.GetId(u, u_oid));
          CHECK(g.GetId(v, v_oid));

          // fo << u_oid << "|" << v_oid << "|" << gd[u] << "|" << g[v] << std::endl;
          candidates_fo << u_oid << " " << v_oid << std::endl;
        }
        candidates_fo.close();
      }
    }
    else if (query_type == "incremental")
    {

      std::unordered_set<VertexPair<vertex_t>> ans, candidates;
      std::unordered_set<vertex_t> g_affected_vertexes;
      std::unordered_set<vertex_t> gd_affected_vertexes;
      std::string out_prefix = FLAGS_out_prefix;
      std::string deltaG = FLAGS_delta_g;
      std::string deltaGd = FLAGS_delta_gd;
      // load candidates, dependencede graphs
      if (!out_prefix.empty())
      {
        std::ifstream apair_fo(out_prefix + "/apair_0");
        std::ifstream candidates_fo(out_prefix + "/candidates_0");
        std::ifstream deltaG_fo(deltaG);
        std::ifstream deltaGd_fo(deltaGd);

        oid_t u_oid, v_oid;
        vertex_t u, v;

        while (apair_fo >> u >> v)
        {
          ans.insert(VertexPair<vertex_t>(u, v));
        }
        apair_fo.close();

        while (candidates_fo >> u >> v)
        {
          candidates.insert(VertexPair<vertex_t>(u, v));
        }
        candidates_fo.close();

        while (deltaG_fo >> u_oid >> v_oid)
        {
          CHECK(g.GetVertex(v_oid, v));
          g_affected_vertexes.insert(v);
        }
        deltaG_fo.close();

        while (deltaGd_fo >> u_oid >> v_oid)
        {
          CHECK(gd.GetVertex(v_oid, v));
          gd_affected_vertexes.insert(v);
        }
        deltaGd_fo.close();
      }

      IncMatchProcedure(ans, candidates, g_affected_vertexes, gd_affected_vertexes, gd, gd_updated, g, g_updated, h_v, h_p, h_r);
    }
    else if (query_type == "batch_spair")
    {
      std::unordered_set<VertexPair<vertex_t>> candidates;
      std::string out_prefix = FLAGS_out_prefix;
      // load candidates, dependencede graphs
      if (!out_prefix.empty())
      {
        std::ifstream candidates_fo(out_prefix + "/candidates_0");
        vertex_t u, v;
        while (candidates_fo >> u >> v)
        {
          candidates.insert(VertexPair<vertex_t>(u, v));
        }
        candidates_fo.close();
      }
      BatchSPairQuery(gd, g, gd_updated, g_updated, h_v, h_p, h_r, candidates);
    }
    else
    {
      LOG(FATAL) << "Invalid param: query_type = " << query_type;
    }

    timer_end();
  }
} // namespace her
#endif // HER_HER_H_
