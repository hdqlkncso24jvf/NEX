#ifndef EXAMPLES_ANALYTICAL_APPS_GAR_DISCOVER_GRAPH_PACKAGE_H_
#define EXAMPLES_ANALYTICAL_APPS_GAR_DISCOVER_GRAPH_PACKAGE_H_

#include "gundam/algorithm/neighborhood_equivalence_class.h"

#include "gundam/tool/isolate_vertex.h"
#include "gundam/tool/vertex_degree_filter.h"
#include "gundam/data_type/datatype.h"

#include "gundam/io/csvgraph.h"

#include "gundam/graph_statistics/graph_basic_statistics.h"

#include "gundam/type_getter/vertex_label.h"
#include "gundam/type_getter/vertex_id.h"
#include "gundam/type_getter/edge_label.h"
#include "gundam/type_getter/edge_id.h"

#include "util/log.h"

namespace grape {

namespace _gar_discover { 

template <typename GraphType>
class GraphPackage{
 private:
  using VertexIDType = typename GUNDAM::VertexID<GraphType>::type;
  using   EdgeIDType = typename GUNDAM::  EdgeID<GraphType>::type;

  using VertexLabelType = typename GUNDAM::VertexLabel<GraphType>::type;
  using   EdgeLabelType = typename GUNDAM::  EdgeLabel<GraphType>::type;

  using EdgeTypeType = std::tuple<VertexLabelType,
                                    EdgeLabelType,
                                  VertexLabelType>;

  using GraphBasicStatisticsType = GUNDAM::GraphBasicStatistics<GraphType>;

 public:
  GraphPackage(
    const std::set<EdgeTypeType>&  specified_edge_type_set,
    const std::set<EdgeLabelType>& specified_edge_label_set,
    const double constant_freq_bound,
    const std::string& kGraphPathVFile,
    const std::string& kGraphPathEFile,
    GraphBasicStatisticsType& graph_basic_statistics) {

    this->load_data_graph_success_ = false;
      
    this->graph_path_v_file_ = kGraphPathVFile;
    this->graph_path_e_file_ = kGraphPathEFile;

    if (GUNDAM::ReadCSVGraph(this->data_graph_, 
                              kGraphPathVFile,
                              kGraphPathEFile) < 0) {
      util::Error("load data graph failed!");
      return;
    }

    this->load_data_graph_success_ = true;

    graph_basic_statistics.AddGraph(this->data_graph_);

    const auto kMaxVertexId = graph_basic_statistics.max_vertex_id();
    const auto kMaxEdgeId   = graph_basic_statistics.max_edge_id();

    util::Info(" vertex_label_set.size(): " + std::to_string(graph_basic_statistics.vertex_label_counter().size()));
    util::Info("   edge_label_set.size(): " + std::to_string(graph_basic_statistics.  edge_label_counter().size()));
    util::Info("    edge_type_set.size(): " + std::to_string(graph_basic_statistics.   edge_type_counter().size()));

    this->max_edge_id_ = kMaxEdgeId;

    if (!specified_edge_type_set.empty()) {

      graph_basic_statistics.PreserveEdgeTypeSet(specified_edge_type_set);

      util::Info("################################");
      util::Info("##   preserve edge type set   ##");
      util::Info("################################");

      util::Info("   edge type considered: "
                + std::to_string(graph_basic_statistics.edge_type_counter().size()));
      util::Info("vertex label considered: "
                + std::to_string(graph_basic_statistics.vertex_label_counter().size()));
      util::Info("  edge label considered: "
                + std::to_string(graph_basic_statistics.edge_label_counter().size()));
    }

    if (!specified_edge_label_set.empty()) {
      graph_basic_statistics.PreserveEdgeLabelSet(specified_edge_label_set);

      util::Info("#################################");
      util::Info("##   preserve edge label set   ##");
      util::Info("#################################");

      util::Info("   edge type considered: "
                + std::to_string(graph_basic_statistics.edge_type_counter().size()));
      util::Info("vertex label considered: "
                + std::to_string(graph_basic_statistics.vertex_label_counter().size()));
      util::Info("  edge label considered: "
                + std::to_string(graph_basic_statistics.edge_label_counter().size()));
    }
    
    if (!specified_edge_type_set.empty()
     || !specified_edge_label_set.empty()) {
        
      for (auto vertex_it = this->data_graph_.VertexBegin();
               !vertex_it.IsDone();) {
        if (graph_basic_statistics.vertex_label_counter().find(vertex_it->label())
         == graph_basic_statistics.vertex_label_counter().end()) {
          vertex_it = this->data_graph_.EraseVertex(vertex_it);
          continue;
        }
        vertex_it++;
      }

      for (auto vertex_it = this->data_graph_.VertexBegin();
               !vertex_it.IsDone();
                vertex_it++) {
        for (auto out_edge_it = vertex_it->OutEdgeBegin();
                 !out_edge_it.IsDone();) {
          const auto edge_type = std::tuple(out_edge_it->src_handle()->label(),
                                            out_edge_it->label(),
                                            out_edge_it->dst_handle()->label());
          if (graph_basic_statistics.edge_type_counter().find(edge_type)
           != graph_basic_statistics.edge_type_counter().end()) {
            // this edge does not need to be removed;
            out_edge_it++;
            continue;
          }
          auto dst_handle = out_edge_it->dst_handle();
          out_edge_it = vertex_it->EraseEdge(out_edge_it);
        }
      }

      GUNDAM::RemoveIsolateVertex(this->data_graph_);
    }

    // assume that the vertex with the same label has the same
    // attributes set, collect all legal keys and values for
    // attribute
    graph_basic_statistics.ResetVertexAttr(this->data_graph_,
                                            constant_freq_bound);

    util::Info("data graph vertex number: " + std::to_string(this->data_graph_.CountVertex()));
    util::Info("data graph  edge  number: " + std::to_string(this->data_graph_.CountEdge  ()));
    return;
  }

  inline const std::string& graph_path_v_file() const {
    return this->graph_path_v_file_;
  }

  inline const std::string& graph_path_e_file() const {
    return this->graph_path_e_file_;
  }

  inline bool AddMlEdge(const std::string& kMlLiteralEdgesFile,
                   std::map<EdgeLabelType,
                            EdgeLabelType>& normal_to_ml_edge_label) {
    if (kMlLiteralEdgesFile == "") {
      // does not have ml edge
      return true;
    }
    // has ml literal
    std::ifstream ml_literal_edge_file(kMlLiteralEdgesFile);
    
    if (!ml_literal_edge_file.good()){
      std::cout << " ml literal edge file is not good! " << std::endl;
      return false;
    }
                  
    EdgeIDType edge_id_allocator = this->max_edge_id_;
    edge_id_allocator++;

    while (ml_literal_edge_file) {
      std::string s;
      if (!std::getline( ml_literal_edge_file, s )) 
        break;

      std::istringstream ss( s );
      std::vector <std::string> record;
      record.reserve(3);
      while (ss) {
        std::string s;
        if (!std::getline( ss, s, ',' )) {
          break;
        }
        record.emplace_back( s );
      }
       VertexIDType  src_id    = GUNDAM::StringToDataType< VertexIDType>(record[0]);
      EdgeLabelType edge_label = GUNDAM::StringToDataType<EdgeLabelType>(record[1]);
       VertexIDType  dst_id    = GUNDAM::StringToDataType< VertexIDType>(record[2]);

      assert(normal_to_ml_edge_label.find(edge_label)
          != normal_to_ml_edge_label.end());
      EdgeLabelType ml_edge_label = normal_to_ml_edge_label[edge_label];
      auto [ edge_handle,
              edge_ret ] = this->data_graph_.AddEdge(src_id, dst_id, 
                                                  ml_edge_label, 
                                                    edge_id_allocator++);
      assert(edge_ret);
    }
    return true;
  }

  GraphType& data_graph(){
    return this->data_graph_;
  }

  inline const auto& data_graph_nec() const {
    return this->data_graph_nec_;
  }

  inline void GenerateGraphNec() {
    this->data_graph_nec_ = GUNDAM::Nec(this->data_graph_);
    return;
  }

  inline bool load_data_graph_success() const {
    return this->load_data_graph_success_;
  }

  inline void GenerateCentralSet() {
    std::vector<typename GUNDAM::VertexHandle<GraphType>::type> centrel_vertex;
    centrel_vertex = GUNDAM::VertexDegreeFilter<
                     GUNDAM::FilterType::kHigherOrEqualTo,
                     GUNDAM::EdgeDirection::kInOut>(this->data_graph_, 3);
    for (const auto& vertex : centrel_vertex) {
      this->centrel_vertex_set_[vertex->label()].emplace_back(vertex);
    }
    for (auto& [vertex_label, centrel_vertex_set] 
                      : this->centrel_vertex_set_) {
      std::sort(centrel_vertex_set.begin(),
                centrel_vertex_set.end());
      util::Debug("vertex_label: "              + std::to_string(vertex_label));
      util::Debug("centrel_vertex_set.size(): " + std::to_string(centrel_vertex_set.size()));
    }
    return;
  }

  inline const auto& centrel_vertex_set(
         const typename GUNDAM::VertexLabel <GraphType>::type& vertex_label) const {
    auto centrel_vertex_set_it = this->centrel_vertex_set_.find(vertex_label);
    assert(centrel_vertex_set_it
        != this->centrel_vertex_set_.end());
    return centrel_vertex_set_it->second;
  }

  inline void SampleCentralSet(double sample_ratio) {
    assert(!this->centrel_vertex_set_.empty());
    assert(sample_ratio > 0 
        && sample_ratio < 1.0);
    for (auto& [vertex_label, centrel_vertex_set] 
                      : this->centrel_vertex_set_) {
      assert(std::is_sorted(centrel_vertex_set.begin(),
                            centrel_vertex_set.end()));
      std::vector<
      typename GUNDAM::VertexHandle<GraphType>::type> preserved_centrel_vertex_set;
      for (const auto& centrel_vertex_handle  
                     : centrel_vertex_set) {
        const double kR = ((double) rand() / (RAND_MAX));
        if (kR > sample_ratio) {
          /// this vertex should not be preserved
          continue;
        }
        /// this vertex should be preserved
        preserved_centrel_vertex_set.emplace_back(centrel_vertex_handle);
      }
      centrel_vertex_set.swap(preserved_centrel_vertex_set);
      assert(std::is_sorted(centrel_vertex_set.begin(),
                            centrel_vertex_set.end()));
      util::Debug("vertex_label: "              + std::to_string(vertex_label));
      util::Debug("centrel_vertex_set.size(): " + std::to_string(centrel_vertex_set.size()));
    }
    return;
  }

 private:
  bool load_data_graph_success_;

  std::string graph_path_v_file_,
              graph_path_e_file_;

   GraphType  data_graph_;
  EdgeIDType max_edge_id_;

  std::vector<
  std::vector<typename GUNDAM::VertexHandle<GraphType>::type>> data_graph_nec_;

  std::map<typename GUNDAM::VertexLabel <GraphType>::type,
           std::vector<
           typename GUNDAM::VertexHandle<GraphType>::type>> centrel_vertex_set_;
};

}; // _gar_discover

}; // grape

#endif // EXAMPLES_ANALYTICAL_APPS_GAR_DISCOVER_GRAPH_PACKAGE_H_