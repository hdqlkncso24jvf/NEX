#ifndef EXAMPLES_ANALYTICAL_APPS_GAR_ACCURACY_ANALYSIS_GAR_ACCURACY_ANALYSIS_H_
#define EXAMPLES_ANALYTICAL_APPS_GAR_ACCURACY_ANALYSIS_GAR_ACCURACY_ANALYSIS_H_

#include <grape/grape.h>
#include <omp.h>

#include <functional>

#include "gundam/graph_type/graph.h"
#include "gundam/graph_type/small_graph.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/large_graph2.h"

#include "gundam/match/match.h"
#include "gundam/match/matchresult.h"

#include "gundam/io/csvgraph.h"
#include "gundam/tool/find_edge.h"

#include "gar/gar_supp.h"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "gar_accuracy_analysis/gar_accuracy_analysis_context.h"

#include "gar/csv_gar.h"
#include "gar/gar.h"
#include "gar/literal_stand_alone_info.h"
#include "gar/gar_config.h"

#include "util/log.h"
#include "util/yaml_config.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../timer.h"

namespace grape {

namespace _gar_accuracy_analysis {

// to select vertexes_pair_num positive test set for edge literal,
// e.g. vertex pair connected by edge, for following testing
template <typename   PatternType,
          typename DataGraphType>
void CollectPositiveEdgeSet(DataGraphType& data_graph,
        const gar::LiteralStandAloneInfo<
                              PatternType, 
                            DataGraphType>& stand_alone_info,
                                   size_t  vertexes_pair_num,
        std::vector<std::tuple<
          typename GUNDAM::VertexHandle<DataGraphType>::type,  // src vertex handle
        std::vector<
          typename GUNDAM::   EdgeLabel<DataGraphType>::type>, // edge label
          typename GUNDAM::VertexHandle<DataGraphType>::type,  // dst vertex handle
          bool, // whether exist edges, e.g. is positive or negative dataset 
          bool> // whether can be chased by the gars
        >& test_set) {

  assert(test_set.empty());

  using EdgeIDType = typename GUNDAM::EdgeID<DataGraphType>::type;

  EdgeIDType max_edge_id = 0;
  for (auto vertex_it = data_graph.VertexBegin();
           !vertex_it.IsDone();
            vertex_it++) {
    for (auto out_edge_it = vertex_it->OutEdgeBegin();
             !out_edge_it.IsDone();
              out_edge_it++) {
      max_edge_id = max_edge_id > out_edge_it->id()?
                    max_edge_id : out_edge_it->id();
    }
  }
  std::set<EdgeIDType> selected_edge_id_set;
  for (size_t i = 0; i < vertexes_pair_num; i++) {
    while (true) {
      auto edge_handle = data_graph.FindEdge(std::rand() % (max_edge_id + 1));
      // since the edge id might not continues, the random selected
      // edge id might actually not exist in the data graph
      if (!edge_handle) {
        // this edge id does not exist in the data graph
        // re-rand
        continue;
      }
      // has found a valid edge id, to find whether this selected
      // edge can be the chase result of the test set
      if (edge_handle->src_handle()->label() != stand_alone_info.x_label()
       || edge_handle->dst_handle()->label() != stand_alone_info.y_label()
       || edge_handle->label() != stand_alone_info.edge_label()) {
        // this selected edge 
        continue;
      }

      auto [ selected_edge_it,
             selected_edge_ret ] 
           = selected_edge_id_set.emplace(edge_handle->id());
      if (!selected_edge_ret) {
        // add fail, this edge has been selected
        continue;
      }
      // has found an edge that have not been considered before
      test_set.emplace_back(edge_handle->src_handle(), 
      std::vector<
          typename GUNDAM::   EdgeLabel<DataGraphType>::type>
                           {edge_handle->label()}, 
                            edge_handle->dst_handle(), 
                            true,  // is positive data 
                            false);// has not been chased by the rules

      assert(selected_edge_id_set.size() == test_set.size());
      break;
    }
    assert(selected_edge_id_set.size() == i + 1);
    assert(            test_set.size() == i + 1);
  }
  assert(selected_edge_id_set.size() == vertexes_pair_num);
  assert(            test_set.size() == vertexes_pair_num);
  return;
}

// to select vertexes_pair_num negative test set for edge literal,
// e.g. vertex pairs not connected by edge, or both negative and 
// positive data for following testing
template <typename   PatternType,
          typename DataGraphType>
void CollectNegativeOrBothEdgeSet(DataGraphType& data_graph,
        const gar::LiteralStandAloneInfo<
                              PatternType, 
                            DataGraphType>& stand_alone_info,
                                bool negative_test_data_only,
                                   size_t  vertexes_pair_num,
        std::vector<std::tuple<
          typename GUNDAM::VertexHandle<DataGraphType>::type,  // src vertex handle
        std::vector<
          typename GUNDAM::   EdgeLabel<DataGraphType>::type>, // edge label
          typename GUNDAM::VertexHandle<DataGraphType>::type,  // dst vertex handle
          bool, // whether exist edges, e.g. is positive or negative dataset 
          bool> // whether can be chased by the gars
        >& test_set) {

  assert(test_set.empty());

  using DataGraphVertexHandleType 
      = typename GUNDAM::VertexHandle<DataGraphType>::type;

  using VertexIDType 
      = typename GUNDAM::VertexID<DataGraphType>::type;

  std::set<std::pair<DataGraphVertexHandleType, // src vertex handle
                     DataGraphVertexHandleType  // dst vertex handle
                    >
          > selected_vertex_pair;
  // generate the test set according to the stand_alone_info
  // from the data graph 
  VertexIDType max_vertex_id = 0;
  for (auto vertex_it = data_graph.VertexBegin();
           !vertex_it.IsDone();
            vertex_it++) {
    max_vertex_id = max_vertex_id > vertex_it->id()?
                    max_vertex_id : vertex_it->id();
  }
  for (int i = 0; i < vertexes_pair_num; i++) {
    while (true) {
      DataGraphVertexHandleType src_handle = DataGraphVertexHandleType();
      assert(!src_handle);
      while (true) {
        src_handle = data_graph.FindVertex(std::rand() % (max_vertex_id + 1));
        if (!src_handle) {
          // does not have this vertex, since the id of vertexes
          // in data graph might not continues
          continue;
        }
        if (src_handle->label() == stand_alone_info.x_label()) {
          break;
        }
      }
      assert(src_handle);
      assert(src_handle->label() == stand_alone_info.x_label());

      DataGraphVertexHandleType dst_handle = DataGraphVertexHandleType();
      assert(!dst_handle);
      while (true) {
        dst_handle = data_graph.FindVertex(std::rand() % (max_vertex_id + 1));
        if (!dst_handle) {
          // does not have this vertex, since the id of vertexes
          // in data graph might not continues
          continue;
        }
        if (src_handle == dst_handle) {
          // src and dst vertex cannot be the same
          // since does not consider self-loop
          continue;
        }
        if (dst_handle->label() == stand_alone_info.y_label()) {
          break;
        }
      }
      assert(dst_handle);
      assert(dst_handle->label() == stand_alone_info.y_label());

      auto [selected_vertex_it,
            selected_vertex_ret] = selected_vertex_pair.emplace(src_handle,
                                                                dst_handle);
      if (!selected_vertex_ret) {
        // this vertex pair has already existed
        // has not added successfully
        continue;
      }
      // this vertex has not been added before
      bool src_dst_connected_by_edge_label = false;
      // need to verify there are no edge 
      // connects these two vertex
      for (auto out_edge_it = src_handle->OutEdgeBegin();
               !out_edge_it.IsDone();
                out_edge_it++) {
        if (out_edge_it->label() != stand_alone_info.edge_label()) {
          continue;
        }
        if (out_edge_it->dst_handle() != dst_handle) {
          continue;
        }
        src_dst_connected_by_edge_label = true;
        break;
      }
      if (negative_test_data_only
       && src_dst_connected_by_edge_label) {
        // need only to find negative test data
        // but this one is positive
        continue;
      }
      test_set.emplace_back(src_handle, 
      std::vector<
          typename GUNDAM::   EdgeLabel<DataGraphType>::type>
         {stand_alone_info.edge_label()},
                            dst_handle, 
       src_dst_connected_by_edge_label,  // is positive or negative
                                 false); // has not been chased by rules
      assert(selected_vertex_pair.size()
                      >= test_set.size());
      break;
    }
    assert(selected_vertex_pair.size() >= i + 1);
    assert(            test_set.size() == i + 1);
  }
  assert(selected_vertex_pair.size() >= vertexes_pair_num);
  assert(            test_set.size() == vertexes_pair_num);
  return;
}

};

template <typename FRAG_T>
class GarAccuracyAnalysis : public ParallelAppBase<FRAG_T, GarAccuracyAnalysisContext<FRAG_T>>,
                            public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
  using    VertexIDType = typename GarAccuracyAnalysisContext<FRAG_T>::   VertexIDType;
  using VertexLabelType = typename GarAccuracyAnalysisContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType = typename GarAccuracyAnalysisContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType = typename GarAccuracyAnalysisContext<FRAG_T>::  EdgeLabelType;
  using    DataGraph    = typename GarAccuracyAnalysisContext<FRAG_T>::  DataGraph;
  using      Pattern    = typename GarAccuracyAnalysisContext<FRAG_T>::    Pattern;

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(GarAccuracyAnalysis<FRAG_T>, 
                          GarAccuracyAnalysisContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for GarAccuracyAnalysis.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void PEval(const fragment_t& frag, 
                    context_t& ctx,
            message_manager_t& messages) {

    // assume that the gar have same rhs

    using GarType = gar::GraphAssociationRule <Pattern, DataGraph>;
    using LiteralStandAloneInfoType 
                  = gar::LiteralStandAloneInfo<Pattern, DataGraph>;

    using   PatternVertexHandleType = typename GUNDAM::VertexHandle< Pattern >::type;
    using DataGraphVertexHandleType = typename GUNDAM::VertexHandle<DataGraph>::type;
    using DataGraphEdgeLabelType    = typename GUNDAM::  EdgeLabel <DataGraph>::type;

    util::Info("yaml file: " + ctx.yaml_file_);

    YAML::Node config = YAML::LoadFile(ctx.yaml_file_);

    DataGraph data_graph;
    if (!config["DataGraph"]) {
      util::Error("does not specify data graph path!");
      return;
    }
    YAML::Node data_graph_info_node = config["DataGraph"];

    if (!data_graph_info_node["VFile"]) {
      util::Error("cannot get data graph v file");
      return;
    }
    const std::string kDataGraphPathVFile 
            = data_graph_info_node["VFile"].as<std::string>();

    if (!data_graph_info_node["EFile"]) {
      util::Error("cannot get data graph e file");
      return;
    }
    const std::string kDataGraphPathEFile 
            = data_graph_info_node["EFile"].as<std::string>();

    if (GUNDAM::ReadCSVGraph(data_graph, 
                            kDataGraphPathVFile,
                            kDataGraphPathEFile) < 0) {
      util::Error("load data graph failed!");
      return;
    }
    
    if (!config["OutputLogFile"]) {
      util::Error("does not specify output log file!");
      return;
    }
    const std::string kOutputLogFile 
             = config["OutputLogFile"].as<std::string>();

    size_t vertexes_pair_num = 0;
    std::string test_data_type = "both";

    std::vector<std::string> test_set_file_path_path_set;

    if (config["TestSetFile"]) {
      // sepcified test set
      YAML::Node test_set_file_path_config = config["TestSetFile"];
      if (!test_set_file_path_config.IsSequence()) {
        test_set_file_path_path_set.emplace_back(
        test_set_file_path_config.as<std::string>());
      }
      else {
        if (test_set_file_path_config.size() == 0) {
          util::Error("Specified empty test file set");
          return;
        }
        for (int i = 0; i < test_set_file_path_config.size(); i++) {
          const std::string kTestSetFile = test_set_file_path_config[i].as<std::string>();
          test_set_file_path_path_set.emplace_back(kTestSetFile);
        }
        assert(!test_set_file_path_path_set.empty());
      }
    }
    else {
      // generate test set
      if (!config["VertexesPairNum"]) {
        util::Error("does not specify vertexes pair number!");
        return;
      }
      vertexes_pair_num = config["VertexesPairNum"].as<size_t>();

      if (config["TestDataType"]) {
        test_data_type = config["TestDataType"].as<std::string>();
        if (test_data_type != "positive"
         && test_data_type != "negative"
         && test_data_type != "both") {
          util::Error("unknown TestDataType: " + test_data_type);
          return;
        }
      }
    }

    std::ofstream output_log(kOutputLogFile);
    if (!output_log) {
      util::Error("open file fail!" + kOutputLogFile);
      return;
    }

    if (!config["GARPath"]) {
      util::Error("cannot get gar dirs!");
      return;
    }
    YAML::Node gar_path_set_config = config["GARPath"];

    std::vector<gar::LiteralStandAloneInfo<Pattern, DataGraph>> specified_rhs_literal_info_set;
    if (config["SpecifiedRhsLiteralSet"]) {
      YAML::Node specified_rhs_literal_set_config 
       = config["SpecifiedRhsLiteralSet"];
      if (test_set_file_path_path_set.empty()
       && specified_rhs_literal_set_config.size() > 1) {
        util::Error("Specified: " + std::to_string(specified_rhs_literal_set_config.size()) 
                 + " rhs literals, currently supports only to specify multi rhs literal when is test set is specified");
        return;
      }
      for (int i = 0; i < specified_rhs_literal_set_config.size(); i++) {
        auto [ stand_alone_literal_info,
               stand_alone_literal_info_ret ]
          = util::LiteralStandAloneInfoFromYaml<Pattern, DataGraph>(
                specified_rhs_literal_set_config[i]);
        if (stand_alone_literal_info.literal_type() != gar::LiteralType::kEdgeLiteral) {
          util::Error("support only edge literal now!");
          return;
        }
        if (!stand_alone_literal_info_ret) {
          util::Error("the " + std::to_string(i) + "'th literal info is illegal");
          continue;
        }
        specified_rhs_literal_info_set.emplace_back(stand_alone_literal_info);
      }
      std::sort(specified_rhs_literal_info_set.begin(),
                specified_rhs_literal_info_set.end());
      assert(std::is_sorted(specified_rhs_literal_info_set.begin(),
                            specified_rhs_literal_info_set.end()));
    }

    std::vector<GarType> gar_list;
    // the set of rhs literals actually contained in the input gar set
    for (int i = 0; i < gar_path_set_config.size(); i++) {
      YAML::Node gar_path_config = gar_path_set_config[i];

      if (gar_path_config["VSetFile"]) {
        const std::string kGarPathVSetFile 
               = gar_path_config["VSetFile"].as<std::string>();

        if (!gar_path_config["ESetFile"]){
          util::Error("cannot get gar e set file");
          return;
        }
        const std::string kGarPathESetFile 
               = gar_path_config["ESetFile"].as<std::string>();

        if (!gar_path_config["XSetFile"]){
          util::Error("cannot get gar x set file");
          return;
        }
        const std::string kGarPathXSetFile 
               = gar_path_config["XSetFile"].as<std::string>();

        if (!gar_path_config["YSetFile"]){
          util::Error("cannot get gar y set file");
          return;
        }
        const std::string kGarPathYSetFile 
               = gar_path_config["YSetFile"].as<std::string>();

        std::vector<GarType> gar_set;
        auto read_gar_res = gar::ReadGARSet(
                            gar_set, kGarPathVSetFile, kGarPathESetFile,
                                     kGarPathXSetFile, kGarPathYSetFile);
        if (read_gar_res < 0) {
          util::Error("read gar error! v set file: " + kGarPathVSetFile
                                   + " e set file: " + kGarPathESetFile
                                   + " x set file: " + kGarPathXSetFile
                                   + " y set file: " + kGarPathYSetFile);
          return;
        }

        for (const auto& gar : gar_set) {
          if ( gar.y_literal_set().Count() != 1 ) {
            util::Error("can only process gar with one rhs literal");
            return;
          }
          assert(std::is_sorted(specified_rhs_literal_info_set.begin(),
                                specified_rhs_literal_info_set.end()));
          auto stand_alone_info = (*gar.y_literal_set().begin())->stand_alone_info();
          // to find whether this literal is contained in the specified_rhs_literal_info_set
          if (!specified_rhs_literal_info_set.empty() // has specified rhs literal
           && !std::binary_search(specified_rhs_literal_info_set.begin(), // but this literal is not contained in the set
                                  specified_rhs_literal_info_set.end(),
                                            stand_alone_info)) { 
                                    
            std::string stand_alone_info_str;
            stand_alone_info_str << stand_alone_info;

            util::Info("stand_alone_info_str: " 
                    +  stand_alone_info_str
                    + " is not contained in the specified rhs literal set");
            continue;
          }
          gar_list.emplace_back(gar);
        }
        continue;
      }

      if (!gar_path_config["VFile"]){
        util::Error("cannot get gar v file");
        return;
      }
      const std::string kGarPathVFile 
             = gar_path_config["VFile"].as<std::string>();

      if (!gar_path_config["EFile"]){
        util::Error("cannot get gar e file");
        return;
      }
      const std::string kGarPathEFile 
             = gar_path_config["EFile"].as<std::string>();

      if (!gar_path_config["XFile"]){
        util::Error("cannot get gar x file");
        return;
      }
      const std::string kGarPathXFile 
             = gar_path_config["XFile"].as<std::string>();

      if (!gar_path_config["YFile"]){
        util::Error("cannot get gar y file");
        return;
      }
      const std::string kGarPathYFile 
             = gar_path_config["YFile"].as<std::string>();

      GarType gar;
      auto read_gar_res = gar::ReadGAR(gar, kGarPathVFile, kGarPathEFile,
                                            kGarPathXFile, kGarPathYFile);

      if (read_gar_res < 0) {
        util::Error("read gar error! v file: " + kGarPathVFile
                                 + " e file: " + kGarPathEFile
                                 + " x file: " + kGarPathXFile
                                 + " y file: " + kGarPathYFile);
        return;
      }

      if ( gar.y_literal_set().Count() != 1 ) {
        util::Error("can only process gar with one rhs literal");
        return;
      }
      assert(std::is_sorted(specified_rhs_literal_info_set.begin(),
                            specified_rhs_literal_info_set.end()));
      auto stand_alone_info = (*gar.y_literal_set().begin())->stand_alone_info();
      // to find whether this literal is contained in the specified_rhs_literal_info_set
      if (!specified_rhs_literal_info_set.empty() // has specified rhs literal
       && !std::binary_search(specified_rhs_literal_info_set.begin(), // but this literal is not contained in the set
                              specified_rhs_literal_info_set.end(),
                                        stand_alone_info)) { 
                                
        std::string stand_alone_info_str;
        stand_alone_info_str << stand_alone_info;

        util::Info("stand_alone_info_str: " 
                 +  stand_alone_info_str
                 + " is not contained in the specified rhs literal set");
        continue;
      }
      gar_list.emplace_back(gar);
    }

    std::vector<std::tuple<DataGraphVertexHandleType,  // src vertex handle
                  std::vector<DataGraphEdgeLabelType>, // edge label
                           DataGraphVertexHandleType,  // dst vertex handle
                           bool, // whether exist edges, e.g. is positive or negative dataset 
                           bool> // whether can be chased by the gars
               > test_set;

    if (!test_set_file_path_path_set.empty()) {
      std::vector<DataGraphEdgeLabelType> edge_label_set;
      for (const auto& stand_alone_info : specified_rhs_literal_info_set) {
        edge_label_set.emplace_back(stand_alone_info.edge_label());
      }
      std::sort(edge_label_set.begin(),
                edge_label_set.end());

      edge_label_set.erase(
          std::unique(edge_label_set.begin(), 
                      edge_label_set.end()), 
                      edge_label_set.end());

      for (const auto& test_set_file_path 
                     : test_set_file_path_path_set) {
        std::ifstream test_set_file(test_set_file_path);
        if (!test_set_file.good()) {
          util::Error("test_set_file_path: " + test_set_file_path + " is not good");
          return;
        }
        constexpr size_t kLiteralVertexNum = 2; // edge literal
        while (test_set_file) {
          std::string s;
          if (!std::getline( test_set_file, s ))
            break;

          std::istringstream ss( s );
          std::vector <std::string> record;
          std::string buf; 
          while (ss >> buf)
            record.emplace_back(buf);

          std::vector <VertexIDType> support_id_set;
          for (const auto& id_str : record) {
            support_id_set.emplace_back(std::stoi(id_str));
          }
          if (kLiteralVertexNum != support_id_set.size()) {
            util::Error("support set miss match in line: " 
                       + std::to_string(test_set.size())
                       + " of file: " 
                       + test_set_file_path);
            return;
          }

          const auto& src_id = support_id_set[0];
          const auto& dst_id = support_id_set[1];

          const auto src_handle = data_graph.FindVertex(src_id);
          const auto dst_handle = data_graph.FindVertex(dst_id);

          if (!src_handle) {
            util::Error("vertex: " + std::to_string(src_id) 
                      + " cannot be found in data graph");
            return;
          }
          if (!dst_handle) {
            util::Error("vertex: " + std::to_string(dst_id) 
                      + " cannot be found in data graph");
            return;
          }

          bool is_positive = false;

          for (const auto& stand_alone_info : specified_rhs_literal_info_set) {
            if (src_handle->label() != stand_alone_info.x_label()) {
              continue;
            }
            if (dst_handle->label() != stand_alone_info.y_label()) {
              continue;
            }
            if (!GUNDAM::FindEdge<DataGraph>(src_handle,
                                             stand_alone_info.edge_label(),
                                             dst_handle)) {
              continue;
            }
            is_positive = true;
            break;
          }
          
          test_set.emplace_back(src_handle,
                               edge_label_set,
                                dst_handle,
                                is_positive,
                                false);
        }
      }
    }
    else {
      assert(specified_rhs_literal_info_set.size() == 1);
      LiteralStandAloneInfoType stand_alone_info = *specified_rhs_literal_info_set.begin();
      test_set.reserve(vertexes_pair_num);
      if (test_data_type == "positive") {
        _gar_accuracy_analysis::CollectPositiveEdgeSet(
                                data_graph,
                          stand_alone_info,
                          vertexes_pair_num, test_set);
      }
      else if (test_data_type == "negative") {
        _gar_accuracy_analysis::CollectNegativeOrBothEdgeSet(
                                data_graph,
                          stand_alone_info, true, // negative only
                          vertexes_pair_num, test_set);
      }
      else {
        assert(test_data_type == "both");
        _gar_accuracy_analysis::CollectNegativeOrBothEdgeSet(
                                data_graph,
                          stand_alone_info, false, // both positive and negative
                          vertexes_pair_num, test_set);
      }
    }

    util::Info("total gar to process: " + std::to_string(gar_list.size()));
    size_t gar_counter = 0;
    // for all gars 
    for (auto& gar : gar_list) {
      gar::LiteralInfo<Pattern, DataGraph> rhs_literal_info = (*gar.y_literal_set().begin())->info();
      assert(rhs_literal_info.literal_type() == gar::LiteralType::kEdgeLiteral);
      PatternVertexHandleType x_handle = gar.pattern().FindVertex(rhs_literal_info.x_id());
      assert(x_handle); // should have such a vertex
      PatternVertexHandleType y_handle = gar.pattern().FindVertex(rhs_literal_info.y_id());
      assert(y_handle); // should have such a vertex

      using CandidateSetContainer 
          = std::map<PatternVertexHandleType, 
       std::vector<DataGraphVertexHandleType>>;

      util::Info("processing gar: " + std::to_string(gar_counter));

      gar_counter++;

      CandidateSetContainer candidate_set;

      const bool kInitCandidateSetSucc 
        = GUNDAM::_dp_iso_using_match::InitCandidateSet<GUNDAM::MatchSemantics::kIsomorphism>(
          gar.pattern(), data_graph, candidate_set);
      if (!kInitCandidateSetSucc) {
        // assert(false);
        // should not be empty
        continue;
      }

      const bool kRefineCandidateSetSucc = GUNDAM::_dp_iso_using_match::RefineCandidateSet(
          gar.pattern(), data_graph, candidate_set);
      if (!kRefineCandidateSetSucc) {
        // assert(false);
        // should not be empty not
        continue;
      }

      #pragma omp parallel for schedule(dynamic) 
      for (int test_set_idx = 0;
               test_set_idx < test_set.size();
               test_set_idx++) {
        
        auto& [src_handle, edge_label_set, dst_handle,
               exist_in_data_graph,
               can_be_chased_by_gars] = test_set[test_set_idx];
        if (can_be_chased_by_gars) {
          // already can be chased by other gars
          continue;
        }
        // if can be match by the pattern without rhs, 
        // then it can be chased
        GUNDAM::Match<Pattern, DataGraph> partial_match;
        if (x_handle->label()  == src_handle->label()
         && std::binary_search(edge_label_set.begin(),
                               edge_label_set.end(),
                               rhs_literal_info.edge_label())
         && y_handle->label()  == dst_handle->label()) {
          bool add_map_ret_0 = partial_match.AddMap(x_handle, src_handle);
          assert(add_map_ret_0);
          bool add_map_ret_1 = partial_match.AddMap(y_handle, dst_handle);
          assert(add_map_ret_1);
        }
        else if (x_handle->label()  == dst_handle->label()
              && std::binary_search(edge_label_set.begin(),
                                    edge_label_set.end(),
                                    rhs_literal_info.edge_label())
              && y_handle->label()  == src_handle->label()) {
          bool add_map_ret_0 = partial_match.AddMap(x_handle, dst_handle);
          assert(add_map_ret_0);
          bool add_map_ret_1 = partial_match.AddMap(y_handle, src_handle);
          assert(add_map_ret_1);
        }
        else {
          // is not the rhs corresponds to the 
          assert(false);
          continue;
        }
        assert(partial_match.size() == 2);
        auto [x_supp, xy_supp] = gar::GarSupp(gar, data_graph,
                                                partial_match,
                                                candidate_set);
        assert( x_supp >= 0 &&  x_supp <= 1);
        assert(xy_supp >= 0 && xy_supp <= 1);
        assert(x_supp <= x_supp);
        assert(!can_be_chased_by_gars);
        if (x_supp == 1) {
          can_be_chased_by_gars = true;
        }
      }
    }

    output_log << "src_id,edge_label,dst_id,exist_in_data_graph,can_be_chased_by_gars"
               << std::endl;
    for (auto& [src_handle, edge_label_set, dst_handle,
                exist_in_data_graph,
                can_be_chased_by_gars] : test_set) {
      output_log << src_handle->id();
          

      if (edge_label_set.size() == 1) {
        output_log << "," << edge_label_set[0];
      }
      else {
        assert(!edge_label_set.empty());
        for (size_t edge_label_idx = 0;
                    edge_label_idx < edge_label_set.size();
                    edge_label_idx++) {
          output_log << ",";
          if (edge_label_idx == 0) {
            output_log << "{";
          }
          output_log << edge_label_set[edge_label_idx];
        }
        output_log << "}";
      }

      output_log << "," << dst_handle->id() 
                 << "," << exist_in_data_graph
                 << "," << can_be_chased_by_gars
                 << std::endl;
    }

    #ifdef PROFILING
    ctx.exec_time -= GetCurrentTime();
    #endif

    #ifdef PROFILING
    ctx.exec_time += GetCurrentTime();
    ctx.postprocess_time -= GetCurrentTime();
    #endif
    // messages.ForceContinue();

    #ifdef PROFILING
    ctx.postprocess_time += GetCurrentTime();
    #endif
    
    return;
  }

  /**
   * @brief Incremental evaluation for GarAccuracyAnalysis.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void IncEval(const fragment_t& frag, 
                      context_t& ctx,
              message_manager_t& messages) { 

    #ifdef PROFILING
    ctx.preprocess_time -= GetCurrentTime();
    #endif

    #ifdef PROFILING
    ctx.preprocess_time += GetCurrentTime();
    ctx.exec_time -= GetCurrentTime();
    #endif

    #ifdef PROFILING
    ctx.exec_time += GetCurrentTime();
    ctx.postprocess_time -= GetCurrentTime();
    #endif

    #ifdef PROFILING
    ctx.postprocess_time += GetCurrentTime();
    #endif

    return;
  }
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_GAR_ACCURACY_ANALYSIS_GAR_ACCURACY_ANALYSIS_H_