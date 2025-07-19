#ifndef EXAMPLES_ANALYTICAL_APPS_GOR_ACCURACY_ANALYSIS_GOR_ACCURACY_ANALYSIS_H_
#define EXAMPLES_ANALYTICAL_APPS_GOR_ACCURACY_ANALYSIS_GOR_ACCURACY_ANALYSIS_H_

#include <grape/grape.h>
#include <omp.h>

#include <functional>

#include "gundam/io/csvgraph.h"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "gor_accuracy_analysis/gor_accuracy_analysis_context.h"
#include "gar_accuracy_analysis/gar_accuracy_analysis.h"

#include "gar/gar_config.h"
#include "gar/gar_supp.h"
#include "gar/literal_stand_alone_info.h"

#include "gor/csv_gor.h"
#include "gor/gor.h"
#include "gor/gor_match.h"

#include "util/log.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../timer.h"

namespace grape {

template <typename FRAG_T>
class GorAccuracyAnalysis : public ParallelAppBase<FRAG_T, GorAccuracyAnalysisContext<FRAG_T>>,
                            public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
  using    VertexIDType = typename GorAccuracyAnalysisContext<FRAG_T>::   VertexIDType;
  using VertexLabelType = typename GorAccuracyAnalysisContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType = typename GorAccuracyAnalysisContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType = typename GorAccuracyAnalysisContext<FRAG_T>::  EdgeLabelType;
  using    DataGraph    = typename GorAccuracyAnalysisContext<FRAG_T>::  DataGraph;
  using      Pattern    = typename GorAccuracyAnalysisContext<FRAG_T>::    Pattern;

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(GorAccuracyAnalysis<FRAG_T>, 
                          GorAccuracyAnalysisContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for GorAccuracyAnalysis.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void PEval(const fragment_t& frag, 
                    context_t& ctx,
            message_manager_t& messages) {

    util::Info("yaml file: " + ctx.yaml_file_);

    using GorType = gor::GraphOracleRule<Pattern, DataGraph>;
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
    
    if (!config["OutputLogFile"]){
      util::Error("does not specify output log file!");
      return;
    }
    const std::string kOutputLogFile 
             = config["OutputLogFile"].as<std::string>();

    if (!config["VertexesPairNum"]){
      util::Error("does not specify vertexes pair number!");
      return;
    }
    const size_t kVertexesPairNum
        = config["VertexesPairNum"].as<size_t>();

    std::string test_data_type = "both";
    if (config["TestDataType"]){
      test_data_type = config["TestDataType"].as<std::string>();
      if (test_data_type != "positive"
       && test_data_type != "negative"
       && test_data_type != "both") {
        util::Error("unknown TestDataType: " + test_data_type);
        return;
      }
    }

    std::ofstream output_log(kOutputLogFile);
    if (!output_log) {
      util::Error("open file fail!" + kOutputLogFile);
      return;
    }

    std::vector<std::pair<GorType, VertexIDType>> gor_list;

    if (!config["GORPath"]) {
      util::Error("cannot get gor dirs!");
      return;
    }
    YAML::Node gor_path_set_config = config["GORPath"];

    for (int i = 0; i < gor_path_set_config.size(); i++){
      YAML::Node gor_path_config = gor_path_set_config[i];

      if (!gor_path_config["VFile"]){
        util::Error("cannot get gor v file");
        return;
      }
      const std::string kGorPathVFile 
             = gor_path_config["VFile"].as<std::string>();

      if (!gor_path_config["EFile"]){
        util::Error("cannot get gor e file");
        return;
      }
      const std::string kGorPathEFile 
             = gor_path_config["EFile"].as<std::string>();

      if (!gor_path_config["XFile"]){
        util::Error("cannot get gor x file");
        return;
      }
      const std::string kGorPathXFile 
             = gor_path_config["XFile"].as<std::string>();

      if (!gor_path_config["YFile"]){
        util::Error("cannot get gor y file");
        return;
      }
      const std::string kGorPathYFile 
             = gor_path_config["YFile"].as<std::string>();

      GorType gor;
      gor::ReadGOR(gor, kGorPathVFile, kGorPathEFile,
                        kGorPathXFile, kGorPathYFile);
      if (!gor.x_literal_set().Empty()
        || gor.y_literal_set().Count() != 1) {
        util::Error("can only process gor with empty lhs literal set and one rhs literal");
        return;
      }
      if ((*gor.y_literal_set().begin())->type() != gar::LiteralType::kEdgeLiteral) {
        util::Error("can only process gor with rhs edge literal");
        return;
      }
      if (!gor_list.empty()) {
        // to find whether it is 
        auto stand_alone_info  = (*(gor_list.begin()
                                ->first.y_literal_set().begin()))->stand_alone_info();
        if ( stand_alone_info != ( *gor.y_literal_set().begin() )->stand_alone_info()) {
          std::string stand_alone_info_str;
          stand_alone_info_str << stand_alone_info;

          std::string gor_stand_alone_info_str;
          gor_stand_alone_info_str << (*gor.y_literal_set().begin())->stand_alone_info();

          util::Error("stand_alone_info_str: " 
                    +  stand_alone_info_str
                    + "gor_stand_alone_info_str: "
                    +  gor_stand_alone_info_str);

          util::Error("can only process the gor set with same rhs literal");
          return;
        }
      }
      VertexIDType pivot_id;

      if (gor_path_config["PivotId"]){
        const VertexIDType kGorPivotId
            = gor_path_config["PivotId"].as<VertexIDType>();

        Pattern rhs_pattern(gar::_gar_supp::LiteralPattern(gor, true));

        if (!rhs_pattern.FindVertex(kGorPivotId)) {
          // cannot find the pivot vertex in rhs pattern
          std::string rhs_pattern_vertex_set;
          for (auto vertex_it = rhs_pattern.VertexBegin();
                   !vertex_it.IsDone();
                    vertex_it++) {
            rhs_pattern_vertex_set += " " + GUNDAM::ToString(vertex_it->id());
          }
          util::Error("cannot find the pivot vertex with id: " + GUNDAM::ToString(kGorPivotId)
                      + " in rhs pattern contains of vertex: " + rhs_pattern_vertex_set);
          return;
        }
        //  gor.pattern().FindVertex(kGorPivotId)
        // != rhs_pattern.FindVertex(kGorPivotId)
        // need to find it from graph pattern again
        pivot_id = kGorPivotId;
      }
      else {
        // randomly select one from rhs_pattern
        Pattern rhs_pattern(gar::_gar_supp::LiteralPattern(gor,  true));

        std::vector<VertexIDType> rhs_gor_vertex_id_set;
        for (auto vertex_it = rhs_pattern.VertexBegin();
                !vertex_it.IsDone();
                  vertex_it++) {
          rhs_gor_vertex_id_set.emplace_back(vertex_it->id());
        }
        std::sort(rhs_gor_vertex_id_set.begin(),
                  rhs_gor_vertex_id_set.end());
        assert(!rhs_gor_vertex_id_set.empty());

        pivot_id = *rhs_gor_vertex_id_set.begin();
      }
      gor_list.emplace_back(gor, pivot_id);
    }

    util::Debug(" here0!");

    LiteralStandAloneInfoType stand_alone_info 
      = (*(gor_list.begin()->first.y_literal_set().begin()))->stand_alone_info();

    util::Debug(" here1!");

    std::vector<std::tuple<DataGraphVertexHandleType,  // src vertex handle
                  std::vector<DataGraphEdgeLabelType>, // edge label
                           DataGraphVertexHandleType,  // dst vertex handle
                           bool, // whether exist edges, e.g. is positive or negative dataset 
                           bool> // whether can be chased by the gars
               > test_set;

    if (test_data_type == "positive") {
      _gar_accuracy_analysis::CollectPositiveEdgeSet(
                              data_graph,
                        stand_alone_info,
                        kVertexesPairNum, test_set);
    }
    else if (test_data_type == "negative") {
      _gar_accuracy_analysis::CollectNegativeOrBothEdgeSet(
                              data_graph,
                        stand_alone_info, true, // negative only
                        kVertexesPairNum, test_set);
    }
    else {
      assert(test_data_type == "both");
      _gar_accuracy_analysis::CollectNegativeOrBothEdgeSet(
                              data_graph,
                        stand_alone_info, false, // both positive and negative
                        kVertexesPairNum, test_set);
    }

    std::  map <DataGraphVertexHandleType, 
    std::vector<DataGraphVertexHandleType>> possible_match_set;

    // for all gors 
    for (auto& [gor, pivot_id] : gor_list) {
      util::Debug("##########");
      auto pivot = gor.pattern().FindVertex(pivot_id);
      assert(pivot);

      gar::LiteralInfo<Pattern, DataGraph> rhs_literal_info = (*gor.y_literal_set().begin())->info();
      assert(rhs_literal_info.literal_type() == gar::LiteralType::kEdgeLiteral);
      PatternVertexHandleType x_handle = gor.pattern().FindVertex(rhs_literal_info.x_id());
      assert(x_handle);
      PatternVertexHandleType y_handle = gor.pattern().FindVertex(rhs_literal_info.y_id());
      assert(y_handle);

      using CandidateSetContainer 
          = std::map<PatternVertexHandleType, 
       std::vector<DataGraphVertexHandleType>>;

      std::vector<CandidateSetContainer> match_result_set;

      util::Info("## gor match begin ##");
      gor::GORMatch(gor, pivot, data_graph, match_result_set);
      util::Info("## gor match end ##");

      for (auto& match_result : match_result_set) {
        util::Debug("match_result.size(): " + std::to_string(match_result.size()));
        util::Debug("gor.pattern().CountVertex(): " + std::to_string(gor.pattern().CountVertex()));
        util::Debug("x_handle->id(): " + x_handle->id());
        assert(match_result.find(x_handle) != match_result.end());
        assert(match_result.find(y_handle) != match_result.end());
        // it would be unnecessary to sort match set of x_handle
        // std::sort(match_result[x_handle].begin(),
        //           match_result[x_handle].end());
        std::sort(match_result[y_handle].begin(),
                  match_result[y_handle].end());
        for (const auto& x_match_handle : match_result[x_handle]) {
          const auto x_possible_match_set_size
                     = possible_match_set[x_match_handle].size();
          std::copy (match_result[y_handle].begin(), 
                     match_result[y_handle].end(), 
                     std::back_inserter(possible_match_set[x_match_handle]));
          assert(possible_match_set[x_match_handle].size()
            == x_possible_match_set_size 
                        + match_result[y_handle].size());
          std::inplace_merge (possible_match_set[x_match_handle].begin(),
                              possible_match_set[x_match_handle].begin()
                          + x_possible_match_set_size,
                              possible_match_set[x_match_handle].end());

          auto unique_it = std::unique(possible_match_set[x_match_handle].begin(), 
                                       possible_match_set[x_match_handle].end());
          possible_match_set[x_match_handle].erase(unique_it, possible_match_set[x_match_handle].end());
        }
      }
    }

    for (auto& [src_handle, 
                edge_label, 
                dst_handle,
                exist_in_data_graph,
                can_be_chased_by_gars] : test_set) {
      auto src_possible_match_it
             = possible_match_set.find(src_handle);
      if ( src_possible_match_it
            == possible_match_set.end()){
        continue;
      }
      assert(!can_be_chased_by_gars);
      if (!std::binary_search(src_possible_match_it->second.begin(),
                              src_possible_match_it->second. end (),
                              dst_handle)) {
        continue;
      }
      can_be_chased_by_gars = true;
    }

    output_log << "src_id,edge_label,dst_id,exist_in_data_graph,can_be_chased_by_gars"
               << std::endl;
    for (auto& [src_handle, edge_label, dst_handle,
                exist_in_data_graph,
                can_be_chased_by_gars] : test_set) {
      assert(edge_label.size() == 1);
      output_log << src_handle->id()
          << "," << *(edge_label.begin())
          << "," << dst_handle->id() 
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
   * @brief Incremental evaluation for GorAccuracyAnalysis.
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

#endif  // EXAMPLES_ANALYTICAL_APPS_GOR_ACCURACY_ANALYSIS_GOR_ACCURACY_ANALYSIS_H_