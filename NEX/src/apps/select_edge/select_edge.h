#ifndef _EXAMPLES_ANALYTICAL_APPS_SELECT_EDGE_SELECT_EDGE_H_
#define _EXAMPLES_ANALYTICAL_APPS_SELECT_EDGE_SELECT_EDGE_H_

#include <grape/grape.h>
#include <omp.h>

#include <functional>

#include "gundam/io/csvgraph.h"

#include "gundam/graph_type/graph.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/large_graph2.h"
#include "gundam/graph_type/small_graph.h"

#include "gundam/type_getter/edge_handle.h"
#include "gundam/type_getter/vertex_handle.h"

#include "gundam/sample/edge_sample.h"
#include "gundam/sample/vertex_sample.h"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "select_edge/select_edge_context.h"

#include "include/util/yaml_config.h"
#include "include/util/log.h"

#include "gar/csv_gar.h"
#include "gar/gar.h"
#include "gar/gar_supp.h"
#include "gar/gar_match.h"
#include "gar/gar_config.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../timer.h"

namespace grape {

template <typename FRAG_T>
class SelectEdge : public ParallelAppBase<FRAG_T, SelectEdgeContext<FRAG_T>>,
                   public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
  using    VertexIDType = typename SelectEdgeContext<FRAG_T>::   VertexIDType;
  using VertexLabelType = typename SelectEdgeContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType = typename SelectEdgeContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType = typename SelectEdgeContext<FRAG_T>::  EdgeLabelType;
  using    DataGraph    = typename SelectEdgeContext<FRAG_T>::   DataGraph;
  using GraphPattern    = typename SelectEdgeContext<FRAG_T>::GraphPattern;

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(SelectEdge<FRAG_T>, 
                          SelectEdgeContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for SelectEdge.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void PEval(const fragment_t& frag, 
                    context_t& ctx,
            message_manager_t& messages) {
    
    messages.InitChannels(thread_num());
    auto begin = timer();
    timer_next("build index");

    std::cout<<"yaml file: "<<ctx.yaml_file_<<std::endl;

    YAML::Node config = YAML::LoadFile(ctx.yaml_file_);

    if (!config["DataGraphPath"]) {
      std::cout << "cannot get data graph path" << std::endl;
      return;
    }
    YAML::Node data_graph_path_config = config["DataGraphPath"];

    if (!data_graph_path_config["GraphDir"]){
      std::cout<<"the dir of the data graph has not been been configured"<<std::endl;
      return;
    }
    const std::string kDataGraphDir
      = data_graph_path_config["GraphDir"].as<std::string>();

    if (!data_graph_path_config["GraphName"]){
      std::cout<<"the name of the data graph has not been been configured"<<std::endl;
      return;
    }
    const std::string kDataGraphName
      = data_graph_path_config["GraphName"].as<std::string>();

    const std::string kDataGraphPrefix 
                    = kDataGraphDir + "/" 
                    + kDataGraphName;

    const std::string kDataGraphPathVFile 
                    = kDataGraphPrefix + "_v.csv",
                      kDataGraphPathEFile 
                    = kDataGraphPrefix + "_e.csv";

    using GarType = gar::GraphAssociationRule<GraphPattern, DataGraph>;

    using EdgeTypeType = std::tuple<VertexLabelType,
                                      EdgeLabelType,
                                    VertexLabelType>;

    using LiteralStandAloneInfoType = gar::LiteralStandAloneInfo<GraphPattern,
                                                                    DataGraph>;

    std::set<EdgeTypeType> specified_edge_type_set;

    if (config["SpecifiedEdgeTypeSet"]) {
      YAML::Node specified_edge_type_set_config
       = config["SpecifiedEdgeTypeSet"];
      for(int i=0; i < specified_edge_type_set_config.size(); ++i){
        YAML::Node specified_edge_type_config
                 = specified_edge_type_set_config[i];
                 
        auto [ specified_edge_type, 
               specified_edge_type_ret ] 
          = util::EdgeTypeFromYaml<DataGraph>(specified_edge_type_config);
        if (!specified_edge_type_ret){
          std::cout << "the " << i << "'th edge type is illegal!" << std::endl;
          continue;
        }

        auto [specified_edge_type_set_it,
              specified_edge_type_set_ret ]
            = specified_edge_type_set.emplace(specified_edge_type);        
        if (! specified_edge_type_set_ret){
          std::cout << "duplicate edge type:" << std::endl
                    << "\t src label: " << std::get<0>(specified_edge_type) << std::endl
                    << "\tedge label: " << std::get<1>(specified_edge_type) << std::endl
                    << "\t dst label: " << std::get<2>(specified_edge_type) << std::endl;
          return;
        }
      } 
      if (specified_edge_type_set.empty()){
        std::cout << "specified empty edge type" << std::endl;
        return;
      }
    }

    if (!config["SortedEdgeTypeSet"]) {
      std::cout << "does not denote sorted edge type set"
                << std::endl;
      return;
    }
    YAML::Node all_edge_type_set_config = config["SortedEdgeTypeSet"];

    std::vector<std::vector<EdgeTypeType>> grouped_edge_type_seq;

    for (size_t i = 0; i < all_edge_type_set_config.size(); i++) {
      std::vector<EdgeTypeType> edge_type_seq;
      YAML::Node edge_type_set_config = all_edge_type_set_config[i]["Group"]?
                                        all_edge_type_set_config[i]["Group"]
                                      : all_edge_type_set_config;
      assert(all_edge_type_set_config[i]["Group"] 
          || all_edge_type_set_config.size() == 1);
      for (size_t j = 0; j < edge_type_set_config.size(); j++) {
        const auto& edge_type_config = edge_type_set_config[j];
        auto [ edge_type, edge_type_ret ] = util::EdgeTypeFromYaml<DataGraph>(edge_type_config);
        if (!edge_type_ret){
          std::cout << "the " << j << "'th edge type is illegal!" << std::endl;
          continue;
        }
        bool existed = false;
        for (const auto& added_edge_type : edge_type_seq) {
          if (added_edge_type == edge_type){
            existed = true;
            break;
          }
        }
        if (existed){
          continue;
        }
        if (!specified_edge_type_set.empty()) {
          // has specified edge types, to find whether
          // this is contained in 
          if (specified_edge_type_set.find(edge_type)
          == specified_edge_type_set.end()) {
            // is not contained in the specified_edge_type_set
            continue;
          }
        }
        edge_type_seq.emplace_back(edge_type);
      }
      grouped_edge_type_seq.emplace_back(std::move(edge_type_seq));
      if (!all_edge_type_set_config[i]["Group"]) {
        break;
      }
    }

    std::vector<EdgeTypeType> all_edge_type_seq;

    std::set<EdgeTypeType> added_edge_type_seq;

    util::Info("# grouped_edge_type_seq.size(): "
            + std::to_string(
                  grouped_edge_type_seq.size())
            + " #");

    for (const auto& edge_type_seq : grouped_edge_type_seq) {
      util::Info("\t" + std::to_string(edge_type_seq.size()));
    }

    while (true) {
      bool all_empty = true;
      for (auto& edge_type_seq : grouped_edge_type_seq) {
        if (edge_type_seq.empty()) {
          continue;
        }
        all_empty = false;
        if (added_edge_type_seq.emplace(edge_type_seq.front()).second) {
          // has been added succesfully
          // has not been added before
          all_edge_type_seq.emplace_back(edge_type_seq.front());
        }
        edge_type_seq.erase(edge_type_seq.begin());
      }
      if (all_empty) {
        break;
      }
    }
    assert(added_edge_type_seq.size() 
          == all_edge_type_seq.size());

    util::Info("# all_edge_type_seq.size(): "
            + std::to_string(
                  all_edge_type_seq.size())
            + " #");

    added_edge_type_seq.clear();

    std::set<int> preserve_edge_type_number;
    if (!config["EdgeTypeNumber"]) {
      std::cout << "does not specify how many edge types to select"
                << std::endl;
      return;
    }
    YAML::Node preserve_edge_type_config = config["EdgeTypeNumber"];

    for (size_t i = 0; i < preserve_edge_type_config.size(); i++){
      int preseve_num = preserve_edge_type_config[i].as<int>();
      auto [preserve_edge_type_number_it,
            preserve_edge_type_number_ret]
          = preserve_edge_type_number.emplace(preseve_num);
      if (! preserve_edge_type_number_ret){
        std::cout << "duplicate edge type number: " << std::endl;
        return;
      }
    }

    std::string additional_name = "";
    if (config["AdditionalName"]) {
      additional_name = config["AdditionalName"].as<std::string>();
    }

    DataGraph data_graph;
    GUNDAM::ReadCSVGraph(data_graph, 
                        kDataGraphPathVFile,
                        kDataGraphPathEFile);

    for (auto preserve_num_it  = preserve_edge_type_number.rbegin();
              preserve_num_it != preserve_edge_type_number.rend();
              preserve_num_it++) {
      if (all_edge_type_seq.size() < *preserve_num_it) {
        continue;
      }
      std::set<EdgeTypeType> preserve_edge_type_set;
      for (size_t idx = 0; idx < *preserve_num_it; idx ++){
        assert(idx < all_edge_type_seq.size());
        auto [ preserve_edge_type_set_it,
               preserve_edge_type_set_ret ]
             = preserve_edge_type_set.emplace(all_edge_type_seq[idx]);
        // should added successfully
        assert(preserve_edge_type_set_ret);
      }
      assert(preserve_edge_type_set.size() 
         == *preserve_num_it);
      std::vector<EdgeIDType> edge_id_to_remove;
      for (auto vertex_it = data_graph.VertexBegin();
               !vertex_it.IsDone();
                vertex_it++){
        for (auto edge_it = vertex_it->OutEdgeBegin();
                 !edge_it.IsDone();
                  edge_it++){
          EdgeTypeType edge_type(edge_it->src_handle()->label(),
                                 edge_it->label(),
                                 edge_it->dst_handle()->label());
          if (preserve_edge_type_set.find(edge_type)
           == preserve_edge_type_set.end()){
            // does not preserve this edge type, this edge needs
            // to be removed
            edge_id_to_remove.emplace_back(edge_it->id());
            continue;
          }
        }
      }
      for (const auto& edge_id : edge_id_to_remove) {
        auto ret = data_graph.EraseEdge(edge_id);
        assert(ret > 0);
      }
      GUNDAM::RemoveIsolateVertex(data_graph);
      const std::string kDataGraphPathVFile 
                      = kDataGraphPrefix 
                      + "_" + additional_name 
                      + "_edge_type_" + std::to_string(*preserve_num_it) 
                      + "_v.csv",
                        kDataGraphPathEFile 
                      = kDataGraphPrefix 
                      + "_" + additional_name 
                      + "_edge_type_" + std::to_string(*preserve_num_it) 
                      + "_e.csv";
      GUNDAM::WriteCSVGraph(data_graph, kDataGraphPathVFile,  
                                        kDataGraphPathEFile);
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
  }

  /**
   * @brief Incremental evaluation for SelectEdge.
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

#endif  // EXAMPLES_ANALYTICAL_APPS_SELECT_EDGE_SELECT_EDGE_H_