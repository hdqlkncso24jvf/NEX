#ifndef _EXAMPLES_ANALYTICAL_APPS_GRAPH_CONVERT_GRAPH_CONVERT_H_
#define _EXAMPLES_ANALYTICAL_APPS_GRAPH_CONVERT_GRAPH_CONVERT_H_

#include <grape/grape.h>
#include <omp.h>

#include <functional>

#include "gundam/io/csvgraph.h"

#include "gundam/graph_type/graph.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/large_graph2.h"
#include "gundam/graph_type/small_graph.h"

#include "gundam/match/match.h"
#include "gundam/match/matchresult.h"

#include "gundam/type_getter/edge_handle.h"
#include "gundam/type_getter/vertex_handle.h"

#include "gundam/algorithm/bfs.h"
#include "gundam/algorithm/match_using_match.h"

#include "gundam/tool/vertex_id_continued.h"
#include "gundam/tool/edge_id_continued.h"
#include "gundam/tool/isolate_vertex.h"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "graph_convert/graph_convert_context.h"

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

namespace _graph_convert{

template <class A, class B, typename AttrKeyPrefixType>
inline int CopyAllAttributesWithPrefix(A& from, B& to, AttrKeyPrefixType kAttrPrefix) {
  int count = 0;
  for (auto it_a = from->AttributeBegin(); !it_a.IsDone(); ++it_a) {
    switch (it_a->value_type()) {
      case GUNDAM::BasicDataType::kTypeInt:
        if (!to->template AddAttribute<int>(
                            kAttrPrefix + it_a->key(),
                                          it_a->template const_value<int>())
                 .second) {
          return -1;
        }
        break;
      case GUNDAM::BasicDataType::kTypeInt64:
        if (!to->template AddAttribute<int64_t>(
                            kAttrPrefix + it_a->key(),
                                          it_a->template const_value<int64_t>())
                 .second) {
          return -1;
        }
        break;
      case GUNDAM::BasicDataType::kTypeDouble:
        if (!to->template AddAttribute<double>(
                            kAttrPrefix + it_a->key(),
                                          it_a->template const_value<double>())
                 .second) {
          return -1;
        }
        break;
      case GUNDAM::BasicDataType::kTypeString:
        if (!to->template AddAttribute<std::string>(
                            kAttrPrefix + it_a->key(),
                                          it_a->template const_value<std::string>())
                 .second) {
          return -1;
        }
        break;
      case GUNDAM::BasicDataType::kTypeDateTime:
        if (!to->template AddAttribute<GUNDAM::DateTime>(
                            kAttrPrefix + it_a->key(),
                                          it_a->template const_value<GUNDAM::DateTime>())
                 .second) {
          return -1;
        }
        break;
      case GUNDAM::BasicDataType::kTypeUnknown:
      default:
        return -2;
    }
    ++count;
  }
  return count;
}

}; // namespace _graph_convert

template <typename FRAG_T>
class GraphConvert : public ParallelAppBase<FRAG_T, GraphConvertContext<FRAG_T>>,
                     public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
  using    VertexIDType = typename GraphConvertContext<FRAG_T>::   VertexIDType;
  using VertexLabelType = typename GraphConvertContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType = typename GraphConvertContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType = typename GraphConvertContext<FRAG_T>::  EdgeLabelType;
  using    DataGraph    = typename GraphConvertContext<FRAG_T>::DataGraph;
  using      Pattern    = typename GraphConvertContext<FRAG_T>::  Pattern;

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(GraphConvert<FRAG_T>, 
                          GraphConvertContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for GraphConvert.
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

    std::string convert_method = "no_edge_to_edge";
    if (config["ConvertMethod"]){
      convert_method = config["ConvertMethod"].as<std::string>();
      if (convert_method !=   "no_edge_to_edge"
       && convert_method != "with_edge_to_edge"
       && convert_method != "with_bidir_edge_to_edge"){
        std::cout<<"illegal convert method: "<<convert_method<<std::endl;
        return;
      }
    }

    bool output_vertex_edge_mark = false;
    if (config["OutputVertexEdgeMark"]) {
      output_vertex_edge_mark = config["OutputVertexEdgeMark"].as<bool>();
    }

    const std::string kIsVertexMark = "is_vertex_mark";

    const std::string kDataGraphPrefix 
                    = kDataGraphDir + "/" 
                    + kDataGraphName;

    const std::string kDataGraphPathVFile 
                    = kDataGraphPrefix + "_v.csv",
                      kDataGraphPathEFile 
                    = kDataGraphPrefix + "_e.csv";

    DataGraph data_graph;
    GUNDAM::ReadCSVGraph(data_graph, 
                        kDataGraphPathVFile,
                        kDataGraphPathEFile);

    if (!GUNDAM::VertexIDContinued(data_graph)){
      std::cout << "data_graph vertex id is not continued" << std::endl;
      // return;
    }
    bool original_graph_edge_id_connected = true;
    std::map<EdgeIDType, EdgeIDType> edge_id_map;
    // if(!GUNDAM::  EdgeIDContinued(data_graph)){
    //   std::cout << "data_graph edge id is not continued" << std::endl;
    //   original_graph_edge_id_connected = false;
    //   for (auto vertex_it = data_graph.VertexBegin();
    //            !vertex_it.IsDone();
    //             vertex_it++) {
    //     for (auto edge_it = vertex_it->OutEdgeBegin();
    //              !edge_it.IsDone();
    //               edge_it++) {
    //       auto [edge_id_it,
    //             edge_id_ret] = edge_id_map.emplace(edge_it->id(),
    //                                                edge_id_map.size());
    //       assert(edge_id_ret);
    //     }
    //   }
    // }
    if( GUNDAM::  HasIsolateVertex(data_graph)){
      std::cout << "data_graph has isolated vertex" << std::endl;
      // return;
    }

    // can convert graph with non-continued id
    // assert(GUNDAM::VertexIDContinued(data_graph));
    // assert(GUNDAM::  EdgeIDContinued(data_graph));

    VertexIDType max_vertex_id = 0;
    for (auto vertex_it = data_graph.VertexBegin();
             !vertex_it.IsDone();
              vertex_it++) {
      if (max_vertex_id < vertex_it->id()){
        max_vertex_id = vertex_it->id();
      }
    }

    EdgeIDType min_edge_id = std::numeric_limits<EdgeIDType>::max();
    for (auto vertex_cit = data_graph.VertexBegin();
             !vertex_cit.IsDone();
              vertex_cit++){
      for (auto edge_it = vertex_cit->OutEdgeBegin();
               !edge_it.IsDone();
                edge_it++){
        if (min_edge_id > edge_it->id()){
          min_edge_id = edge_it->id();
        }
      }
    }
    
    // new id for vertex converted from edge
    // new_vertex_id = edge_id + max_vertex_id - min_edge_id + 1

    DataGraph converted_graph;
    for (auto vertex_cit = data_graph.VertexBegin();
             !vertex_cit.IsDone();
              vertex_cit++){
      auto [vertex_handle,
            vertex_ret] = converted_graph.AddVertex(
                                             vertex_cit->id(),
                                             vertex_cit->label());
      // should added successfully
      assert(vertex_ret);
      GUNDAM::CopyAllAttributes(vertex_cit, vertex_handle);
      if (output_vertex_edge_mark){
        vertex_handle->AddAttribute(kIsVertexMark, 1);
      }
    }

    const EdgeLabelType kDefaultEdgeLabel = 0;

    GUNDAM::SimpleArithmeticIDGenerator<
        typename DataGraph::EdgeType::IDType> edge_id_gen;

    for (auto vertex_cit = data_graph.VertexBegin();
             !vertex_cit.IsDone();
              vertex_cit++){
      for (auto edge_it = vertex_cit->OutEdgeBegin();
               !edge_it.IsDone();
                edge_it++){
        assert (!original_graph_edge_id_connected
              ||(edge_id_map.find(edge_it->id()) != edge_id_map.end()));
        const VertexIDType kSrcVertexId = edge_it->src_handle()->id(),
                           kDstVertexId = edge_it->dst_handle()->id(),
                           kNewVertexId = original_graph_edge_id_connected?
                                              edge_it->id()
                                            + max_vertex_id
                                            - min_edge_id + 1
                                            : edge_id_map[edge_it->id()]
                                            + max_vertex_id
                                            - min_edge_id + 1;

        auto [vertex_handle, vertex_ret] 
         = converted_graph.AddVertex(kNewVertexId, edge_it->label());
        assert(vertex_ret);

        GUNDAM::CopyAllAttributes(edge_it, vertex_handle);
        if (output_vertex_edge_mark){
          vertex_handle->AddAttribute(kIsVertexMark, 0);
        }

        auto src_vertex_handle
           = data_graph.FindVertex(kSrcVertexId);
        assert(src_vertex_handle);

        auto dst_vertex_handle
           = data_graph.FindVertex(kDstVertexId);
        assert(dst_vertex_handle);

        _graph_convert::CopyAllAttributesWithPrefix(src_vertex_handle,
                                                        vertex_handle,
                                                   "src_attr_");

        _graph_convert::CopyAllAttributesWithPrefix(dst_vertex_handle,
                                                        vertex_handle,
                                                   "dst_attr_");

        converted_graph.AddEdge(kSrcVertexId, 
                                kDstVertexId,
                                kDefaultEdgeLabel,
                                edge_id_gen.GetID());
        converted_graph.AddEdge(kSrcVertexId, 
                                kNewVertexId,
                                kDefaultEdgeLabel,
                                edge_id_gen.GetID());
        converted_graph.AddEdge(kNewVertexId, 
                                kDstVertexId,
                                kDefaultEdgeLabel,
                                edge_id_gen.GetID());
        if (convert_method == "no_edge_to_edge"){
          continue;
        }
        assert(convert_method == "with_bidir_edge_to_edge"
            || convert_method == "with_edge_to_edge");
        // added edge between edges(in data_graph)
        for (auto src_in_edge_it = src_vertex_handle->InEdgeBegin();
                 !src_in_edge_it.IsDone();
                  src_in_edge_it++) {
          // src_in_edge_it should not point to the same edge as edge_it
          assert(src_in_edge_it->id() != edge_it->id());
          const VertexIDType kSrcInEdgeVertexId = src_in_edge_it->id()
                                                + max_vertex_id
                                                - min_edge_id + 1;
          auto src_out_edge_vertex_handle 
            = converted_graph.FindVertex(kSrcInEdgeVertexId);
          if (!src_out_edge_vertex_handle){
            // this edge is not added into converted_graph yet
            continue;
          }
          auto [add_edge_handle, 
                add_edge_ret] = converted_graph.AddEdge(kSrcInEdgeVertexId,
                                                              kNewVertexId,
                                                        kDefaultEdgeLabel,
                                                        edge_id_gen.GetID());
          assert(add_edge_ret);
        }
        for (auto dst_out_edge_it = dst_vertex_handle->OutEdgeBegin();
                 !dst_out_edge_it.IsDone();
                  dst_out_edge_it++) {
          // dst_out_edge_it should not point to the same edge as edge_it
          assert(dst_out_edge_it->id() != edge_it->id());
          const VertexIDType kDstOutEdgeVertexId = dst_out_edge_it->id()
                                                 + max_vertex_id
                                                 - min_edge_id + 1;
          auto dst_out_edge_vertex_handle 
            = converted_graph.FindVertex(kDstOutEdgeVertexId);
          if (!dst_out_edge_vertex_handle){
            // this edge is not added into converted_graph yet
            continue;
          }
          auto [add_edge_handle, 
                add_edge_ret] = converted_graph.AddEdge(kNewVertexId,
                                                 kDstOutEdgeVertexId,
                                                    kDefaultEdgeLabel,
                                                  edge_id_gen.GetID());
          assert(add_edge_ret);
        }
        if (convert_method == "with_edge_to_edge"){
          continue;
        }
        assert(convert_method == "with_bidir_edge_to_edge");
        for (auto src_out_edge_it = src_vertex_handle->OutEdgeBegin();
                 !src_out_edge_it.IsDone();
                  src_out_edge_it++) {
          if (src_out_edge_it->id() == edge_it->id()){
            continue;
          }
          const VertexIDType kSrcOutEdgeVertexId = src_out_edge_it->id()
                                                 + max_vertex_id
                                                 - min_edge_id + 1;
          auto src_out_edge_vertex_handle 
            = converted_graph.FindVertex(kSrcOutEdgeVertexId);
          if (!src_out_edge_vertex_handle){
            // this edge is not added into converted_graph yet
            continue;
          }
          auto [add_edge_handle, 
                add_edge_ret] = converted_graph.AddEdge(kSrcOutEdgeVertexId,
                                                               kNewVertexId,
                                                        kDefaultEdgeLabel,
                                                        edge_id_gen.GetID());
          assert(add_edge_ret);
        }
        for (auto dst_in_edge_it = dst_vertex_handle->InEdgeBegin();
                 !dst_in_edge_it.IsDone();
                  dst_in_edge_it++) {
          if (dst_in_edge_it->id() == edge_it->id()){
            continue;
          }
          const VertexIDType kDstInEdgeVertexId = dst_in_edge_it->id()
                                                + max_vertex_id
                                                - min_edge_id + 1;
          auto dst_in_edge_vertex_handle 
            = converted_graph.FindVertex(kDstInEdgeVertexId);
          if (!dst_in_edge_vertex_handle){
            // this edge is not added into converted_graph yet
            continue;
          }
          auto [add_edge_handle, 
                add_edge_ret] = converted_graph.AddEdge(kNewVertexId,
                                                  kDstInEdgeVertexId,
                                                    kDefaultEdgeLabel,
                                                  edge_id_gen.GetID());
          assert(add_edge_ret);
        }
      }
    }

    assert(converted_graph.CountVertex()
             == data_graph.CountVertex()
              + data_graph.CountEdge());

    if (!GUNDAM::VertexIDContinued(converted_graph)){
      std::cout << "converted vertex id is not continued" << std::endl;
      // return;
    }
    if(!GUNDAM::  EdgeIDContinued(converted_graph)){
      std::cout << "converted edge id is not continued" << std::endl;
      // return;
    }
    if( GUNDAM::  HasIsolateVertex(converted_graph)){
      std::cout << "converted has isolated vertex in converted graph" << std::endl;
      // return;
    }

    const std::string 
      kConvertedGraphVFile = kDataGraphPrefix 
                         + "_converted_" 
                         +   convert_method 
                         + "_v.csv",
      kConvertedGraphEFile = kDataGraphPrefix 
                         + "_converted_" 
                         +   convert_method 
                         + "_e.csv";

    GUNDAM::WriteCSVGraph(converted_graph, 
                         kConvertedGraphVFile,
                         kConvertedGraphEFile);

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
   * @brief Incremental evaluation for GraphConvert.
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

#endif  // EXAMPLES_ANALYTICAL_APPS_GRAPH_CONVERT_GRAPH_CONVERT_H_