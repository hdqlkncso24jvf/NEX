#ifndef _EXAMPLES_ANALYTICAL_APPS_SUPPORT_SAMPLE_SUPPORT_SAMPLE_H_
#define _EXAMPLES_ANALYTICAL_APPS_SUPPORT_SAMPLE_SUPPORT_SAMPLE_H_

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
#include "gundam/algorithm/bfs_limit_width.h"
#include "gundam/algorithm/match_using_match.h"

#include "gundam/tool/operator/merge_by_id.h"
#include "gundam/tool/k_hop.h"
#include "gundam/tool/random_walk/random_walk.h"
#include "gundam/tool/same_pattern.h"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "support_sample/support_sample_context.h"

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

namespace _support_sample{

  using SupportIdType = int64_t; 

  using SupportIdSetType = std::set<SupportIdType>;

  using SubstructureNumberType = uint64_t;

  using SupportPatternIdxType = int;

  template <typename HandleType>
  inline void AddUncoveredSubstructure(
                     HandleType handle, //  vertex handle/iterator 
                                        // or edge handle/iterator
                  SupportIdType kSupportId){
    using AttrKeyType = std::remove_reference_t<decltype(handle->AttributeBegin()->key())>;
    const AttrKeyType kUncoveredSupportIdSetKey = "uncovered_support_id_set";
    auto [attr_handle, ret] = handle->AddAttribute(kUncoveredSupportIdSetKey,
                                                    SupportIdSetType());
    assert(!attr_handle.IsNull());
    attr_handle->template value<SupportIdSetType>().emplace(kSupportId);
    return;
  }

  size_t SelectDenotedSupportIdSet(
       const std::vector<
             std::set<SupportIdType>>& multi_supp_denoted_support_id_set, 
                  std::vector<size_t>& multi_supp_added_denoted_size,
                    const std::string& multi_supp_select_method){
    assert(multi_supp_select_method == "same_number");
    size_t min_id = 0;
    bool has_set_min_id = false;

    std::cout << "multi_supp_denoted_support_id_set:" << std::endl;
    for (const auto& denoted_support_id_set
        : multi_supp_denoted_support_id_set){
      std::cout << "\t" << denoted_support_id_set.size();
    }
    std::cout << std::endl;

    std::cout << "multi_supp_added_denoted_size:" << std::endl;
    for (const auto& added_denoted_size
        : multi_supp_added_denoted_size){
      std::cout << "\t" << added_denoted_size;
    }
    std::cout << std::endl;

    for (size_t i = 0; i < multi_supp_denoted_support_id_set.size(); i++){
      if (multi_supp_denoted_support_id_set[i].empty())
        continue;
      if ((has_set_min_id)
        &&(multi_supp_added_denoted_size[i]
        >= multi_supp_added_denoted_size[min_id])){
        continue;
      }
      min_id = i;
      has_set_min_id = true;
    }
    std::cout << "min_id: " << min_id << std::endl;
    return min_id;
  }

  size_t SelectDenotedSupportIdSet(
              const std::vector<bool>& multi_supp_all_supp_added, 
                  std::vector<size_t>& multi_supp_added_size,
                    const std::string& multi_supp_select_method){
    assert(multi_supp_select_method == "same_number");
    size_t min_id = 0;
    bool has_set_min_id = false;

    std::cout << "multi_supp_all_supp_added:" << std::endl;
    for (const auto& all_supp_added
        : multi_supp_all_supp_added){
      std::cout << "\t" << all_supp_added;
    }
    std::cout << std::endl;

    std::cout << "multi_supp_added_size:" << std::endl;
    for (const auto& added_size
        : multi_supp_added_size){
      std::cout << "\t" << added_size;
    }
    std::cout << std::endl;

    for (size_t i = 0; i < multi_supp_all_supp_added.size(); i++){
      if (multi_supp_all_supp_added[i])
        continue;
      if ((has_set_min_id)
        &&(multi_supp_added_size[i]
        >= multi_supp_added_size[min_id])){
        continue;
      }
      min_id = i;
      has_set_min_id = true;
    }
    std::cout << "min_id: " << min_id << std::endl;
    return min_id;
  }

  template <typename HandleType>
  inline void RemoveUncoveredNum(
                     HandleType handle, //  vertex handle/iterator 
                                        // or edge handle/iterator
         std::map<SupportIdType,
         SubstructureNumberType>& support_uncovered_size){
    using AttrKeyType = std::remove_reference_t<decltype(handle->AttributeBegin()->key())>;
    const AttrKeyType kUncoveredSupportIdSetKey = "uncovered_support_id_set";
    auto [attr_handle, ret] = handle->AddAttribute(kUncoveredSupportIdSetKey,
                                                    SupportIdSetType());
    assert(!attr_handle.IsNull()); // should not be null
    // std::cout << "remove id: "<< std::endl;
    for (auto support_id_it  = attr_handle->template value<SupportIdSetType>().begin();
              support_id_it != attr_handle->template value<SupportIdSetType>().end();){
      auto ret_it = support_uncovered_size.find(*support_id_it);
      assert(ret_it != support_uncovered_size.end());
      assert(ret_it->second > 0);
      ret_it->second = ret_it->second - 1;
      // std::cout << "\t" << *support_id_it << "-" << ret_it->second;
      support_id_it = attr_handle->template value<SupportIdSetType>().erase(support_id_it);
    }
    assert(attr_handle->template value<SupportIdSetType>().empty());
    handle->EraseAttribute(attr_handle->key());
    // std::cout << std::endl;
    return;
  }

  template <typename          DataGraphType,
            typename ObtainSubstructureType>
  inline SubstructureNumberType UncoveredSubstructureSize(
                        DataGraphType& data_graph,
                        DataGraphType& current_fragment,
  const std::vector<typename GUNDAM::VertexHandle<DataGraphType>::type>& supp_match,
                        SupportIdType  kSupportId, 
               ObtainSubstructureType obtain_substructure) {

    using DataGraphVertexHandle = typename GUNDAM::VertexHandle<DataGraphType>::type;

    using DataGraphEdgeIDType = typename DataGraphType::EdgeType::IDType;

    using AttrKeyType = std::remove_reference_t<decltype(data_graph.VertexBegin()
                                                               ->AttributeBegin()
                                                               ->key())>;

    std::set<DataGraphVertexHandle> 
      substructure_visited_vertex_handle_set;

    std::set<DataGraphEdgeIDType> substructure_visited_edge_id_set;

    SubstructureNumberType uncovered_vertex_edge_num = 0;

    auto substructure_callback 
     = [&substructure_visited_vertex_handle_set,
        &substructure_visited_edge_id_set,
        &current_fragment,
        &uncovered_vertex_edge_num,
        &kSupportId](
          DataGraphVertexHandle vertex_handle){

      const AttrKeyType kSubstructureIdSetKey = "substructure_support_id_set";
      
      auto [attr_handle, ret] = vertex_handle->AddAttribute(kSubstructureIdSetKey,
                                                             SupportIdSetType());
      assert(!attr_handle.IsNull());
      auto [added_support_id_it,
            added_support_id_ret] = attr_handle->template value<SupportIdSetType>().emplace(kSupportId);
      assert(added_support_id_ret);

      auto [substructure_visited_vertex_it, 
            substructure_visited_vertex_ret]
          = substructure_visited_vertex_handle_set.emplace(vertex_handle);
      assert(substructure_visited_vertex_ret);
      
      auto current_fragment_vertex_handle
         = current_fragment.FindVertex(vertex_handle->id());
      if (!current_fragment_vertex_handle){
        // this vertex is not considered in current_fragment
        // would be added into support_uncovered_size
        uncovered_vertex_edge_num++;
        AddUncoveredSubstructure(vertex_handle, kSupportId);
      }

      for (auto out_edge_it = vertex_handle->OutEdgeBegin();
               !out_edge_it.IsDone(); 
                out_edge_it++) {
        auto substructure_visited_it
           = substructure_visited_vertex_handle_set.find(out_edge_it->dst_handle());
        if ( substructure_visited_it 
          == substructure_visited_vertex_handle_set.end()) {
          // the vertex does not exist in the substructure, 
          continue;
        }

        auto [substructure_visited_edge_id_it,
              substructure_visited_edge_id_ret] 
            = substructure_visited_edge_id_set.emplace(out_edge_it->id());
        if ( !substructure_visited_edge_id_ret){
          // this edge has already been considered
          continue;
        }
        auto current_fragment_edge_handle
           = current_fragment.FindEdge(out_edge_it->id());
        if (!current_fragment_edge_handle){
          // this edge is not contained in current_fragment
          uncovered_vertex_edge_num++;
          AddUncoveredSubstructure(out_edge_it, kSupportId);
        }
      }

      for (auto in_edge_it = vertex_handle->InEdgeBegin();
               !in_edge_it.IsDone(); 
                in_edge_it++) {
        auto substructure_visited_it
           = substructure_visited_vertex_handle_set.find(in_edge_it->src_handle());
        if ( substructure_visited_it 
          == substructure_visited_vertex_handle_set.end()) {
          // the vertex does not exist in the substructure, 
          continue;
        }

        auto [substructure_visited_edge_id_it,
              substructure_visited_edge_id_ret] 
            = substructure_visited_edge_id_set.emplace(in_edge_it->id());
        if (!substructure_visited_edge_id_ret){
          // this edge has already been considered
          continue;
        }
        auto current_fragment_edge_handle
           = current_fragment.FindEdge(in_edge_it->id());
        if (!current_fragment_edge_handle){
          // this edge is not contained in current_fragment
          uncovered_vertex_edge_num++;
          AddUncoveredSubstructure(in_edge_it, kSupportId);
        }
      }
      return true;
    };

    obtain_substructure(data_graph, supp_match, 
                                    substructure_callback);
    return uncovered_vertex_edge_num;
  }

  // bfs through the data graph, to find the substructure in it
  template <typename DataGraphType>
  inline DataGraphType ObtainSubstructure(DataGraphType& data_graph,
       std::set<
       std::vector<typename GUNDAM::VertexHandle<DataGraphType>
                                  ::type>>& supp_vertex_set_batch,
        const std::map<SupportIdType, int>& supp_id_batch) {

    using VertexCounterType = typename DataGraphType::VertexCounterType;
    using DataGraphVertexHandle = typename GUNDAM::VertexHandle<DataGraphType>::type;

    DataGraphType substructure;

    assert(supp_vertex_set_batch.size() == supp_id_batch.size());

    using AttrKeyType = std::remove_reference_t<decltype((*(supp_vertex_set_batch.begin()->begin()))
                                                          ->AttributeBegin()
                                                          ->key())>;

    auto prune_not_in_substructure_callback 
      = [&supp_id_batch](DataGraphVertexHandle vertex_handle){
      const AttrKeyType kSubstructureIdSetKey = "substructure_support_id_set";
      auto attr_handle = vertex_handle->FindAttribute(kSubstructureIdSetKey);
      if ( attr_handle.IsNull()){
        // this vertex is not contained in the substructure
        return true;
      }
      auto& covered_support_id_set = attr_handle->template value<SupportIdSetType>();
      bool not_covered_by_support = true;
      for (const auto& supp_id : supp_id_batch) {
        auto covered_support_id_it
           = covered_support_id_set.find(supp_id.first);
        if ( covered_support_id_it == covered_support_id_set.end()){
          // is not covered by this selected support
          continue;
        }
        // covered by this selected support
        not_covered_by_support = false;
        covered_support_id_set.erase(covered_support_id_it);
        if (covered_support_id_set.empty()){
          vertex_handle->EraseAttribute(kSubstructureIdSetKey);
          break;
        }
      }
      return not_covered_by_support;
    };

    auto construct_substructure_callback 
     = [&substructure](DataGraphVertexHandle vertex_handle){
      auto [substructure_vertex_handle,
            substructure_vertex_ret]
          = substructure.AddVertex(vertex_handle->id(),
                                   vertex_handle->label());
      // this vertex should not have been added 
      // into the substructure before
      assert(substructure_vertex_ret);
      assert(substructure_vertex_handle);

      GUNDAM::CopyAllAttributes(vertex_handle, 
                   substructure_vertex_handle);

      for (auto out_edge_it = vertex_handle->OutEdgeBegin();
               !out_edge_it.IsDone();
                out_edge_it++) {
        auto substructure_vertex_handle
           = substructure.FindVertex(out_edge_it->dst_handle()->id());
        if (!substructure_vertex_handle){
          // this vertex cannot be found in substructure
          // this edge does not need to be added
          continue;
        }
        // this edge needs to be added
        auto [substructure_edge_handle, 
              substructure_edge_ret]
            = substructure.AddEdge(out_edge_it->src_handle()->id(),
                                   out_edge_it->dst_handle()->id(),
                                   out_edge_it->label(),
                                   out_edge_it->id());
        // should added successfully
        assert(substructure_edge_handle);
        assert(substructure_edge_ret);
        GUNDAM::CopyAllAttributes(out_edge_it, 
                         substructure_edge_handle);
      }

      for (auto in_edge_it = vertex_handle->InEdgeBegin();
               !in_edge_it.IsDone();
                in_edge_it++) {
        auto substructure_vertex_handle
           = substructure.FindVertex(in_edge_it->src_handle()->id());
        if (!substructure_vertex_handle){
          // this vertex cannot be found in substructure
          // this edge does not need to be added
          continue;
        }
        // this edge needs to be added
        auto [substructure_edge_handle, 
              substructure_edge_ret]
            = substructure.AddEdge(in_edge_it->src_handle()->id(),
                                   in_edge_it->dst_handle()->id(),
                                   in_edge_it->label(),
                                   in_edge_it->id());
        // should added successfully
        assert(substructure_edge_handle);
        assert(substructure_edge_ret);
        GUNDAM::CopyAllAttributes(in_edge_it, 
                        substructure_edge_handle);
      }
      return true;
    };

    std::set<DataGraphVertexHandle> src_vertex_handle_set;
    for (const auto& supp_vertex_set
                   : supp_vertex_set_batch) {
      for (auto& supp_vertex_handle : supp_vertex_set){
        src_vertex_handle_set.emplace(supp_vertex_handle);
      }
    }
    VertexCounterType vertex_num = 0;
    vertex_num += GUNDAM::Bfs<true>(data_graph, src_vertex_handle_set,   
                                      construct_substructure_callback,
                                   prune_not_in_substructure_callback);
    assert(vertex_num == substructure.CountVertex());
    return substructure;
  }

  template <typename DataGraphVertexHandle>
  void Maintain(
       std::map<SupportIdType, int>& support_id_batch,
       std::map<SupportIdType,
                std::pair<SupportPatternIdxType,
                          std::vector<DataGraphVertexHandle>
                         >
               >& support_dict,
       std::vector<size_t>& multi_supp_added_size,
       std::vector<size_t>& multi_supp_matched_unadded_size,
       std::map<SupportIdType, 
       SubstructureNumberType>& support_uncovered_size,
       std::set<std::vector<DataGraphVertexHandle>>& matched_unadded_support_set,
       std::set<std::vector<DataGraphVertexHandle>>&           added_support_set){

    for (const auto& support_id : support_id_batch) {
      assert(support_dict.find(support_id.first) 
          != support_dict.end());
      assert(support_uncovered_size.find(support_id.first) 
          != support_uncovered_size.end());
      auto support_it = support_dict.find(support_id.first);
      auto support    = support_it->second.second;
      assert(matched_unadded_support_set.find(support)
          != matched_unadded_support_set.end());
      assert(added_support_set.find(support)
          == added_support_set.end());

      auto support_pattern_idx = support_it->second.first;
      assert(support_pattern_idx < multi_supp_added_size.size());
      multi_supp_added_size.at(support_pattern_idx)++;
      assert(support_pattern_idx < multi_supp_matched_unadded_size.size());
      assert(!multi_supp_matched_unadded_size.empty());
      multi_supp_matched_unadded_size.at(support_pattern_idx)--;

      auto   support_dict_erase_size
           = support_dict.erase(support_id.first);
      assert(support_dict_erase_size != 0);
      #ifndef NDEBUG
      auto   support_uncovered_size_it
          = support_uncovered_size.find(support_id.first);
      assert(support_uncovered_size_it->second == 0);
      #endif
      auto   support_uncovered_size_erase_size
          = support_uncovered_size.erase(support_id.first);
      assert(support_uncovered_size_erase_size != 0);
      auto   matched_unadded_support_set_erase_size
          = matched_unadded_support_set.erase(support);
      assert(matched_unadded_support_set_erase_size != 0);
      auto [ added_support_set_it,
            added_support_set_ret ]
          = added_support_set.emplace(support);
      assert(added_support_set_ret);

      assert(support_dict.find(support_id.first) 
          == support_dict.end());
      assert(support_uncovered_size.find(support_id.first) 
          == support_uncovered_size.end());
      assert(matched_unadded_support_set.find(support)
          == matched_unadded_support_set.end());
      assert(added_support_set.find(support)
          != added_support_set.end());
    }
    return;
  }

  template <typename DataGraphType>
  void MaintainUncoveredNum(DataGraphType& substructure, 
                            DataGraphType& data_graph,
                   std::map<SupportIdType, 
                   SubstructureNumberType>& support_uncovered_size){
    using EdgeIDType = typename DataGraphType::EdgeType::IDType;
    for (auto vertex_it = substructure.VertexBegin();
             !vertex_it.IsDone(); vertex_it++){
      auto   date_graph_vertex_handle
           = data_graph.FindVertex(vertex_it->id());
      assert(date_graph_vertex_handle);
      RemoveUncoveredNum(date_graph_vertex_handle, 
                         support_uncovered_size);
      std::set<EdgeIDType> out_edge_id_set;
      for (auto edge_it = vertex_it->OutEdgeBegin();
               !edge_it.IsDone(); edge_it++){
        out_edge_id_set.emplace(edge_it->id());
      }
      for (auto edge_it = date_graph_vertex_handle->OutEdgeBegin();
               !edge_it.IsDone(); edge_it++){
        if (out_edge_id_set.find(edge_it->id()) 
         == out_edge_id_set.end()){
          // this edge is not contained in substructure
          continue;
        }
        RemoveUncoveredNum(edge_it, support_uncovered_size);
      }
    }
  }

  void MaintainSortedUncoveredSupport(
   std::map<SubstructureNumberType,
            std::set<SupportIdType>>& sorted_uncovered_support,
      std::map<SupportIdType, 
      SubstructureNumberType>& temp_support_uncovered_size,
      std::map<SupportIdType, 
      SubstructureNumberType>&      support_uncovered_size){
    
    // maintain the sorted_uncovered_support
    assert(temp_support_uncovered_size.size()
             == support_uncovered_size.size());
    for (auto support_uncovered_size_it
            = support_uncovered_size.begin();
              support_uncovered_size_it
           != support_uncovered_size.end();){
      auto temp_support_uncovered_size_it
         = temp_support_uncovered_size.find(
                support_uncovered_size_it->first);
      // should be found in temp_support_uncovered_size
      assert(temp_support_uncovered_size_it 
          != temp_support_uncovered_size.end());
      if (temp_support_uncovered_size_it->second
            == support_uncovered_size_it->second){
        // temp_support_uncovered_size_it->second
        // is same as      
        //      support_uncovered_size_it->second
        support_uncovered_size_it++;
        continue;
      }
      // the uncovered vertex and edge num has been changed from 
      //    temp_support_uncovered_size_it->second
      // to      support_uncovered_size_it->second

      auto sorted_uncovered_support_it 
         = sorted_uncovered_support.find(temp_support_uncovered_size_it->second);
      assert(sorted_uncovered_support_it
          != sorted_uncovered_support.end());
      auto sorted_uncovered_support_id_it 
         = sorted_uncovered_support_it->second.find(support_uncovered_size_it->first);
      // should have been found in this set
      assert(sorted_uncovered_support_id_it
          != sorted_uncovered_support_it->second.end());
      sorted_uncovered_support_it->second.erase(sorted_uncovered_support_id_it);
      // do not erase it even if it has already been empty
      // sorted_uncovered_support_it->second.empty()
      sorted_uncovered_support_it = sorted_uncovered_support.emplace_hint(
                                    sorted_uncovered_support_it,
                                   support_uncovered_size_it->second,
                                   std::set<SupportIdType>());
      assert(sorted_uncovered_support_it 
          != sorted_uncovered_support.end());
      sorted_uncovered_support_it->second.emplace(support_uncovered_size_it->first);
      support_uncovered_size_it++;
    }
    for (auto sorted_uncovered_support_it
            = sorted_uncovered_support.begin();
              sorted_uncovered_support_it
           != sorted_uncovered_support.end();){
      if (!sorted_uncovered_support_it->second.empty()){
        sorted_uncovered_support_it++;
        continue;
      }
      sorted_uncovered_support_it 
        = sorted_uncovered_support.erase(sorted_uncovered_support_it);
    }
    return;
  }

  template <typename GraphType>
  void ExportSampleGraph(GraphType& sampled_graph,
          const std::set<
                  std::vector<
                    typename GUNDAM::VertexHandle<GraphType>::type
                > >& added_support_set,
          const std::string& kDataGraphPrefix,
          const std::string& kSampleMethod,
          const std::string& additional_name,
          const std::string& substructure_method,
                      double ratio,
                      time_t begin_time){
    const std::string kSampledGraphPrefix 
                       = kDataGraphPrefix 
              + "_" + kSampleMethod 
                    + additional_name
              + "_" + substructure_method
              + "_" + std::to_string(ratio);


    const std::string kSampledGraphTimeFile 
                    = kSampledGraphPrefix + "_time.txt";
    std::ofstream time_file(kSampledGraphTimeFile);
    time_file << std::time(NULL) - begin_time << std::endl;
    time_file.close();

    const std::string kSampledGraphVFile = kSampledGraphPrefix + "_v.csv",
                      kSampledGraphEFile = kSampledGraphPrefix + "_e.csv";
    GUNDAM::WriteCSVGraph(sampled_graph, kSampledGraphVFile, 
                                         kSampledGraphEFile);

    const std::string kSampledGraphSupportFile 
                    = kSampledGraphPrefix + "_supp.txt";
    std::ofstream support_sample_file;
    support_sample_file.open(kSampledGraphSupportFile, std::ios::out 
                                                     | std::ios::trunc);
    for (const auto& considered_supp : added_support_set){
      for (const auto& considered_supp_handle : considered_supp){
        support_sample_file<<"\t"<<considered_supp_handle->id();
      }
      support_sample_file<<std::endl;
    }
    support_sample_file.close();
    return;
  }

};

template <typename FRAG_T>
class SupportSample : public ParallelAppBase<FRAG_T, SupportSampleContext<FRAG_T>>,
                      public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
  using    VertexIDType = typename SupportSampleContext<FRAG_T>::   VertexIDType;
  using VertexLabelType = typename SupportSampleContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType = typename SupportSampleContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType = typename SupportSampleContext<FRAG_T>::  EdgeLabelType;
  using    DataGraph    = typename SupportSampleContext<FRAG_T>::DataGraph;
  using      Pattern    = typename SupportSampleContext<FRAG_T>::  Pattern;

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(SupportSample<FRAG_T>, 
                          SupportSampleContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for SupportSample.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void PEval(const fragment_t& frag, 
                    context_t& ctx,
            message_manager_t& messages) {

    using namespace _support_sample;
    
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

    DataGraph data_graph;
    GUNDAM::ReadCSVGraph(data_graph, 
                        kDataGraphPathVFile,
                        kDataGraphPathEFile);

    using SupportMatchSetType = GUNDAM::MatchSet<Pattern, DataGraph>;

    std::vector<std::pair<Pattern, SupportMatchSetType>> supp_match_set;

    if (!config["SuppPattern"]) {
      std::cout << "cannot get support pattern path" << std::endl;
      return;
    }
    YAML::Node supp_pattern_paths_config = config["SuppPattern"];

    bool has_denoted_supp = false;
    for (int i = 0; i < supp_pattern_paths_config.size(); i++) {
      YAML::Node supp_pattern_path_config 
               = supp_pattern_paths_config[i];
      if (!supp_pattern_path_config["VFile"]) {
        std::cout << "cannot get support pattern v file" << std::endl;
        return;
      }
      const std::string kSuppPatternPathVFile
            = supp_pattern_path_config["VFile"].as<std::string>();

      if (!supp_pattern_path_config["EFile"]) {
        std::cout << "cannot get support pattern e file" << std::endl;
        return;
      }
      const std::string kSuppPatternPathEFile
            = supp_pattern_path_config["EFile"].as<std::string>();

      Pattern support_pattern; 
      GUNDAM::ReadCSVGraph(support_pattern, 
                          kSuppPatternPathVFile,
                          kSuppPatternPathEFile);

      for (int support_id = 0;
               support_id < supp_match_set.size();
               support_id++) {
        if (GUNDAM::SamePattern(support_pattern, 
                                supp_match_set[support_id].first)) {
          std::cout << "the " << i << "'th support pattern is duplicated with"
                    << "the " << support_id << "'th support pattern"
                    << std:: endl;
          return;
        }
      }

      if (supp_pattern_path_config["SuppFile"]) {
        has_denoted_supp = true;
        // support file is specified
        const std::string supp_file_path 
                        = supp_pattern_path_config["SuppFile"].as<std::string>();

        std::ifstream supp_file(supp_file_path);

        GUNDAM::MatchSet supp_match(support_pattern, data_graph, supp_file);

        supp_match_set.emplace_back(std::move(support_pattern), 
                                    std::move(supp_match));
        continue;
      }
      // the support is not specified, add an empty match
      GUNDAM::MatchSet supp_match(support_pattern, data_graph);
      supp_match_set.emplace_back(std::move(support_pattern), 
                                  std::move(supp_match));
    }
    if (has_denoted_supp) {
      // has read support file, just to check whether has read
      // support file for each support
      for (const auto& supp_match
                     : supp_match_set) {
        if (supp_match.second.Empty()) {
          std::cout << "does not read support file for all supports"
                    << std::endl;
          return;
        }
      }
    }
    
    std::string multi_supp_select_method = "same_number";
    if (config["MultiSupportSelectMethod"]) {
      if (supp_pattern_paths_config.size() == 1) {
        std::cout << "only contain one support, does not need to specify the select method from multi support set"
                  << std::endl;
      }
      multi_supp_select_method = config["MultiSupportSelectMethod"].as<std::string>();
    }
    // only support same number now
    assert(multi_supp_select_method == "same_number");

    if (!config["SupportSampleFile"]) {
      std::cout << "cannot get support sample file path" << std::endl;
      return;
    }
    const std::string kSupportSampleFile
             = config["SupportSampleFile"].as<std::string>();

    if (!config["SupportSampleMethod"]) {
      std::cout << "does not specify support sample method" << std::endl;
      return;
    }
    const std::string kSupportSampleMethod
             = config["SupportSampleMethod"].as<std::string>();

    if (!config["Ratios"]) {
      std::cout << "cannot get sample ratios!" << std::endl;
      return;
    }
    YAML::Node ratios_config = config["Ratios"];
    if (ratios_config.size() == 0){
      std::cout << "cannot get sample ratios!" << std::endl;
      return;
    }
    std::map<size_t, double> ratio_list;
    size_t max_vertex_edge_num = 0;

    int batch_size = 1;
    if (config["BatchSize"]){
      batch_size = config["BatchSize"].as<int>();
    }

    bool denoted_support_only = false;
    if (config["DenotedSupportOnly"])  {
      denoted_support_only = config["DenotedSupportOnly"].as<bool>();
    }
 
    const size_t kDataGraphVertexEdgeNum 
                = data_graph.CountVertex()
                + data_graph.CountEdge();

    for(int i=0; i < ratios_config.size(); ++i){
      max_vertex_edge_num = ratios_config[i].as<double>() * kDataGraphVertexEdgeNum;
      ratio_list.emplace(max_vertex_edge_num,
                         ratios_config[i].as<double>());
    }
    assert(max_vertex_edge_num > 0
        && max_vertex_edge_num <= kDataGraphVertexEdgeNum);

    for (const auto& ratio : ratio_list){
      std::cout<<ratio.first<<"\t"<<ratio.second<<std::endl;
    }
            
    std::string additional_name = "";
    if (config["AdditionalName"]){
      additional_name = config["AdditionalName"].as<std::string>();
      additional_name = "_" + additional_name;
    }

    if (!config["Substructure"]) {
      std::cout << "does not specify substructure" << std::endl;
      return;
    }
    YAML::Node substructure_set_config = config["Substructure"];

    for (int i = 0; i < substructure_set_config.size(); i++) {
      YAML::Node substructure_config 
               = substructure_set_config[i];

      std::cout << "processing the " << i << "'th substructure method"
                << std::endl;

      std::map<size_t, double> remained_ratio_list = ratio_list;

      if (!substructure_config["Method"]){
        std::cout << "does not specify substructure method" << std::endl;
        return;
      }
      const std::string kSubstructureMethod
              = substructure_config["Method"].as<std::string>();
    
      std::string substructure_method;
      // parameter for bfs and bfs_limit_width
      int bfs_radius = 0;
      // parameter for bfs_limit_width
      int width_limit = 0;
      // parameter for random walk
      int random_walk_depth = 0,
          random_walk_size  = 0;
      if (kSubstructureMethod == "bfs") {
        if (!substructure_config["Radius"]){
          std::cout << "radius of Bfs method is not specified!" << std::endl;
          return;
        }
        bfs_radius = substructure_config["Radius"].as<int>();
        substructure_method = "bfs_" + std::to_string(bfs_radius);
      }
      else if (kSubstructureMethod == "bfs_limit_width") {
        if (!substructure_config["Radius"]){
          std::cout << "radius of BfsWidthLimit method is not specified!" << std::endl;
          return;
        }
        bfs_radius = substructure_config["Radius"].as<int>();
        if (!substructure_config["WidthLimit"]){
          std::cout << "width limit of BfsWidthLimit method is not specified!" << std::endl;
          return;
        }
        width_limit = substructure_config["WidthLimit"].as<int>();
        substructure_method = "bfs_width_limit_r" + std::to_string(bfs_radius) 
                                           + "_w" + std::to_string(width_limit);
      }
      else if (kSubstructureMethod == "random_walk") {
        if (!substructure_config["Depth"]){
          std::cout << "depth of random walk method is not specified!" << std::endl;
          return;
        }
        random_walk_depth = substructure_config["Depth"].as<int>();
        if (!substructure_config["Size"]){
          std::cout << "size of random walk method is not specified!" << std::endl;
          return;
        }
        random_walk_size = substructure_config["Size"].as<int>();
        substructure_method = "random_walk_d" + std::to_string(random_walk_depth)
                                      + "_s" + std::to_string(random_walk_size);
      }
      else {
        std::cout<<"unknown substructure method: "<<kSubstructureMethod<<std::endl;
        assert(false);
        return;
      }

      // #############################################
      // ##    all supported substructure method    ##
      // #############################################

      // return the size of obtained substructure
      auto obtain_substructure 
      = [&kSubstructureMethod,
          // parameter for bfs method and bfs_limit_width
          &bfs_radius,
          // parameter for bfs_limit_width
          &width_limit,
          // parameter for random walk method
          &random_walk_depth,
          &random_walk_size](DataGraph& data_graph,
              const std::vector<typename GUNDAM::VertexHandle<DataGraph>::type>& support_vertex_set,
              std::function<bool(typename GUNDAM::VertexHandle<DataGraph>::type)> vertex_callback) -> uint32_t {
        using VertexHandleType = typename GUNDAM::VertexHandle<DataGraph>::type;
        if (kSubstructureMethod == "bfs"){
          std::set<VertexHandleType> src_handle_set;
          for (auto& support_vertex_handle
                  : support_vertex_set) {
            src_handle_set.emplace(support_vertex_handle);
          }
          return GUNDAM::Bfs<true>(data_graph, src_handle_set, vertex_callback, bfs_radius);
        }
        else if (kSubstructureMethod == "bfs_limit_width"){
          std::set<VertexHandleType> src_handle_set;
          for (auto& support_vertex_handle
                  : support_vertex_set) {
            src_handle_set.emplace(support_vertex_handle);
          }
          return GUNDAM::BfsLimitWidth<true>(data_graph, src_handle_set, width_limit, vertex_callback, bfs_radius);
        }
        assert(kSubstructureMethod == "random_walk");
        std::set<VertexHandleType> visited_vertex_set;
        auto remove_duplicate_vertex_callback 
                  = [&visited_vertex_set,
                    &support_vertex_set,
                    &random_walk_size,
                    &vertex_callback](VertexHandleType vertex_handle) -> bool {
          auto [visited_vertex_it,
                visited_vertex_ret]
              = visited_vertex_set.emplace(vertex_handle);
          assert(visited_vertex_set.size() 
            <= (support_vertex_set.size() * random_walk_size));
          if (!visited_vertex_ret){
            // this vertex has already been visited
            return true;
          }
          // it is the frist time for this vertex to be visited
          return vertex_callback(vertex_handle);
        };
        for (auto& support_vertex_handle : support_vertex_set) {
          GUNDAM::RandomWalk<true>(data_graph, support_vertex_handle, 
                                                random_walk_depth,
                                                random_walk_size, 
                                                remove_duplicate_vertex_callback);
        }
        return visited_vertex_set.size();
      };

      if (kSupportSampleMethod != "locality"
       && kSupportSampleMethod != "uniform"){
        std::cout<<"unknown sample method: "<<kSupportSampleMethod<<std::endl;
        return;
      }

      // store the denoted_support_set for different support pattern seperately
      std::vector<std::set<std::vector<typename GUNDAM::VertexHandle<DataGraph>::type>>>
          multi_supp_denoted_support_set;
      multi_supp_denoted_support_set.resize(supp_match_set.size());
      for (int support_pattern_idx = 0;
              support_pattern_idx < supp_match_set.size();
              support_pattern_idx++){
        auto& [support_pattern,
              supp_match] 
            = supp_match_set[support_pattern_idx];
        assert(support_pattern_idx < multi_supp_denoted_support_set.size());
        auto& denoted_support_set = multi_supp_denoted_support_set[support_pattern_idx];
        for (auto supp_match_it = supp_match.MatchBegin();
                !supp_match_it.IsDone();
                  supp_match_it++) {
          std::vector<typename GUNDAM::VertexHandle<DataGraph>::type> 
            data_graph_vertex_handle_set;
          data_graph_vertex_handle_set.reserve(support_pattern.CountVertex());
          for (auto supp_vertex_it = support_pattern.VertexBegin();
                   !supp_vertex_it.IsDone();
                    supp_vertex_it++){
            data_graph_vertex_handle_set.emplace_back(supp_match_it->MapTo(supp_vertex_it));
          }
          auto [denoted_support_set_it, 
                denoted_support_set_ret] 
              = denoted_support_set.emplace(data_graph_vertex_handle_set);
          // should have been added successfully, since different support pattern cannot
          // match to the same vertex_set in data graph
          assert(denoted_support_set_ret);
        }
      }

      std::ofstream support_sample_file;
      support_sample_file.open(kSupportSampleFile, std::ios::out 
                                                 | std::ios::trunc);

      using GraphPatternVertexHandle = typename GUNDAM::VertexHandle< Pattern >::type;
      using    DataGraphVertexHandle = typename GUNDAM::VertexHandle<DataGraph>::type;

      // the matched set for different support pattern are stored 
      // in the same container
      std::set<std::vector<DataGraphVertexHandle>> 
                      matched_unadded_support_set,
                                added_support_set;

      using SupportPatternIdxType = int;

      // the matched set for different support pattern are stored 
      // in the same container
      // for each support id, the id of the support pattern it belongs to
      // and the vertex set in the data graph that it can match to
      std::map<SupportIdType,
              std::pair<SupportPatternIdxType,
                        std::vector<DataGraphVertexHandle>>
              > support_dict;
      
      // store the size of uncovered substructure
      std::map<SupportIdType,
      SubstructureNumberType> support_uncovered_size;
      
      std::map<SubstructureNumberType,
              std::set<SupportIdType>> sorted_uncovered_support;

      // first added all denoted support, then add other support
      // according to the locality

      DataGraph current_fragment;

      auto begin_time = std::time(NULL);
      
      // the support id are allocated in an uniform space but
      // are stored in different containers, 
      //    e.g. same id won't exist in different support pattern
      std::vector<std::set<SupportIdType>> multi_supp_denoted_support_id_set;
      multi_supp_denoted_support_id_set.resize(supp_match_set.size());
      // add all denoted supports into consideration, 
      // calcuate the uncovered vertex & edge number for each 
      // of them
      for (int support_pattern_idx = 0;
               support_pattern_idx < supp_match_set.size();
               support_pattern_idx++){
        auto& denoted_support_set    = multi_supp_denoted_support_set   [support_pattern_idx];
        auto& denoted_support_id_set = multi_supp_denoted_support_id_set[support_pattern_idx];
        for (auto& denoted_support : denoted_support_set) {
          // maintain support_uncovered_size and sorted_uncovered_support
          // mark its substructure in the data graph for optimization
          const auto kSuppId = matched_unadded_support_set.size()
                                      + added_support_set.size();
                      
          const SubstructureNumberType 
               kUncoveredSubstructureNum 
              = UncoveredSubstructureSize(data_graph,
                                          current_fragment,
                                          denoted_support,
                                         kSuppId,
                                          obtain_substructure);

          // add it into support_dict
          auto [support_dict_it,
                support_dict_ret] 
              = support_dict.emplace(kSuppId, std::pair(support_pattern_idx, denoted_support));

          assert(support_dict_ret);

          auto [ support_uncovered_size_it,
                 support_uncovered_size_ret ]
               = support_uncovered_size.emplace(kSuppId, kUncoveredSubstructureNum);
          assert(support_uncovered_size_ret);

          auto [ matched_unadded_support_it,
                 matched_unadded_support_ret ]
               = matched_unadded_support_set.emplace(denoted_support);
          assert(matched_unadded_support_ret);

          // mantain the sorted uncovered support set
          auto sorted_uncovered_support_it
            = sorted_uncovered_support.find(kUncoveredSubstructureNum);
          if (sorted_uncovered_support_it 
          == sorted_uncovered_support.end()){
            bool sorted_uncovered_support_ret = false;
            // is not contained in sorted_uncovered_support
            std::tie (sorted_uncovered_support_it,
                      sorted_uncovered_support_ret ) 
                    = sorted_uncovered_support
                            .emplace(std::move(kUncoveredSubstructureNum),
                                    std::set<SupportIdType>());
            assert(sorted_uncovered_support_ret);
            assert(sorted_uncovered_support_it 
                != sorted_uncovered_support.end());
          }
          auto [ sorted_uncovered_support_supp_id_set_it,
                 sorted_uncovered_support_supp_id_set_ret ]
               = sorted_uncovered_support_it->second.emplace(kSuppId);
          assert(sorted_uncovered_support_supp_id_set_ret);

          auto [ denoted_support_id_it,
                 denoted_support_id_ret ]
               = denoted_support_id_set.emplace(kSuppId);
          assert(denoted_support_id_ret);
        }
      }

      #ifndef NDEBUG
      size_t all_denoted_support_set_size = 0;
      for (const auto& denoted_support_set 
          : multi_supp_denoted_support_set){
        all_denoted_support_set_size += denoted_support_set.size();
      }
      assert(all_denoted_support_set_size == matched_unadded_support_set.size());
      assert(all_denoted_support_set_size == support_dict.size());

      size_t all_denoted_support_id_set_size = 0;
      for (const auto& denoted_support_id_set 
          : multi_supp_denoted_support_id_set){
        all_denoted_support_id_set_size += denoted_support_id_set.size();
      }
      assert(all_denoted_support_set_size 
          == all_denoted_support_id_set_size);

      auto kDenotedSuppSize = all_denoted_support_set_size;
      #endif // NDEBUG

      multi_supp_denoted_support_set.clear();

      assert(added_support_set.empty());

      using MatchType = GUNDAM::Match<Pattern, DataGraph>;

      std::function<bool(const MatchType&)> 
        prune_nothing_callback = [](const MatchType& match){
        return false;
      };

      std::vector<size_t> multi_supp_matched_unadded_size;

      auto find_new_supp_callback 
        = [&support_dict,
          &support_uncovered_size,
          &data_graph,
          &current_fragment,
          &multi_supp_matched_unadded_size,
          &matched_unadded_support_set,
                    &added_support_set,
          &sorted_uncovered_support,
          &batch_size,
          &obtain_substructure](Pattern& support_pattern,
                                SupportPatternIdxType support_pattern_idx,
                                bool find_all_support_in_fragment,
                                int& found_support_number,
                                const MatchType& match){
        // from match set to supp
        std::vector<DataGraphVertexHandle> supp;
        supp.reserve(support_pattern.CountVertex());
        assert(match.size() == support_pattern.CountVertex());
        for (auto vertex_it =  support_pattern.VertexBegin();
                 !vertex_it.IsDone();
                  vertex_it++){
          auto dst_handle = match.MapTo(vertex_it);
          assert(dst_handle);
          supp.emplace_back(data_graph.FindVertex(dst_handle->id()));
        }
        assert(supp.size() == support_pattern.CountVertex());

        auto matched_unadded_support_it
           = matched_unadded_support_set.find(supp);
        if ( matched_unadded_support_it
          != matched_unadded_support_set.end()){
          // has already been considered
          // continue matching

          // could be considered before
          // assert(false);
          return true;
        }
        auto added_support_it  = added_support_set.find(supp);
        if ( added_support_it != added_support_set.end()){
          // has already been considered
          // continue matching
          
          // could be considered before
          // assert(false);
          return true;
        }

        // found a new support, allocate an id for it
        const auto kSuppId = matched_unadded_support_set.size()
                                     + added_support_set.size();
        // add it into support_dict
        auto [ support_dict_it,
               support_dict_ret ]
             = support_dict.emplace(kSuppId, std::pair(support_pattern_idx,supp));
        assert(support_dict_ret);

        // add it into matched_unadded_support_set
        bool matched_unadded_support_ret = false;
        std::tie(matched_unadded_support_it,
                 matched_unadded_support_ret) 
               = matched_unadded_support_set.emplace(supp);
        assert(matched_unadded_support_ret);

        assert(multi_supp_matched_unadded_size.size() > support_pattern_idx);
        multi_supp_matched_unadded_size[support_pattern_idx]++;

        #ifndef NDEBUG
        size_t sum_multi_supp_matched_unadded_size = 0;
        for (const auto& matched_unadded_size
            : multi_supp_matched_unadded_size) {
          sum_multi_supp_matched_unadded_size 
                      += matched_unadded_size;
        }
        assert(sum_multi_supp_matched_unadded_size 
                          == matched_unadded_support_set.size());
        #endif
        
        // calculate how many vertex and edge in the substurcture around the
        // selected support are not covered by current_fragment
        const SubstructureNumberType 
             kUncoveredSubstructureNum 
            = UncoveredSubstructureSize(data_graph,
                                        current_fragment,
                                        supp,
                                       kSuppId,
                                        obtain_substructure);

        auto [support_uncovered_size_it,
              support_uncovered_size_ret]
            = support_uncovered_size.emplace(kSuppId,
                    kUncoveredSubstructureNum);
        assert(support_uncovered_size_ret);

        assert(matched_unadded_support_it
            != matched_unadded_support_set.end());

        // mantain the sorted uncovered support set
        auto sorted_uncovered_support_it
           = sorted_uncovered_support.find(kUncoveredSubstructureNum);
        if (sorted_uncovered_support_it 
         == sorted_uncovered_support.end()){
          bool sorted_uncovered_support_ret = false;
          // is not contained in sorted_uncovered_support
          std::tie (sorted_uncovered_support_it,
                    sorted_uncovered_support_ret ) 
                  = sorted_uncovered_support
                          .emplace(std::move(kUncoveredSubstructureNum),
                                   std::set<SupportIdType>());
          assert(sorted_uncovered_support_ret);
          assert(sorted_uncovered_support_it 
              != sorted_uncovered_support.end());
        }
        auto [ sorted_uncovered_support_supp_id_set_it,
               sorted_uncovered_support_supp_id_set_ret ]
             = sorted_uncovered_support_it->second.emplace(kSuppId);
        assert(sorted_uncovered_support_supp_id_set_ret);
        if (find_all_support_in_fragment){
          // continue matching
          return true;
        }
        assert(found_support_number >= 0 
            && found_support_number < batch_size);
        found_support_number++;
        if (found_support_number < batch_size){
          // continue matching
          return true;
        }
        // has found enough new un-considered support in data graph
        return false;
      };

      std::vector<size_t> multi_supp_added_size;
      multi_supp_added_size.resize(supp_match_set.size(), 0);
      multi_supp_matched_unadded_size.reserve(supp_match_set.size());
      for (int support_pattern_idx = 0; 
               support_pattern_idx < supp_match_set.size();
               support_pattern_idx++) {
        multi_supp_matched_unadded_size.emplace_back(
        multi_supp_denoted_support_id_set[support_pattern_idx].size());
      }
      std::vector<size_t> multi_supp_added_denoted_size;
      multi_supp_added_denoted_size.resize(supp_match_set.size(), 0);

      // both the support_uncovered_size and sorted_uncovered_support
      // are mantained at the same time 
      while (has_denoted_supp){
        // select a support_pattern_idx to add according to the
        // multi_supp_select_method strategy
        auto selected_support_pattern_idx
           = SelectDenotedSupportIdSet(multi_supp_denoted_support_id_set, 
                                       multi_supp_added_denoted_size, 
                                       multi_supp_select_method);
        std::cout << "multi_supp_matched_unadded_size: " << std::endl;
        for (const auto& matched_unadded_size 
            : multi_supp_matched_unadded_size){
          std::cout << "\t" << matched_unadded_size;
        }
        std::cout << std::endl;
        std::cout << "multi_supp_added_size:" << std::endl;
        for (const auto& added_size : multi_supp_added_size){
          std::cout << "\t" << added_size;
        }
        std::cout << std::endl;

        assert(multi_supp_denoted_support_id_set.size()
                      > selected_support_pattern_idx);

        auto& selected_denoted_support_id_set
          = multi_supp_denoted_support_id_set.at(
                      selected_support_pattern_idx);
        if (selected_denoted_support_id_set.empty()){
          // which means that the denoted support id for 
          // all support pattern are all added
          #ifndef NDEBUG
          for (const auto& denoted_support_id_set
              : multi_supp_denoted_support_id_set) {
            assert(denoted_support_id_set.empty());
          }
          #endif 
          break;
        }

        // find a batch of support with minimal uncovered vertex and edge
        // add all support whose substructure does not introduce
        // new vertex and edge in it 
        std::map<SupportIdType, int> selected_denoted_support_id_batch;
        for (auto uncovered_support_it  = sorted_uncovered_support.begin();
                  uncovered_support_it != sorted_uncovered_support. end ();){
          if (uncovered_support_it->first == 0) {
            std::cout << uncovered_support_it->first << " # size: "
                      << uncovered_support_it->second.size()
                      << std::endl;
            assert(selected_denoted_support_id_batch.empty());
            // added all support whose substructure introduce
            // no new vertexes and edges into consideration
            // e.g. the substructure of those support has 
            //      already been contained in the current_fragment
            std::map<SupportIdType, int> selected_support_id_batch;
            for (const auto& support_id : uncovered_support_it->second) {
              bool contained_in_denoted_support = false;
              for (int support_pattern_idx = 0; 
                       support_pattern_idx < supp_match_set.size();
                       support_pattern_idx++) {
                assert(multi_supp_denoted_support_id_set.size() > support_pattern_idx);
                auto& denoted_support_id_set = multi_supp_denoted_support_id_set[support_pattern_idx];
                // remove it from denoted_support_id_set if it is denoted
                auto denoted_support_id_set_it
                   = denoted_support_id_set.find(support_id);
                if ( denoted_support_id_set_it 
                  != denoted_support_id_set.end()){
                  // support_id is contained in denoted_support_id_set
                  // remove it from denoted_support_id_set
                  denoted_support_id_set.erase(denoted_support_id_set_it);
                  multi_supp_added_denoted_size[support_pattern_idx]++;
                  // it could be possible that all denoted support
                  // contained in the current fragment now would
                  // introduce no uncovered vertex nor edge
                  if (support_pattern_idx == selected_support_pattern_idx){
                    // has selected a support for the selected support patttern
                    // set the uncovered_num and selected_denoted_support_id
                    // to avoid assert failure
                    // ############################################
                    // uncovered_num = 0;
                    // selected_denoted_support_id = support_id;
                    // ############################################
                    auto [ selected_denoted_support_id_batch_it,
                           selected_denoted_support_id_batch_ret ]
                         = selected_denoted_support_id_batch.emplace(support_id, 0);
                    assert(selected_denoted_support_id_batch_ret);
                  }
                  // this id could exist in one denoted support set at most
                  contained_in_denoted_support = true;
                  break;
                }
                // is not contained in the denoted
                // support set of this support_pattern_idx
              }
              if (!contained_in_denoted_support) {
                auto [ selected_support_id_batch_it,
                       selected_support_id_batch_ret ]
                     = selected_support_id_batch.emplace(support_id, 0);
                assert(selected_support_id_batch_ret);
              }
            }
            // might not equal, some of the support with uncovered number equal to 0
            // might not be contained in denoted support
            // assert(selected_denoted_support_id_batch.size()
            //          == uncovered_support_it->second.size());
            Maintain(selected_denoted_support_id_batch,
                                      support_dict,
                                      multi_supp_added_size,
                                      multi_supp_matched_unadded_size,
                                      support_uncovered_size,
                                      matched_unadded_support_set,
                                                added_support_set);
            selected_denoted_support_id_batch.clear();
            Maintain(selected_support_id_batch,
                              support_dict,
                              multi_supp_added_size,
                              multi_supp_matched_unadded_size,
                              support_uncovered_size,
                              matched_unadded_support_set,
                                        added_support_set);
            selected_support_id_batch.clear();
            uncovered_support_it = sorted_uncovered_support.erase(uncovered_support_it);
            continue;
          }
          std::cout << "uncovered_support_it->first: "
                    <<  uncovered_support_it->first
                    << std::endl;
          assert(uncovered_support_it->first != 0);
          if (selected_denoted_support_id_set.empty()){
            // all selected_denoted_support_id has been added
            // none left
            // which means that all denoted support contained in 
            // sorted_uncovered_support has uncovered_num equal to 0
            #ifndef NDEBUG
            for (const auto& selected_denoted_support_id
                           : selected_denoted_support_id_batch) {
              assert(selected_denoted_support_id.second == 0);
            }
            #endif // NDEBUG
            break;
          }
          bool has_moved_to_next_uncovered_support = false;
          // there are still denoted support left in sorted_uncovered_support
          // find the one with minimal uncovered_num to add into current_fragment
          // bool containes_selected_denoted_supp = false;
          // to find whether this contains the denoted support 
          for (auto support_id_it  = uncovered_support_it->second.begin();
                    support_id_it != uncovered_support_it->second.end();){
            const auto kSupportId = *support_id_it;
            // find whether this support is contained in
            // the selected denoted support set
            auto selected_denoted_support_id_it
               = selected_denoted_support_id_set.find(kSupportId);
            if ( selected_denoted_support_id_it
              == selected_denoted_support_id_set.end()){
              // this support is not contained in selected denoted support set
              support_id_it++;
              continue;
            }
            // this support is denoted 
            // containes_selected_denoted_supp = true;
            // ############################################
            // uncovered_num = uncovered_support_it->first;
            // selected_denoted_support_id = kSupportId;
            // ############################################

            auto [ selected_denoted_support_id_batch_it,
                   selected_denoted_support_id_batch_ret ] 
                 = selected_denoted_support_id_batch.emplace(kSupportId,
                                                             uncovered_support_it->first);
            assert(selected_denoted_support_id_batch_ret);

            // remove this from sorted_uncovered_support
            support_id_it = uncovered_support_it->second.erase(support_id_it);
            if (uncovered_support_it->second.empty()) {
              uncovered_support_it = sorted_uncovered_support.erase(uncovered_support_it);
              has_moved_to_next_uncovered_support = true;
              break;
            }
            assert(selected_denoted_support_id_batch.size() <= batch_size);
            if (selected_denoted_support_id_batch.size() == batch_size){
              // has collected enough support
              break;
            }
          }
          if (!has_moved_to_next_uncovered_support) {
            uncovered_support_it++;
          }

          std::cout << "selected_denoted_support_id_batch.size(): "
                    <<  selected_denoted_support_id_batch.size()
                    << std::endl;
          std::cout << "#  remained uncovered support  #" << std::endl;

          for (auto uncovered_support_it  = sorted_uncovered_support.begin();
                    uncovered_support_it != sorted_uncovered_support. end ();
                    uncovered_support_it++){
            std::cout << uncovered_support_it->first << " # size: "
                      << uncovered_support_it->second.size()
                      << std::endl;
          }

          assert(selected_denoted_support_id_batch.size() <= batch_size);
          if (selected_denoted_support_id_batch.size() == batch_size){
            // has collected enough support
            // in sorted_uncovered_support, does not
            // need to further matching
            // ############################################
            // assert(uncovered_num != -1);
            // assert( selected_denoted_support_id != -1);
            // ############################################
            assert(!selected_denoted_support_id_batch.empty());
            #ifndef NDEBUG
            // at least there should be one support
            // whose uncovered substructure size not equal to zero 
            for (const auto& selected_denoted_support_id 
                           : selected_denoted_support_id_batch) {
              assert(selected_denoted_support_id.second != 0);
            }
            #endif // NDEBUG
            break;
          }
          // could be erased in <*>
          if (uncovered_support_it == sorted_uncovered_support.end()){
            break;
          }
        }
        #ifndef NDEBUG
        if (matched_unadded_support_set.empty()){
          // if there are no matched_unadded_support left
          // then all support added in selected_denoted_support_id_batch
          // should have uncovered size zero
          // which have been removed before
          for (const auto& selected_denoted_support_id 
                         : selected_denoted_support_id_batch) {
            assert(selected_denoted_support_id.second == 0);
          }
        }
        #endif // NDEBUG
        assert(!selected_denoted_support_id_batch.empty());
        // ############################################
        // assert(uncovered_num != -1);
        // assert(selected_denoted_support_id != -1);
        // ############################################

        std::cout << "##  sorted_uncovered_support ended  ##"
                  << std::endl;
        std::cout << "selected_denoted_support_id_batch.size(): "
                  <<  selected_denoted_support_id_batch.size()
                  << std::endl;
        std::cout << "#  remained uncovered support  #" << std::endl;

        for (auto uncovered_support_it  = sorted_uncovered_support.begin();
                  uncovered_support_it != sorted_uncovered_support. end ();
                  uncovered_support_it++){
          std::cout << uncovered_support_it->first << " # size: "
                    << uncovered_support_it->second.size()
                    << std::endl;
        }

        std::set<std::vector<DataGraphVertexHandle>> 
                   selected_denoted_supp_batch;
        for (auto& selected_denoted_support_id
                 : selected_denoted_support_id_batch) {
          auto   support_dict_it  = support_dict.find(selected_denoted_support_id.first);
          assert(support_dict_it != support_dict.end());
          auto selected_denoted_supp = support_dict_it->second.second;

          auto [ selected_denoted_supp_batch_it,
                 selected_denoted_supp_batch_ret ] 
               = selected_denoted_supp_batch.emplace(selected_denoted_supp);
          assert(selected_denoted_supp_batch_ret);
        }

        assert(selected_denoted_support_id_batch.size()
            == selected_denoted_supp_batch.size());

        // obtain the substructure of it
        auto substructure 
          = ObtainSubstructure(data_graph, selected_denoted_supp_batch, 
                                           selected_denoted_support_id_batch);
        assert(substructure.CountVertex() > 0);

        std::cout << "substructure.CountVertex() + substructure.CountEdge(): "
                  <<  substructure.CountVertex() + substructure.CountEdge() << std::endl;
        if (selected_denoted_support_id_batch.size() == 1){
          std::cout << "uncovered_num: "
                    <<  selected_denoted_support_id_batch.begin()->second 
                    << std::endl;
        }

        auto temp_support_uncovered_size
                = support_uncovered_size;
        MaintainUncoveredNum(substructure, data_graph, support_uncovered_size);
        Maintain(selected_denoted_support_id_batch,
                 support_dict,
                 multi_supp_added_size,
                 multi_supp_matched_unadded_size,
                 support_uncovered_size,
                 matched_unadded_support_set,
                           added_support_set);

        for (const auto& selected_denoted_support_id
                       : selected_denoted_support_id_batch) {
          auto temp_support_uncovered_size_erase_size
             = temp_support_uncovered_size.erase(selected_denoted_support_id.first);
          assert(temp_support_uncovered_size_erase_size != 0);
        }

        MaintainSortedUncoveredSupport(sorted_uncovered_support,
                                 temp_support_uncovered_size,
                                      support_uncovered_size);

        std::set<VertexIDType> before_merge_current_fragment_id_set;
        for (auto vertex_it = current_fragment.VertexBegin();
                !vertex_it.IsDone();
                  vertex_it++){
          auto [ before_merge_current_fragment_id_set_it,
                before_merge_current_fragment_id_set_ret]
              = before_merge_current_fragment_id_set.emplace(vertex_it->id());
          assert(before_merge_current_fragment_id_set_ret);
        }

        const auto kFragementSizeBeforeMerge
          = current_fragment.CountVertex()
          + current_fragment.CountEdge();
        std::cout << "<<  begin to merge  >>" << std::endl;
        // add it into current fragment 
        auto merge_ret = GUNDAM::MergeById(current_fragment,
                                          substructure);
        std::cout << "<<  substructure merged  >>" << std::endl;
        assert(merge_ret);
        const auto kFragementSizeAfterMerge
          = current_fragment.CountVertex()
          + current_fragment.CountEdge();
        #ifndef NDEBUG
        if (selected_denoted_support_id_batch.size() == 1){
          assert((kFragementSizeBeforeMerge + selected_denoted_support_id_batch.begin()->second )
               == kFragementSizeAfterMerge);
        }
        #endif // NDEBUG

        // remove them from denoted_support_id_set
        for (const auto& selected_denoted_support_id
                       : selected_denoted_support_id_batch) {

          auto selected_denoted_support_id_set_erase_size
             = selected_denoted_support_id_set.erase(selected_denoted_support_id.first);

          assert(((selected_denoted_support_id.second == 0) 
               &&  selected_denoted_support_id_set.empty()
               && (selected_denoted_support_id_set_erase_size == 0))
              || ((selected_denoted_support_id.second != 0) 
               && (selected_denoted_support_id_set_erase_size != 0)));

          if (selected_denoted_support_id_set_erase_size != 0){
            assert(selected_support_pattern_idx < multi_supp_added_denoted_size.size());
            multi_supp_added_denoted_size[selected_support_pattern_idx]++;
          }
        }

        std::vector<DataGraphVertexHandle> delta_vertex;
        for (auto vertex_it = current_fragment.VertexBegin();
                 !vertex_it.IsDone();
                  vertex_it++){
          if (before_merge_current_fragment_id_set.find(vertex_it->id())
           != before_merge_current_fragment_id_set.end()){
            // this vertex already exist before merge
            continue;
          }
          delta_vertex.emplace_back(vertex_it);
        }

        if (!denoted_support_only) {
          // parameter that is catched in find_new_supp_callback
          for (int support_pattern_idx = 0;
                  support_pattern_idx < supp_match_set.size();
                  support_pattern_idx++){

            auto& support_pattern = supp_match_set[support_pattern_idx].first;
      
            std::function<bool(const MatchType&)>  
                  current_find_new_supp_callback 
              = [&support_pattern,
                &support_pattern_idx,
                &find_new_supp_callback](const MatchType& match){
              int found_support_number = 0;
              return find_new_supp_callback(support_pattern, 
                                            support_pattern_idx,
                                            true, 
                                            found_support_number,
                                            match);
            };

            std::cout << "<<  " << support_pattern_idx <<"'th support incremental match begin  >>" << std::endl;
            GUNDAM::IncrementalMatchUsingMatch(support_pattern,
                                              current_fragment,
                                              delta_vertex,
                                              current_find_new_supp_callback);
            std::cout << "<<  " << support_pattern_idx <<"'th support incremental match end  >>" << std::endl;
          }
        }

        for (int support_pattern_idx = 0;
                 support_pattern_idx < multi_supp_denoted_support_id_set.size();
                 support_pattern_idx++){
          std::cout << "support_pattern_idx: "
                    <<  support_pattern_idx
                    << "\tsize: "
                    << multi_supp_denoted_support_id_set.at(support_pattern_idx).size()
                    << std::endl;
        }
        std::cout <<"added_support_set.size: "
                  << added_support_set.size()<<std::endl
                  <<"substructure.CountVertex(): "
                  << substructure.CountVertex()<<std::endl
                  <<"current_fragment.CountVertex(): "
                  << current_fragment.CountVertex()<<std::endl
                  <<"current_fragment.CountEdge(): "
                  << current_fragment.CountEdge()<<std::endl
                  <<"current_fragment.CountVertex() + current_fragment.CountEdge(): "
                  << current_fragment.CountVertex() + current_fragment.CountEdge()<<std::endl;

        support_sample_file << added_support_set.size()<<"\t"
                            << current_fragment.CountVertex()<<"\t"
                            << current_fragment.CountEdge()<<"\t"
                            << current_fragment.CountVertex()
                            + current_fragment.CountEdge()<<std::endl;

        while (kFragementSizeAfterMerge >= remained_ratio_list.begin()->first){
          ExportSampleGraph(current_fragment,
                          added_support_set,
                          kDataGraphPrefix, 
                          kSupportSampleMethod,
                          additional_name,
                          substructure_method,
                          remained_ratio_list.begin()->second,
                          begin_time);
          remained_ratio_list.erase(remained_ratio_list.begin());
          if (remained_ratio_list.empty()){
            break;
          }
        }
        if (remained_ratio_list.empty()){
          break;
        }
      }
      if (has_denoted_supp) {
        std::cout << "## all denoted support are added ##"
                  << std::endl;
        if (denoted_support_only) {
          std::cout << "## add denoted support only ##"
                    << std::endl;
          return;
        }
      }

      using CandidateSetContainer =
          std::map<GraphPatternVertexHandle, 
                  std::vector<DataGraphVertexHandle>>;

      // preprocess the candidate set for all support pattern
      std::vector<CandidateSetContainer> multi_supp_data_graph_candidate_set;
      multi_supp_data_graph_candidate_set.resize(supp_match_set.size());

      for (int support_pattern_idx = 0;
               support_pattern_idx < supp_match_set.size();
               support_pattern_idx++) {

        const bool kInitCandidateSetSucc 
          = GUNDAM::_dp_iso::InitCandidateSet<GUNDAM::MatchSemantics::kIsomorphism>(
            supp_match_set[support_pattern_idx].first,
            data_graph, 
            multi_supp_data_graph_candidate_set[support_pattern_idx]);
        if (!kInitCandidateSetSucc) {
          assert(false);
          // should not be empty
          return;
        }

        const bool kRefineCandidateSetSucc = GUNDAM::_dp_iso::RefineCandidateSet(
            supp_match_set[support_pattern_idx].first,
            data_graph, 
            multi_supp_data_graph_candidate_set[support_pattern_idx]);
        if (!kRefineCandidateSetSucc) {
          assert(false);
          // should not be empty not
          return;
        }
      }

      assert(remained_ratio_list.empty() || added_support_set.size() >= kDenotedSuppSize);

      std::vector<bool> multi_supp_all_supp_added(supp_match_set.size(), false);

      // incremental has been added back
      // for (const auto& supp_matched_unadded_size
      //          : multi_supp_matched_unadded_size) {
      //   if (supp_matched_unadded_size != 0) {
      //     // there should not be no matched but not added support
      //     std::cout << "there should not be no matched but not added support"
      //               << std::endl;
      //     // return;
      //   }
      // }

      std::set<DataGraphVertexHandle> unconsidered_vertex_in_data_graph;
      for (auto vertex_it = data_graph.VertexBegin();
               !vertex_it.IsDone();
                vertex_it++) {
        if (current_fragment.FindVertex(vertex_it->id())) {
          // this vertex is contained in current_fragment
          continue;
        }
        // this vertex is not contained in the current_fragment
        auto [ unconsidered_vertex_in_data_graph_it,
               unconsidered_vertex_in_data_graph_ret]
             = unconsidered_vertex_in_data_graph.emplace(vertex_it);
        assert(unconsidered_vertex_in_data_graph_ret);
      }

      // all denoted supports has been considered, select other
      // support according to the locality
      while (!remained_ratio_list.empty()){

        auto selected_support_pattern_idx
           = SelectDenotedSupportIdSet(multi_supp_all_supp_added, 
                                       multi_supp_added_size, 
                                       multi_supp_select_method);

        std::cout << "multi_supp_added_size:" << std::endl;
        for (const auto& added_size : multi_supp_added_size){
          std::cout << "\t" << added_size;
        }
        std::cout << std::endl;

        assert(multi_supp_denoted_support_id_set.size()
                       > selected_support_pattern_idx);
        assert(multi_supp_all_supp_added.size()
                   > selected_support_pattern_idx);
        if (multi_supp_all_supp_added[selected_support_pattern_idx]){
          // all supports are added
          #ifndef NDEBUG
          for (const auto& all_supp_added : multi_supp_all_supp_added) {
            assert(all_supp_added);
          }
          #endif
          break;
        }
        assert(multi_supp_matched_unadded_size.size() > selected_support_pattern_idx);
        std::cout << "# multi_supp_matched_unadded_size[selected_support_pattern_idx]: "
                  <<    multi_supp_matched_unadded_size[selected_support_pattern_idx]
                  << std::endl
                  << "# multi_supp_all_supp_added[selected_support_pattern_idx]: "
                  <<    multi_supp_all_supp_added[selected_support_pattern_idx]
                  << std::endl
                  << "batch_size: "
                  <<  batch_size
                  << std::endl;

        if (multi_supp_matched_unadded_size[selected_support_pattern_idx] < batch_size){
          // there  could be matched but unadded support for other pattern
          // assert(sorted_uncovered_support.empty()); 
          auto& selected_support_pattern = supp_match_set[selected_support_pattern_idx].first;

          std::function<bool(const MatchType&)>  
              current_find_new_supp_callback 
            = [&selected_support_pattern,
               &selected_support_pattern_idx,
               &multi_supp_matched_unadded_size,
               &find_new_supp_callback](const MatchType& match){
            int found_support_number = multi_supp_matched_unadded_size[selected_support_pattern_idx];
            return find_new_supp_callback(selected_support_pattern, 
                                          selected_support_pattern_idx,
                                          false,
                                          found_support_number, match);
          };

          std::vector<DataGraphVertexHandle> delta_vertex_in_data_graph;
          delta_vertex_in_data_graph.reserve(unconsidered_vertex_in_data_graph.size());
          for (const auto& vertex_handle : unconsidered_vertex_in_data_graph) {
            delta_vertex_in_data_graph.emplace_back(vertex_handle);
          }
          GUNDAM::IncrementalMatchUsingMatch(selected_support_pattern,
                                             data_graph,
                                            delta_vertex_in_data_graph,
                                            current_find_new_supp_callback);
          if (multi_supp_matched_unadded_size[selected_support_pattern_idx] == 0) {
            multi_supp_all_supp_added[selected_support_pattern_idx] = true;
            // new support for this pattern cannot be found
            // all support were added but still cannot reach 
            // the graph size
            continue;
          }
        }
        std::cout << "multi_supp_matched_unadded_size[selected_support_pattern_idx]: "
                  <<  multi_supp_matched_unadded_size[selected_support_pattern_idx]
                  << std::endl
                  << "multi_supp_all_supp_added[selected_support_pattern_idx]: "
                  <<  multi_supp_all_supp_added[selected_support_pattern_idx]
                  << std::endl;
        assert(!matched_unadded_support_set.empty());
        assert(!sorted_uncovered_support.empty());
        // select a set of support that bring minimal increament of
        // vertex and edge number
        std::map<SupportIdType, int> selected_support_id_batch;
        for (auto uncovered_support_it  = sorted_uncovered_support.begin();
                  uncovered_support_it != sorted_uncovered_support. end ();){
          if (uncovered_support_it->first == 0) {
            std::cout << uncovered_support_it->first << " # size: "
                      << uncovered_support_it->second.size()
                      << std::endl;
            // added all support whose substructure introduce
            // no new vertexes and edges into consideration
            // e.g. the substructure of those support has 
            //      already been contained in the current_fragment
            for (const auto& support_id 
                 : uncovered_support_it->second) {
              // it could be possible that all support
              // contained in the current fragment now would
              // introduce no uncovered vertex nor edge
              // 
              // set the uncovered_num and selected_support_id
              // to avoid assert failure
              const auto kSelectedAddedSizeBeforeMat
                    = multi_supp_added_size[selected_support_pattern_idx];
              // ############################################
              // uncovered_num = 0;
              // selected_support_id = support_id;
              // ############################################
              selected_support_id_batch.emplace(support_id, 0);
            }
            assert(uncovered_support_it->second.size()
                   == selected_support_id_batch.size());
            Maintain(selected_support_id_batch,
                              support_dict,
                              multi_supp_added_size,
                              multi_supp_matched_unadded_size,
                              support_uncovered_size,
                              matched_unadded_support_set,
                                        added_support_set);
            selected_support_id_batch.clear();
            uncovered_support_it = sorted_uncovered_support.erase(uncovered_support_it);
            continue;
          }
          std::cout << "uncovered_support_it->first: "
                    <<  uncovered_support_it->first
                    << std::endl;
          assert(uncovered_support_it->first != 0);
          // there are still denoted support left in sorted_uncovered_support
          // find the batch with minimal uncovered_num to add into current_fragment
          // to find whether this contains the denoted support
          bool has_moved_to_next_uncovered_support = false;
          for (auto support_id_it  = uncovered_support_it->second.begin();
                    support_id_it != uncovered_support_it->second.end();){
            const auto kSupportId = *support_id_it;
            // to find whether this vertex exists in the denoted data graph
            auto support_dict_it = support_dict.find(kSupportId);
            assert(support_dict_it != support_dict.end());
            if ( support_dict_it->second.first
              != selected_support_pattern_idx){
              // this support is not contained in selected denoted support set
              support_id_it++;
              continue;
            }
            // ############################################
            // uncovered_num = uncovered_support_it->first;
            // selected_support_id = kSupportId;
            // ############################################
            auto [ selected_support_id_batch_it,
                   selected_support_id_batch_ret ]
                 = selected_support_id_batch.emplace(kSupportId, uncovered_support_it->first);
            assert(selected_support_id_batch_ret);

            // remove this from sorted_uncovered_support
            support_id_it = uncovered_support_it->second.erase(support_id_it);
            if (uncovered_support_it->second.empty()) {
              // <*>
              uncovered_support_it = sorted_uncovered_support.erase(uncovered_support_it);
              has_moved_to_next_uncovered_support = true;
              break;
            }
            assert(selected_support_id_batch.size() <= batch_size);
            if (selected_support_id_batch.size() == batch_size){
              // has collected enough support
              break;
            }
          }
          if (!has_moved_to_next_uncovered_support){
            uncovered_support_it++;
          }
          // std::cout << "# *support_id_it loop end #" << std::endl;
          assert(selected_support_id_batch.size() <= batch_size);
          std::cout << "selected_support_id_batch.size(): "
                    <<  selected_support_id_batch.size()
                    << std::endl;
          std::cout << "#  remained uncovered support  #" << std::endl;

          for (auto uncovered_support_it  = sorted_uncovered_support.begin();
                    uncovered_support_it != sorted_uncovered_support. end ();
                    uncovered_support_it++){
            std::cout << uncovered_support_it->first << " # size: "
                      << uncovered_support_it->second.size()
                      << std::endl;
          }
          if (selected_support_id_batch.size() == batch_size){
            // has collected enough support
            // in sorted_uncovered_support, does not
            // need to further matching
            // ############################################
            // assert(uncovered_num != -1);
            // assert( selected_denoted_support_id != -1);
            // ############################################
            assert(!selected_support_id_batch.empty());
            #ifndef NDEBUG
            // at least there should be one support
            // whose uncovered substructure size not equal to zero 
            for (const auto& selected_support_id 
                           : selected_support_id_batch) {
              assert(selected_support_id.second != 0);
            }
            #endif // NDEBUG
            break;
          }
          // could be erased in <*>
          if (uncovered_support_it == sorted_uncovered_support.end()){
            break;
          }
          // std::cout << "#  here #" << std::endl;
          // uncovered_support_it++;
        }
        std::cout << "##  sorted_uncovered_support ended  ##"
                  << std::endl;
        std::cout << "selected_support_id_batch.size(): "
                  <<  selected_support_id_batch.size()
                  << std::endl;
        std::cout << "#  remained uncovered support  #" << std::endl;

        for (auto uncovered_support_it  = sorted_uncovered_support.begin();
                  uncovered_support_it != sorted_uncovered_support. end ();
                  uncovered_support_it++){
          std::cout << uncovered_support_it->first << " # size: "
                    << uncovered_support_it->second.size()
                    << std::endl;
        }
        #ifndef NDEBUG
        for (const auto& selected_support_id 
                       : selected_support_id_batch) {
          assert(selected_support_id.second != 0);
        }
        #endif // NDEBUG
        if (selected_support_id_batch.empty()){
          // assert(matched_unadded_support_set.empty());
          // assert(sorted_uncovered_support.empty());
          // assert(support_dict.empty());
          assert(multi_supp_matched_unadded_size[selected_support_pattern_idx] == 0);
          continue;
        }

        std::set<std::vector<DataGraphVertexHandle>> 
                                selected_support_batch;
        for (auto& selected_support_id
                 : selected_support_id_batch) {
          auto   support_dict_it  = support_dict.find(selected_support_id.first);
          assert(support_dict_it != support_dict.end());
          auto selected_support = support_dict_it->second.second;

          auto [ selected_support_batch_it,
                 selected_support_batch_ret ] 
               = selected_support_batch.emplace(selected_support);
          assert(selected_support_batch_ret);
        }

        assert(selected_support_id_batch.size()
            == selected_support_batch.size());

        // obtain the substructure of it
        auto substructure 
          = ObtainSubstructure(data_graph, selected_support_batch, 
                                           selected_support_id_batch);
        assert(substructure.CountVertex() > 0);

        auto temp_support_uncovered_size
                = support_uncovered_size;
        MaintainUncoveredNum(substructure, data_graph, support_uncovered_size);
        std::cout << "# before Maintain # multi_supp_matched_unadded_size[selected_support_pattern_idx]: "
                  <<  multi_supp_matched_unadded_size[selected_support_pattern_idx]
                  << std::endl
                  << "# before Maintain # multi_supp_all_supp_added[selected_support_pattern_idx]: "
                  <<  multi_supp_all_supp_added[selected_support_pattern_idx]
                  << std::endl;
        Maintain(selected_support_id_batch,
                 support_dict,
                 multi_supp_added_size,
                 multi_supp_matched_unadded_size,
                 support_uncovered_size,
                 matched_unadded_support_set,
                           added_support_set);
        std::cout << "# after Maintain # multi_supp_matched_unadded_size[selected_support_pattern_idx]: "
                  <<  multi_supp_matched_unadded_size[selected_support_pattern_idx]
                  << std::endl
                  << "# after Maintain # multi_supp_all_supp_added[selected_support_pattern_idx]: "
                  <<  multi_supp_all_supp_added[selected_support_pattern_idx]
                  << std::endl;

        for (auto& selected_support_id
                 : selected_support_id_batch) {
          auto temp_support_uncovered_size_erase_size
             = temp_support_uncovered_size.erase(selected_support_id.first);
          assert(temp_support_uncovered_size_erase_size != 0);
        }

        MaintainSortedUncoveredSupport(sorted_uncovered_support,
                                 temp_support_uncovered_size,
                                      support_uncovered_size);

        std::set<VertexIDType> before_merge_current_fragment_id_set;
        for (auto vertex_it = current_fragment.VertexBegin();
                !vertex_it.IsDone();
                  vertex_it++){
          auto [before_merge_current_fragment_id_set_it,
                before_merge_current_fragment_id_set_ret]
              = before_merge_current_fragment_id_set.emplace(vertex_it->id());
          assert(before_merge_current_fragment_id_set_ret);
        }

        const auto kFragementSizeBeforeMerge
          = current_fragment.CountVertex()
          + current_fragment.CountEdge();
        std::cout << "<<  begin to merge  >>" << std::endl;
        // add it into current fragment 
        auto merge_ret = GUNDAM::MergeById(current_fragment,
                                          substructure);
        std::cout << "<<  substructure merged  >>" << std::endl;
        assert(merge_ret);
        const auto kFragementSizeAfterMerge
          = current_fragment.CountVertex()
          + current_fragment.CountEdge();
        #ifndef NDEBUG
        if (selected_support_id_batch.size() == 1){
          assert((kFragementSizeBeforeMerge + selected_support_id_batch.begin()->second )
               == kFragementSizeAfterMerge);
        }
        #endif // NDEBUG

        std::vector<DataGraphVertexHandle> delta_vertex;
        std::set<VertexIDType> delta_vertex_id;
        for (auto vertex_it = current_fragment.VertexBegin();
                !vertex_it.IsDone();
                  vertex_it++){
          if (before_merge_current_fragment_id_set.find(vertex_it->id())
          != before_merge_current_fragment_id_set.end()){
            // this vertex already exist before merge
            continue;
          }
          delta_vertex.emplace_back(vertex_it);
          delta_vertex_id.emplace(vertex_it->id());
        }

        for (auto vertex_in_data_graph_it  = unconsidered_vertex_in_data_graph.begin();
                  vertex_in_data_graph_it != unconsidered_vertex_in_data_graph.end();) {
          if (delta_vertex_id.find((*vertex_in_data_graph_it)->id())
           == delta_vertex_id.end()){
            vertex_in_data_graph_it++;
            continue;
          }
          // this vertex is added into the current fragment, remove from
          // unconsidered_vertex_in_data_graph
          vertex_in_data_graph_it = unconsidered_vertex_in_data_graph.erase(vertex_in_data_graph_it);
        }

        for (int support_pattern_idx = 0;
                 support_pattern_idx < supp_match_set.size();
                 support_pattern_idx++){

          auto& support_pattern = supp_match_set[support_pattern_idx].first;
  
          std::function<bool(const MatchType&)>  
              current_find_new_supp_callback 
            = [&support_pattern,
              &support_pattern_idx,
              &find_new_supp_callback](const MatchType& match){
            int found_support_number = 0;
            return find_new_supp_callback(support_pattern, 
                                          support_pattern_idx,
                                          true,
                                          found_support_number, match);
          };

          std::cout << "<<<  " << support_pattern_idx <<"'th support incremental match begin  >>>" << std::endl;
          GUNDAM::IncrementalMatchUsingMatch(support_pattern,
                                            current_fragment,
                                            delta_vertex,
                                            current_find_new_supp_callback);
          std::cout << "<<<  " << support_pattern_idx <<"'th support incremental match end  >>>" << std::endl;
        }

        std::cout <<"added_support_set.size: "
                  << added_support_set.size()<<std::endl
                  <<"substructure.CountVertex(): "
                  << substructure.CountVertex()<<std::endl
                  <<"current_fragment.CountVertex(): "
                  << current_fragment.CountVertex()<<std::endl
                  <<"current_fragment.CountEdge(): "
                  << current_fragment.CountEdge()<<std::endl
                  <<"current_fragment.CountVertex() + current_fragment.CountEdge(): "
                  << current_fragment.CountVertex() + current_fragment.CountEdge()<<std::endl;
        support_sample_file<<added_support_set.size()<<"\t"
                            <<current_fragment.CountVertex()<<"\t"
                            <<current_fragment.CountEdge()<<"\t"
                            <<current_fragment.CountVertex()
                            + current_fragment.CountEdge()<<std::endl;

        while (kFragementSizeAfterMerge >= remained_ratio_list.begin()->first){
          ExportSampleGraph(current_fragment,
                          added_support_set,
                          kDataGraphPrefix, 
                          kSupportSampleMethod,
                          additional_name,
                          substructure_method,
                            remained_ratio_list.begin()->second,
                            begin_time);
          remained_ratio_list.erase(remained_ratio_list.begin());
          if (remained_ratio_list.empty()){
            break;
          }
        }
        if (remained_ratio_list.empty()){
          break;
        }
      }

      for (auto vertex_it = data_graph.VertexBegin();
               !vertex_it.IsDone(); vertex_it++){
        using AttrKeyType = std::remove_reference_t<decltype(vertex_it->AttributeBegin()->key())>;
        const AttrKeyType kSubstructureIdSetKey = "substructure_support_id_set",
                      kUncoveredSupportIdSetKey =    "uncovered_support_id_set";
        vertex_it->EraseAttribute(kSubstructureIdSetKey);
        vertex_it->EraseAttribute(kUncoveredSupportIdSetKey);
        for (auto edge_it = vertex_it->OutEdgeBegin();
                 !edge_it.IsDone(); edge_it++) {
          edge_it->EraseAttribute(kUncoveredSupportIdSetKey);
        }
      }
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
   * @brief Incremental evaluation for SupportSample.
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

#endif  // EXAMPLES_ANALYTICAL_APPS_SUPPORT_SAMPLE_SUPPORT_SAMPLE_H_