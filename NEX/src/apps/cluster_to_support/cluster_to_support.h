#ifndef EXAMPLES_ANALYTICAL_APPS_CLUSTER_TO_SUPPORT_CLUSTER_TO_SUPPORT_H_
#define EXAMPLES_ANALYTICAL_APPS_CLUSTER_TO_SUPPORT_CLUSTER_TO_SUPPORT_H_

#include <grape/grape.h>
#include <omp.h>

#include <functional>

#include "gundam/tool/same_pattern.h"

#include "gundam/match/match.h"
#include "gundam/match/matchresult.h"

#include "gundam/io/csvgraph.h"

#include "gundam/graph_type/graph.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/large_graph2.h"
#include "gundam/graph_type/small_graph.h"
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "cluster_to_support/cluster_to_support_context.h"
#include "util/file.h"

#include "gar/csv_gar.h"
#include "gar/gar.h"
#include "gar/gar_match.h"
#include "gar/gar_config.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../timer.h"

namespace grape {

template <typename FRAG_T>
class ClusterToSupport : public ParallelAppBase<FRAG_T, ClusterToSupportContext<FRAG_T>>,
                         public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
                  
  using    VertexIDType = typename ClusterToSupportContext<FRAG_T>::   VertexIDType;
  using VertexLabelType = typename ClusterToSupportContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType = typename ClusterToSupportContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType = typename ClusterToSupportContext<FRAG_T>::  EdgeLabelType;
  using    DataGraph = typename ClusterToSupportContext<FRAG_T>::DataGraph;
  using      Pattern = typename ClusterToSupportContext<FRAG_T>::  Pattern;

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(ClusterToSupport<FRAG_T>, 
                          ClusterToSupportContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  Pattern SupportPattern(VertexLabelType vertex_label){
    Pattern support_pattern;
    support_pattern.AddVertex(0, vertex_label);
    return support_pattern;
  }
  
  Pattern SupportPattern(std::tuple<VertexLabelType,
                                      EdgeLabelType,
                                    VertexLabelType> edge_info){
    Pattern support_pattern;
    support_pattern.AddVertex( 0, std::get<0>(edge_info));
    support_pattern.AddVertex( 1, std::get<2>(edge_info));
    support_pattern.AddEdge(0, 1, std::get<1>(edge_info), 0);
    return support_pattern;
  }

  /**
   * @brief Partial evaluation for ClusterToSupport.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void PEval(const fragment_t& frag, 
                    context_t& ctx,
            message_manager_t& messages) {

    using GARPatternToDataGraph 
        = gar::GraphAssociationRule<Pattern, DataGraph>;

    messages.InitChannels(thread_num());
    auto begin = timer();
    timer_next("build index");

    std::cout<<"yaml file: "<<ctx.yaml_file_<<std::endl;

    YAML::Node config = YAML::LoadFile(ctx.yaml_file_);

    if (!config["OriginalGraphPath"]) {
      std::cout << "cannot get original graph path" << std::endl;
      return;
    }
    YAML::Node original_graph_path_config = config["OriginalGraphPath"];

    if (!original_graph_path_config["GraphDir"]){
      std::cout<<"the dir of the original graph has not been been configured"<<std::endl;
      return;
    }
    const std::string kOriginalGraphDir
      = original_graph_path_config["GraphDir"].as<std::string>();

    if (!original_graph_path_config["GraphName"]){
      std::cout<<"the name of the original graph has not been been configured"<<std::endl;
      return;
    }
    const std::string kOriginalGraphName
      = original_graph_path_config["GraphName"].as<std::string>();
      
    const std::string kOriginalGraphPrefix 
                    = kOriginalGraphDir + "/" 
                    + kOriginalGraphName;

    const std::string kOriginalGraphPathVFile 
                    = kOriginalGraphPrefix + "_v.csv",
                      kOriginalGraphPathEFile 
                    = kOriginalGraphPrefix + "_e.csv";

    DataGraph original_graph;
    GUNDAM::ReadCSVGraph(original_graph, 
                        kOriginalGraphPathVFile,
                        kOriginalGraphPathEFile);

    if (!config["ClusterFile"]) {
      std::cout << "cannot get cluster_set file" << std::endl;
      return;
    }
    const std::string kClusterFile
             = config["ClusterFile"].as<std::string>();

    using ClusterIDType = int;

    // does not speficy Cluster means consider all clusters
    std::set<ClusterIDType> select_cluster_id_set;
    if (config["ClusterSet"]) {
      YAML::Node cluster_config = config["ClusterSet"];

      for (int i = 0; i < cluster_config.size(); i++){
        ClusterIDType select_cluster_id 
                          = cluster_config[i].as<ClusterIDType>();
        select_cluster_id_set.emplace(select_cluster_id);
      }
    }

    std::string seperate_by = "comma";
    if (config["SeperateBy"]) {
      seperate_by = config["SeperateBy"].as<std::string>();
      if (seperate_by != "comma"
       && seperate_by != "space"){
        std::cout << "unknown sperate method: "
                  << seperate_by
                  << std::endl;
        return;
      }
    }

    if (!config["SupportDir"]){
      std::cout << "cannot get the directory to output the support pattern and candidate" 
                << std::endl;
      return;
    }
    const std::string kSupportDir = config["SupportDir"].as<std::string>();

    bool specified_support = false;
    bool specified_support_is_edge = false;

    VertexLabelType specified_vertex_label;

    std::tuple<VertexLabelType,
                 EdgeLabelType,
               VertexLabelType> specified_edge_info;

    std::string file_prefix_for_denoted_support = "";

    if (config["Support"]){
      specified_support = true;
      YAML::Node support_config = config["Support"];
      if (!support_config["Type"]){
        std::cout << "cannot get the type of the specified support" 
                  << std::endl;
        return;
      }
      std::string support_type = support_config["Type"].as<std::string>();
      if (support_type != "edge"
       && support_type != "vertex"){
        std::cout << "unknown support type: "
                  << support_type << std::endl;
        return;
      }
      specified_support_is_edge = (support_type == "edge");

      if (!support_config["FilePrefix"]){
        std::cout << "cannot get the output file for the specified support" 
                  << std::endl;
        return;
      }
      file_prefix_for_denoted_support = support_config["FilePrefix"].as<std::string>();

      if (specified_support_is_edge){
        if (!support_config["SrcLabel"]){
          std::cout << "cannot get the src label of specified support" 
                    << std::endl;
          return;
        }
        std::get<0>(specified_edge_info)
                  = support_config["SrcLabel"].as<VertexLabelType>();

        if (!support_config["DstLabel"]){
          std::cout << "cannot get the dst label of specified support" 
                    << std::endl;
          return;
        }
        std::get<2>(specified_edge_info)
                  = support_config["DstLabel"].as<VertexLabelType>();
        
        if (!support_config["EdgeLabel"]){
          std::cout << "cannot get the edge label of specified support" 
                    << std::endl;
          return;
        }
        std::get<1>(specified_edge_info)
                  = support_config["EdgeLabel"].as<EdgeLabelType>();
      }
      else{
        assert(support_type == "vertex");
        if (!support_config["VertexLabel"]){
          std::cout << "cannot get the vertex label of specified support" 
                    << std::endl;
          return;
        }
        specified_vertex_label = support_config["VertexLabel"].as<VertexLabelType>();
      }
    }

    std::string sample_method = "in_all_clusters";
    double support_sample_ratio = 1.0;
    int sample_set = 1;
    int small_cluster_threshold;

    if (config["Sample"]){
      YAML::Node sample_config = config["Sample"];
      if (!sample_config["Method"]){
        std::cout << "does not specify the sample method" << std::endl;
        return;
      }
      sample_method = sample_config["Method"].as<std::string>();
      
      if ((sample_method != "in_all_clusters")
       && (sample_method != "in_each_cluster")
       && (sample_method != "contain_all_small_cluster")){
        std::cout << "unknown sample method: " << sample_method << std::endl;
        return;
      }

      if (!sample_config["Ratio"]) {
        std::cout << "does not specify the sample ratio" << std::endl;
        return;
      }
      support_sample_ratio = sample_config["Ratio"].as<double>();
      
      if (sample_config["Set"]){
        sample_set = sample_config["Set"].as<int>();
        if (sample_set <= 0){
          std::cout << "illegal sample set: " << sample_set 
                    << " should be >= 1"
                    << std::endl;
          return;
        }
      }

      if (sample_method == "contain_all_small_cluster"){
        if (!sample_config["Threshold"]){
          std::cout << "the sample method is specified as \"contain_all_small_cluster\""
                    << " but the threshold is not specified"
                    << std::endl;
          return;
        }
        small_cluster_threshold = sample_config["Threshold"].as<int>();
      }
    }

    std::ifstream cluster_file(kClusterFile);

    std::map <ClusterIDType, std::set<VertexIDType> > cluster_set;

    while (cluster_file) {
      std::string s;
      if (!std::getline( cluster_file, s )) 
        break;

      std::istringstream ss( s );
      std::vector <std::string> record;
      record.reserve(2);
      if (seperate_by == "comma"){
        while (ss) {
          std::string s;
          if (!std::getline( ss, s, ',' )) {
            break;
          }
          record.push_back( s );
        }
      }
      else{
        assert(seperate_by == "space");
        std::string buf; 
        while (ss >> buf)
          record.emplace_back(buf);
      }
      assert(record.size() == 2);
      auto vertex_id = GUNDAM::StringToDataType<VertexIDType>(record[0]);
      int cluster_id = std::stoi(record[1]);
      auto [cluster_it,
            cluster_ret] = cluster_set.emplace(cluster_id, std::set<VertexIDType>());
      auto [vertex_id_it,
            vertex_id_ret] = cluster_it->second.emplace(vertex_id);
      // should added successfully
      assert(vertex_id_ret);
    }

    for (const auto& cluster : cluster_set){
      std::cout<<"cluster: "<<cluster.first
               <<"\t size: "<<cluster.second.size()<<std::endl;
      std::cout<<std::endl;
    }

    for (const auto& select_cluster_id 
                   : select_cluster_id_set){
      auto cluster_it  = cluster_set.find(select_cluster_id);
      if ( cluster_it == cluster_set.end()){
        // select_cluster_id cannot be found in cluster
        std::cout <<"user selected cluster id: "
                  << select_cluster_id
                  <<" cannot be found in cluster set"<<std::endl;
        return;
      }
    }

    if (select_cluster_id_set.empty()){
      // does not specify cluster id, means consider all clusters
      for (const auto& cluster : cluster_set){
        auto [ select_cluster_id_set_it,
               select_cluster_id_set_ret]
             = select_cluster_id_set.emplace(cluster.first);
        // should added successfully
        assert(select_cluster_id_set_ret);
      }
      assert(select_cluster_id_set.size()
                    == cluster_set.size());
    }

    VertexIDType max_vertex_id = 0;
    for (auto vertex_it = original_graph.VertexBegin();
             !vertex_it.IsDone();
              vertex_it++) {
      if (max_vertex_id < vertex_it->id()){
        max_vertex_id = vertex_it->id();
      }
    }

    EdgeIDType min_edge_id = std::numeric_limits<EdgeIDType>::max();
    for (auto vertex_cit = original_graph.VertexBegin();
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

    const auto kEdgeToVertexOffset = max_vertex_id
                                   - min_edge_id + 1;

    using VertexHandleType = typename GUNDAM::VertexHandle<DataGraph>::type;
    using   EdgeHandleType = typename GUNDAM::  EdgeHandle<DataGraph>::type;
    
    std::set<VertexHandleType> specified_vertex_set;

    using VertexSetType = std::map<VertexLabelType, 
                                   std::set<VertexHandleType>>;
                                   
    using   EdgeSetType = std::map<std::tuple<VertexLabelType,
                                                EdgeLabelType,
                                              VertexLabelType>, 
                                    std::set<  EdgeHandleType>>;

    std::map<int, std::pair<VertexSetType, 
                              EdgeSetType>> cluster_support_set;

    for (const auto& select_cluster_id 
                   : select_cluster_id_set){
      auto [cluster_support_it,
            cluster_support_ret] 
          = cluster_support_set.emplace(select_cluster_id,
                                std::pair<VertexSetType,
                                            EdgeSetType>());
      assert(cluster_support_ret);
      VertexSetType&    vertex_set_by_label = cluster_support_it->second.first;
        EdgeSetType& edge_info_set_by_label = cluster_support_it->second.second;
        
      auto cluster_it = cluster_set.find(select_cluster_id);
      assert(cluster_it != cluster_set.end());
      for (const auto& vertex_id : cluster_it->second){
        auto vertex_handle = original_graph.FindVertex(vertex_id);
        if (vertex_handle){
          // this vertex in the cluster of converted graph is a vertex
          // in original graph
          auto [vertex_set_by_label_it,
                vertex_set_by_label_ret] 
              = vertex_set_by_label.emplace(vertex_handle->label(), std::set<VertexHandleType>());
                
          auto [vertex_set_it,
                vertex_set_ret] 
              = vertex_set_by_label_it->second.emplace(vertex_handle);
          
          // should added successfully
          assert(vertex_set_ret);
          continue;
        }
        util::Debug("kEdgeToVertexOffset: " + std::to_string(kEdgeToVertexOffset));
        util::Debug("vertex_id: " + std::to_string(vertex_id));
        // this vertex corresponds to an edge in original graph
        auto edge_handle = original_graph.FindEdge(vertex_id - kEdgeToVertexOffset);
        // should find that
        assert(edge_handle);

        std::tuple edge_info(edge_handle->src_handle()->label(),
                             edge_handle->label(),
                             edge_handle->dst_handle()->label());

        auto [edge_info_set_by_label_it,
              edge_info_set_by_label_ret] 
            = edge_info_set_by_label.emplace(edge_info, std::set<EdgeHandleType>());
              
        auto [edge_info_set_it,
              edge_info_set_ret]
            = edge_info_set_by_label_it->second.emplace(edge_handle);
        assert(edge_info_set_ret);
      }

      std::cout<<"cluster id: "<<select_cluster_id<<std::endl;

      for (const auto& vertex_set 
                     : vertex_set_by_label) {
        std::cout<<"\tvertex label: " <<vertex_set.first<<std::endl
                 <<"\tvertex number: "<<vertex_set.second.size()<<std::endl;

        const std::string kSuppPrefix 
                        = kSupportDir + "/"
                        + "cluster_" 
                        + std::to_string(select_cluster_id)
                        + "_vertex_label_"
                        + GUNDAM::ToString(vertex_set.first);

        Pattern support_pattern(SupportPattern(vertex_set.first));
        GUNDAM::MatchSet support_match_set(support_pattern,
                                           original_graph);

        // GUNDAM::WriteCSVGraph(support_pattern, 
        //                      kSuppPrefix + "_v.csv",
        //                      kSuppPrefix + "_e.csv");

        using PatternVertexHandle = typename GUNDAM::VertexHandle<Pattern>::type;
        PatternVertexHandle support_vertex_handle 
                          = support_pattern.FindVertex(0);
        for (auto& dst_vertex_handle : vertex_set.second){
          GUNDAM::Match<Pattern, DataGraph> support_match;
          auto map_ret = support_match.AddMap(support_vertex_handle,
                                                  dst_vertex_handle);
          assert(map_ret);

          auto [match_it, match_ret] 
                = support_match_set.AddMatch(support_match);
          assert(match_ret);
        }

        // std::ofstream support_match_file(kSuppPrefix + "_supp.txt");

        // GUNDAM::MatchSetToFile(support_match_set, 
        //                        support_match_file);
      }

      for (const auto& edge_info_set : edge_info_set_by_label) {
        std::cout<<"\tsrc vertex label: " <<std::get<0>(edge_info_set.first)<<std::endl
                 <<"\tedge label: "       <<std::get<1>(edge_info_set.first)<<std::endl
                 <<"\tdst vertex label: " <<std::get<2>(edge_info_set.first)<<std::endl
                 <<"\tedge number: "<<edge_info_set.second.size()<<std::endl;
                 
        Pattern support_pattern(SupportPattern(edge_info_set.first));

        const std::string kSuppPrefix 
                        = kSupportDir + "/"
                        + "cluster_" 
                        + std::to_string(select_cluster_id)
                        + "_src_label_" 
                        + GUNDAM::ToString( std::get<0>(edge_info_set.first) )
                        + "_edge_label_" 
                        + GUNDAM::ToString( std::get<1>(edge_info_set.first) )
                        + "_dst_label_" 
                        + GUNDAM::ToString( std::get<2>(edge_info_set.first) );

        // GUNDAM::WriteCSVGraph(support_pattern, 
        //                      kSuppPrefix + "_v.csv",
        //                      kSuppPrefix + "_e.csv");

        GUNDAM::MatchSet support_match_set(support_pattern,
                                           original_graph);

        using PatternVertexHandle = typename GUNDAM::VertexHandle<Pattern>::type;

        PatternVertexHandle support_src_vertex_handle 
                          = support_pattern.FindVertex(0),
                            support_dst_vertex_handle 
                          = support_pattern.FindVertex(1);

        for (auto& edge_handle : edge_info_set.second){
          GUNDAM::Match<Pattern, DataGraph> support_match;
          auto src_vertex_handle = edge_handle->src_handle(),
               dst_vertex_handle = edge_handle->dst_handle();
          auto src_map_ret = support_match.AddMap(
                             support_src_vertex_handle,
                                     src_vertex_handle);
          assert(src_map_ret);
                                     
          auto dst_map_ret = support_match.AddMap(
                             support_dst_vertex_handle,
                                     dst_vertex_handle);
          assert(dst_map_ret);

          auto [match_it, match_ret] 
                = support_match_set.AddMatch(support_match);
          assert(match_ret);
        }

        // std::ofstream support_match_file(kSuppPrefix + "_supp.txt");

        // GUNDAM::MatchSetToFile(support_match_set, 
        //                        support_match_file);
      }
      if (specified_support && !specified_support_is_edge){
        auto vertex_set_by_label_it 
           = vertex_set_by_label.find(specified_vertex_label);
        if ( vertex_set_by_label_it
          == vertex_set_by_label.end()) {
          // this cluster does not contain the specified vertex label
          continue;
        }
        assert(vertex_set_by_label_it
            != vertex_set_by_label.end());
        specified_vertex_set.insert(
                  vertex_set_by_label_it->second.begin(),
                  vertex_set_by_label_it->second. end ());
        continue;
      }
    }

    if (specified_support){
      if (specified_support_is_edge){

        util::Info("in specified_support_is_edge");

        Pattern support_pattern(SupportPattern(specified_edge_info));

        using PatternVertexHandle = typename GUNDAM::VertexHandle<Pattern>::type;
        
        PatternVertexHandle support_src_vertex_handle 
                          = support_pattern.FindVertex(0),
                            support_dst_vertex_handle 
                          = support_pattern.FindVertex(1);

        assert(support_src_vertex_handle);
        assert(support_dst_vertex_handle);

        for (int i = 0; i < sample_set; i++){
          std::string sample_idx_posfix = "";
          if (sample_set != 1) {
            assert(sample_set > 1);
            sample_idx_posfix = "_" + std::to_string(i);
          }

          const std::string kSupportStr 
                    = "_s" + std::to_string(std::get<0>(specified_edge_info))
                    + "_e" + std::to_string(std::get<1>(specified_edge_info))
                    + "_d" + std::to_string(std::get<2>(specified_edge_info));

          std::set<EdgeHandleType> selected_edge_set;

          if (sample_method == "in_all_clusters") {
            std::set<EdgeHandleType> specified_edge_set;
            // gather the selected edge from all clusters
            for (const auto& cluster_support_it 
                           : cluster_support_set){
              auto& edge_info_set_by_label = cluster_support_it.second
                                                               .second;
              auto edge_info_set_by_label_it 
                 = edge_info_set_by_label.find(specified_edge_info);
              if ( edge_info_set_by_label_it
                == edge_info_set_by_label.end()) {
                // this cluster does not contain the specified vertex label
                continue;
              }
              assert(edge_info_set_by_label_it
                  != edge_info_set_by_label.end());
              specified_edge_set.insert(
                edge_info_set_by_label_it->second.begin(),
                edge_info_set_by_label_it->second. end ());
            }
            // then sample from it
            for (const auto& edge_handle : specified_edge_set){
              const double kR = ((double) rand() / (RAND_MAX));
              if (kR > support_sample_ratio) {
                // this support should not be preserved
                continue;
              }
              auto [ selected_edge_it, 
                     selected_edge_ret ] 
                   = selected_edge_set.emplace(edge_handle);
              assert(selected_edge_ret);
            }
          }
          else if (sample_method == "in_each_cluster") {
            // sample in each cluster
            for (const auto& cluster_support_it 
                           : cluster_support_set){
              auto& edge_info_set_by_label = cluster_support_it.second
                                                               .second;
              auto edge_info_set_by_label_it 
                 = edge_info_set_by_label.find(specified_edge_info);
              if ( edge_info_set_by_label_it
                == edge_info_set_by_label.end()) {
                // this cluster does not contain the specified vertex label
                continue;
              }
              assert(edge_info_set_by_label_it
                  != edge_info_set_by_label.end());
              while (true){
                // should at least select one from it
                bool has_found_one = false;
                for (auto& edge_handle 
                         : edge_info_set_by_label_it->second) {
                  const double kR = ((double) rand() / (RAND_MAX));
                  if (kR > support_sample_ratio) {
                    // this support should not be preserved
                    continue;
                  }
                  has_found_one = true;
                  auto [ selected_edge_it,
                         selected_edge_ret ]
                       = selected_edge_set.emplace(edge_handle);
                  assert(selected_edge_ret);
                }
                if (has_found_one){
                  break;
                }
              }
            }
          }
          else if (sample_method == "contain_all_small_cluster") {
            // contains all small clusters, then sample from other
            // bigger cluster, 
            std::set<EdgeHandleType> specified_edge_set,
                                 small_cluster_edge_set,
                                   big_cluster_edge_set;
            int small_cluster_counter = 0,
                  big_cluster_counter = 0;
            for (const auto& cluster_support_it 
                           : cluster_support_set){
              auto& edge_info_set_by_label = cluster_support_it.second
                                                               .second;
              auto edge_info_set_by_label_it 
                 = edge_info_set_by_label.find(specified_edge_info);
              if ( edge_info_set_by_label_it
                == edge_info_set_by_label.end()) {
                // this cluster does not contain the specified vertex label
                continue;
              }
              assert(edge_info_set_by_label_it
                  != edge_info_set_by_label.end());
              specified_edge_set.insert(
                  edge_info_set_by_label_it->second.begin(),
                  edge_info_set_by_label_it->second. end ());
              if (edge_info_set_by_label_it->second.size() <= small_cluster_threshold) {
                // this cluster is small enough
                small_cluster_counter++;
                small_cluster_edge_set.insert(
                  edge_info_set_by_label_it->second.begin(),
                  edge_info_set_by_label_it->second. end ());
                continue;
              }
              // this cluster is not small enough
              big_cluster_counter++;
              big_cluster_edge_set.insert(
                edge_info_set_by_label_it->second.begin(),
                edge_info_set_by_label_it->second. end ());
            }
            if (selected_edge_set.size() >= support_sample_ratio 
                                          * specified_edge_set.size()) {
              const int kRemainedSelectedEdgeSetSize = support_sample_ratio 
                                                    * ((double)(specified_edge_set.size()))
                                                    * ((double) small_cluster_counter) 
                                                    / ((double)(small_cluster_counter + big_cluster_counter));
              const double kSelectedEdgeSampleRatio = ((double)kRemainedSelectedEdgeSetSize)
                                                    / ((double)selected_edge_set.size());
              std::set<EdgeHandleType> temp_selected_edge_set;
              for (auto& edge_handle : selected_edge_set){
                const double kR = ((double) rand() / (RAND_MAX));
                if (kR > kSelectedEdgeSampleRatio) {
                  // this support should not be preserved
                  continue;
                }
                auto [ temp_selected_edge_it,
                       temp_selected_edge_ret ]
                     = temp_selected_edge_set.emplace(edge_handle);
                assert(temp_selected_edge_ret);
              }
              selected_edge_set.swap(temp_selected_edge_set);
            }
            const int kSelectSizeFromBigCluster = support_sample_ratio 
                                                * specified_edge_set.size() 
                                                -  selected_edge_set.size();
            const double kSelectFromBigClusterSampleRatio 
              = ((double)kSelectSizeFromBigCluster)
              / ((double)big_cluster_edge_set.size());
            for (auto& edge_handle : big_cluster_edge_set){
              const double kR = ((double) rand() / (RAND_MAX));
              if (kR > kSelectFromBigClusterSampleRatio) {
                // this support should not be preserved
                continue;
              }
              auto [ selected_edge_it,
                     selected_edge_ret ]
                   = selected_edge_set.emplace(edge_handle);
              assert(selected_edge_ret);
            }
          }
          else{
            assert(false);
          }

          GUNDAM::MatchSet support_match_set(support_pattern,
                                             original_graph);

          for (const auto& selected_edge_handle : selected_edge_set) {
            GUNDAM::Match<Pattern, DataGraph> support_match;

            auto src_vertex_handle = selected_edge_handle->src_handle(),
                 dst_vertex_handle = selected_edge_handle->dst_handle();

            auto src_map_ret = support_match.AddMap(
                                support_src_vertex_handle,
                                        src_vertex_handle);
            assert(src_map_ret);
                                      
            auto dst_map_ret = support_match.AddMap(
                                support_dst_vertex_handle,
                                        dst_vertex_handle);
            assert(dst_map_ret);

            auto [match_it, match_ret] 
                  = support_match_set.AddMatch(support_match);
            assert(match_ret);
          }

          const std::string kSupportPrefix
                          = file_prefix_for_denoted_support 
                          + kSupportStr
                          + "_sample_method_"
                          +   sample_method
                          +   sample_idx_posfix 
                          + "_sample_ratio_"
                          + std::to_string(support_sample_ratio);
          
          GUNDAM::WriteCSVGraph(support_pattern, 
                               kSupportPrefix + "_v.csv",
                               kSupportPrefix + "_e.csv");

          std::ofstream support_match_file(kSupportPrefix + "_supp.txt");

          util::Debug("selected_edge_set.size(): " + std::to_string(selected_edge_set.size()));

          util::Debug("support_match_set.size(): " + std::to_string(support_match_set.size()));

          // getchar();

          GUNDAM::MatchSetToFile(support_match_set, 
                                 support_match_file);
        }
      }
      else{
        Pattern support_pattern(SupportPattern(specified_vertex_label));

        // specified support is vertex
        using PatternVertexHandle = typename GUNDAM::VertexHandle<Pattern>::type;
        
        GUNDAM::MatchSet support_match_set(support_pattern,
                                           original_graph);
        
        PatternVertexHandle support_vertex_handle 
                          = support_pattern.FindVertex(0);


        // for (auto& dst_vertex_handle : specified_vertex_set){
        //   GUNDAM::Match<Pattern, DataGraph> support_match;
        //   auto map_ret = support_match.AddMap(support_vertex_handle,
        //                                           dst_vertex_handle);
        //   assert(map_ret);

        //   auto [match_it, match_ret] 
        //         = support_match_set.AddMatch(support_match);
        //   assert(match_ret);
        // }
        
        GUNDAM::WriteCSVGraph(support_pattern, 
             file_prefix_for_denoted_support + "_v.csv",
             file_prefix_for_denoted_support + "_e.csv");
        
        std::ofstream support_match_file(file_prefix_for_denoted_support + "_supp.txt");

        GUNDAM::MatchSetToFile(support_match_set, 
                               support_match_file);
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
   * @brief Incremental evaluation for ClusterToSupport.
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

#endif  // EXAMPLES_ANALYTICAL_APPS_CLUSTER_TO_SUPPORT_CLUSTER_TO_SUPPORT_H_