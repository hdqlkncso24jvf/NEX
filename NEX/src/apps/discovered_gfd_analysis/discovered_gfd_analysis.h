#ifndef EXAMPLES_ANALYTICAL_APPS_DISCOVERED_GFD_ANALYSIS_DISCOVERED_GFD_ANALYSIS_H_
#define EXAMPLES_ANALYTICAL_APPS_DISCOVERED_GFD_ANALYSIS_DISCOVERED_GFD_ANALYSIS_H_

#include <grape/grape.h>
#include <omp.h>

#include <functional>

#include "gundam/tool/connected.h"
#include "gundam/tool/isolate_vertex.h"

#include "gundam/graph_type/graph.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/large_graph2.h"
#include "gundam/graph_type/small_graph.h"

#include "gundam/match/match.h"
#include "gundam/match/matchresult.h"

#include "gundam/io/csvgraph.h"
#include "gundam/io/csvgraph.h"

#include "gundam/type_getter/edge_handle.h"
#include "gundam/type_getter/vertex_handle.h"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "util/file.h"

#include "discovered_gfd_analysis/discovered_gfd_analysis_context.h"

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
class DiscoveredGfdAnalysis : public ParallelAppBase<FRAG_T, DiscoveredGfdAnalysisContext<FRAG_T>>,
                              public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
  using    VertexIDType  = typename DiscoveredGfdAnalysisContext<FRAG_T>::   VertexIDType;
  using    VertexPtrType = typename DiscoveredGfdAnalysisContext<FRAG_T>::   VertexPtrType;
  using VertexLabelType  = typename DiscoveredGfdAnalysisContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType  = typename DiscoveredGfdAnalysisContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType  = typename DiscoveredGfdAnalysisContext<FRAG_T>::  EdgeLabelType;
  using         Pattern  = typename DiscoveredGfdAnalysisContext<FRAG_T>::  Pattern;
  using       DataGraph  = typename DiscoveredGfdAnalysisContext<FRAG_T>::DataGraph;

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(DiscoveredGfdAnalysis<FRAG_T>, 
                          DiscoveredGfdAnalysisContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  inline std::vector<std::string> GarPathSet(const std::string& gar_dir,
                                             const std::set<std::string>& rule_prefix_set) const {
    const std::string kRulePosfix = "_v.csv";

    std::vector<std::string> qualified_gar_path_set_in_one_dir;

    std::set<std::string> dir_files;

    util::GetFiles(gar_dir, dir_files);
    for (const auto& rule_prefix : rule_prefix_set){
      for (auto gar_file : dir_files){
        const bool isPrefix = rule_prefix.size() 
                              <= gar_file.size() 
            && std::mismatch(rule_prefix.begin(), 
                             rule_prefix. end (),
                                gar_file.begin(), 
                                gar_file. end ()).first == rule_prefix.end();
        const bool isPosfix = kRulePosfix.size() 
                              <= gar_file.size() 
             && std::mismatch(kRulePosfix.begin(), 
                              kRulePosfix. end (),
                                 gar_file.end() - kRulePosfix.size(), 
                                 gar_file.end()).first == kRulePosfix.end();
        if (!isPrefix || !isPosfix){
          continue;
        }

        gar_file = gar_file.substr(0, gar_file.length()
                                 - kRulePosfix.size());
        qualified_gar_path_set_in_one_dir.emplace_back(gar_dir + "/" + gar_file);
      }
    }
    return std::move(qualified_gar_path_set_in_one_dir);
  }

  /**
   * @brief Partial evaluation for DiscoveredGfdAnalysis.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void PEval(const fragment_t& frag, 
                    context_t& ctx,
            message_manager_t& messages) {

    using GarType = gar::GraphAssociationRule<Pattern, DataGraph>;

    messages.InitChannels(thread_num());
    auto begin = timer();
    timer_next("build index");

    std::cout<<"yaml file: "<<ctx.yaml_file_<<std::endl;

    YAML::Node config = YAML::LoadFile(ctx.yaml_file_);

    std::vector<std::string> gars_path;
                             
    std::set<std::string> gar_prefix_set = {"gar_level_"};
    if (config["GarPrefix"]){
      gar_prefix_set.clear();
      gar_prefix_set.emplace(config["GarPrefix"].as<std::string>());
    }
    if (config["GarDir"]) {
      const std::string kGarDir = config["GarDir"].as<std::string>();
      gars_path = GarPathSet(kGarDir, gar_prefix_set);
    }
    else {
      if (!config["GarNames"]){
        std::cout << "cannot get gars!" << std::endl;
        return;
      }
      YAML::Node gar_names_config = config["GarNames"]; 
      gars_path.reserve(gar_names_config.size());
      for(int i=0; i < gar_names_config.size(); ++i){
        const std::string kGarName = gar_names_config[i].as<std::string>();
        gars_path.emplace_back(kGarName);
      }
    }

    std::vector<std::pair<GarType, std::string>> gar_set;
    gar_set.reserve(gars_path.size());
    for (const auto& gar_path : gars_path){
      const std::string kGarVFile = gar_path + "_v.csv",
                        kGarEFile = gar_path + "_e.csv",
                        kGarXFile = gar_path + "_x.csv",
                        kGarYFile = gar_path + "_y.csv";
      GarType gar;
      auto res = gar::ReadGAR(gar, kGarVFile, kGarEFile,
                                   kGarXFile, kGarYFile);
      if (res < 0){
        std::cout<<"loading gar: "<<gar_path<<" failed! :"<<res<<std::endl;
        return;
      }
      gar_set.emplace_back(std::move(gar), gar_path);
    }

    // randomly remove an edge from the pattern that can
    // still preserve the remained pattern connected
    for (const auto& gar : gar_set){
      // enumerate all edges in pattern
      for (auto edge_it = gar.first.pattern().EdgeBegin();
               !edge_it.IsDone();
                edge_it++){
        Pattern src_pattern(gar.first.pattern());
        src_pattern.EraseEdge(edge_it->id());
        const Pattern& kSrcGarPattern0 = src_pattern;
        if (!GUNDAM::Connected(kSrcGarPattern0)){
          // verify whtether such a disconnection is caused by
          // the isolated vertex
          VertexPtrType iso_vertex_ptr = nullptr;
          assert(!iso_vertex_ptr);
          for (auto vertex_it = src_pattern.VertexBegin();
                   !vertex_it.IsDone();
                    vertex_it++){
            if (vertex_it->CountOutVertex() == 0
             && vertex_it->CountInVertex() == 0){
              iso_vertex_ptr = vertex_it;
              assert(iso_vertex_ptr);
              break;
            }
          }
          if (!iso_vertex_ptr){
            // does not find iso vertex
            continue;
          }
          src_pattern.EraseVertex(iso_vertex_ptr->id());
        }
        const Pattern& kSrcGarPattern = src_pattern;
        assert(GUNDAM::Connected(kSrcGarPattern));
        GarType src_gar(kSrcGarPattern);
        // construct a new gar without removed vertexes
        for (const auto& literal_ptr : gar.first.x_literal_set()){
          std::vector<typename GUNDAM::VertexHandle<Pattern>::type> literal_vertex_ptr_set;

          literal_ptr->CalPivot(literal_vertex_ptr_set);
          assert(!literal_vertex_ptr_set.empty());

          bool all_vertexes_contained = true;
          for (const auto& vertex_ptr
                 : literal_vertex_ptr_set){
            if (kSrcGarPattern.FindVertex(vertex_ptr->id())){
              continue;
            }
            all_vertexes_contained = false;
            break;
          }
          if (!all_vertexes_contained){
            // discard this literal
            continue;
          }
          // preserve this literal in new constructed literal
          src_gar.AddX(literal_ptr->info());
        }
        for (const auto& literal_ptr : gar.first.y_literal_set()){
          std::vector<typename GUNDAM::VertexHandle<Pattern>::type> literal_vertex_ptr_set;

          literal_ptr->CalPivot(literal_vertex_ptr_set);
          assert(!literal_vertex_ptr_set.empty());

          bool all_vertexes_contained = true;
          for (const auto& vertex_ptr
                 : literal_vertex_ptr_set){
            if (kSrcGarPattern.FindVertex(vertex_ptr->id())){
              continue;
            }
            all_vertexes_contained = false;
            break;
          }
          if (!all_vertexes_contained){
            // discard this literal
            continue;
          }
          src_gar.AddY(literal_ptr->info());
        }
        if (src_gar.y_literal_set().Empty()){
          // does not have legal y
          continue;
        }
        const auto& kSrcGar = src_gar;
        bool duplicated = false;
        for (const auto& dst_gar : gar_set){
          const auto& kDstGar     = dst_gar.first;
          const auto& kDstGarName = dst_gar.second;

          if (!gar::SameGar(kSrcGar, kDstGar)){
            // kSrcGarPattern and kDstPattern is not the
            // same
            continue;
          }
          // the patterns of these two gars are the same
          duplicated = true;
          break;
        }
        if (!duplicated) {
          std::cout << " # Illegal Catched # "<<std::endl
                    << " gar: "<<gar.second
                    << " with edge: " << edge_it->id()
                    << " removed does not exist in the rest of the gars"
                    << std::endl;
        }
      }
    }

    std::cout << "#############################" << std::endl
              << "# discovered gfd dir tested #" << std::endl
              << "#############################" << std::endl;

    if (config["DataGraph"]) {
      // denoted data graph, verify the support on it
      YAML::Node data_graph_config = config["DataGraph"];

      if (!data_graph_config["GraphDir"]){
        std::cout<<"the dir of the data graph has not been been configured"<<std::endl;
        return;
      }
      const std::string kDataGraphDir
        = data_graph_config["GraphDir"].as<std::string>();

      if (!data_graph_config["GraphName"]){
        std::cout<<"the name of the data graph has not been been configured"<<std::endl;
        return;
      }
      const std::string kDataGraphName
        = data_graph_config["GraphName"].as<std::string>();


      uint64_t support_bound = 1;
      if (data_graph_config["SupportBound"]){
        support_bound = config["SupportBound"].as<uint64_t>();
      }

      DataGraph data_graph;

      int res = GUNDAM::ReadCSVGraph(data_graph, 
              kDataGraphDir + "/" + kDataGraphName + "_v.csv",
              kDataGraphDir + "/" + kDataGraphName + "_e.csv");

      if (res < 0){
        std::cout<<"load data graph error! :"<<res<<std::endl;
        return;
      }

      for (auto& gar : gar_set){
        auto [x_supp, xy_supp] 
          = gar::GarSupp(gar.first, data_graph, 
                        -1, 1200.0);

        assert(xy_supp >= 0);

        if (xy_supp < support_bound){
          std::cout<<" # Illegal Catched #"<<std::endl
                  << " gar: "<<gar.second
                  << " does not satisfy support bound in data graph: "
                  << kDataGraphDir + "/" + kDataGraphName
                  << std::endl;
        }
      }
      std::cout << "################################" << std::endl
                << "# support on data graph tested #" << std::endl
                << "################################" << std::endl;
    }

    if (config["SubGraphGar"]) {
      // configured subgraph gfds, verify whether they are discoverd
      // in this original graph
      YAML::Node sub_graph_gfd_configs = config["SubGraphGar"]; 
      for(int i=0; i < sub_graph_gfd_configs.size(); ++i){

        YAML::Node sub_graph_gfd_config 
                 = sub_graph_gfd_configs[i]; 


        std::set<std::string> sub_graph_gar_prefix_set = {"gar_level_"};
        if (sub_graph_gfd_config["GarPrefix"]){
          sub_graph_gar_prefix_set.clear();
          sub_graph_gar_prefix_set.emplace(sub_graph_gfd_config["GarPrefix"].as<std::string>());
        }
        if (!sub_graph_gfd_config["GarDir"]) {
          std::cout<<"the directory of gars discovered from sub graph is not denoted! "<<std::endl;
          continue;
        }
        const std::string kSubGraphGarDir = sub_graph_gfd_config["GarDir"].as<std::string>();
        std::vector<std::string> sub_graph_gars_path 
                   = GarPathSet(kSubGraphGarDir, sub_graph_gar_prefix_set);

        for (const auto& sub_graph_gar_path 
                       : sub_graph_gars_path){
          const std::string 
             kGarVFile = sub_graph_gar_path + "_v.csv",
             kGarEFile = sub_graph_gar_path + "_e.csv",
             kGarXFile = sub_graph_gar_path + "_x.csv",
             kGarYFile = sub_graph_gar_path + "_y.csv";
          GarType sub_graph_gar;
          auto res = gar::ReadGAR(sub_graph_gar, 
                                  kGarVFile, kGarEFile,
                                  kGarXFile, kGarYFile);
          if (res < 0){
            std::cout<<" loading gar: "<<sub_graph_gar_path
                     <<" failed! :"<<res<<std::endl;
            return;
          }

          const auto& kSubGraphGar     = sub_graph_gar;
          const auto& kSubGraphGarName = sub_graph_gar_path;

          bool duplicated = false;
          for (const auto& gar : gar_set){
            const auto& kGar     = gar.first;
            const auto& kGarName = gar.second;

            if (!gar::SameGar(kSubGraphGar, kGar)){
              // this two gar is not the same 
              continue;
            }
            duplicated = true;
            break;
          }
          if (!duplicated){
            std::cout<<" # Illegal Catched #"<<std::endl
                     <<" gar: "<<kSubGraphGarName
                     <<" cannot be found in the gars discovered from original graph! "<<std::endl;
          }
        }
        std::cout << "########################################" << std::endl
                  << "# gars from subgraph in: " + kSubGraphGarDir + " analysised #" << std::endl
                  << "########################################" << std::endl;
      }
    }

    // std::vector<std::pair<GarType, std::string>> sub_graph_gar_set;
    // sub_graph_gar_set.reserve(sub_graph_gars_path.size());
    // for (const auto& sub_graph_gar_path : sub_graph_gars_path){
    //   const std::string kGarVFile = sub_graph_gar_path + "_v.csv",
    //                     kGarEFile = sub_graph_gar_path + "_e.csv",
    //                     kGarXFile = sub_graph_gar_path + "_x.csv",
    //                     kGarYFile = sub_graph_gar_path + "_y.csv";
    //   GarType gar;
    //   auto res = gar::ReadGAR(gar, kGarVFile, kGarEFile,
    //                                kGarXFile, kGarYFile);
    //   if (res < 0){
    //     std::cout<<"loading gar: "
    //              << sub_graph_gar_path
    //              << " failed! :"<<res<<std::endl;
    //     return;
    //   }
    //   sub_graph_gar_set.emplace_back(std::move(gar), sub_graph_gar_path);
    // }

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
   * @brief Incremental evaluation for DiscoveredGfdAnalysis.
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

#endif  // EXAMPLES_ANALYTICAL_APPS_DISCOVERED_GFD_ANALYSIS_DISCOVERED_GFD_ANALYSIS_H_