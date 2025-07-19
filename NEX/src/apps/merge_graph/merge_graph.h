#ifndef _EXAMPLES_ANALYTICAL_APPS_MERGE_GRAPH_MERGE_GRAPH_H_
#define _EXAMPLES_ANALYTICAL_APPS_MERGE_GRAPH_MERGE_GRAPH_H_

#include <grape/grape.h>
#include <omp.h>

#include <functional>

#include "gundam/io/csvgraph.h"

#include "gundam/tool/operator/merge_by_id.h"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "merge_graph/merge_graph_context.h"

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
class MergeGraph : public ParallelAppBase<FRAG_T, MergeGraphContext<FRAG_T>>,
                   public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
  using DataGraph = typename MergeGraphContext<FRAG_T>::DataGraph;

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(MergeGraph<FRAG_T>, 
                          MergeGraphContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for MergeGraph.
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

    if (!config["MergeConfigure"]) {
      std::cout << "does not specify the merge configure" << std::endl;
      return;
    }
    YAML::Node merge_configures = config["MergeConfigure"];

    for (int i = 0; i < merge_configures.size(); i++) {
      YAML::Node merge_configure = merge_configures[i];

      if (!merge_configure["MergedGraphPath"]) {
        std::cout << "cannot get the path of the merged graph" << std::endl;
        return;
      }

      YAML::Node merged_graph_path_config = merge_configure["MergedGraphPath"];

      if (!merged_graph_path_config["VFile"]){
        std::cout<<"the v file of the merged graph has not been been configured"<<std::endl;
        return;
      }
      const std::string kMergedGraphVFile
        = merged_graph_path_config["VFile"].as<std::string>();

      if (!merged_graph_path_config["EFile"]){
        std::cout<<"the e file of the merged graph has not been been configured"<<std::endl;
        return;
      }
      const std::string kMergedGraphEFile
        = merged_graph_path_config["EFile"].as<std::string>();

      if (!merge_configure["GraphPathToMerge"]) {
        std::cout << "cannot get the path of graphes to merge" << std::endl;
        return;
      }
      YAML::Node graph_pathes_to_merge_config = merge_configure["GraphPathToMerge"];
      
      DataGraph merged_graph;

      bool load_data_graph_fail = false;

      for (int i = 0; i < graph_pathes_to_merge_config.size(); i++) {
        YAML::Node graph_path_to_merge_config
              = graph_pathes_to_merge_config[i];
        if (!graph_path_to_merge_config["VFile"]) {
          std::cout << "cannot get v file for " << i << "'th graph" << std::endl;
          return;
        }
        const std::string kDataGraphVFile = graph_path_to_merge_config["VFile"].as<std::string>();
        if (!graph_path_to_merge_config["EFile"]) {
          std::cout << "cannot get e file for " << i << "'th graph" << std::endl;
          return;
        }
        const std::string kDataGraphEFile = graph_path_to_merge_config["EFile"].as<std::string>();

        DataGraph graph_to_merge;
        if (GUNDAM::ReadCSVGraph(graph_to_merge, kDataGraphVFile,
                                                 kDataGraphEFile) < 0){
          std::cout << "load data graph fail: " << std::endl
                    << "\t v file: " << kDataGraphVFile << std::endl
                    << "\t e file: " << kDataGraphEFile << std::endl;
          load_data_graph_fail = true;
          break;
        }
        
        GUNDAM::MergeById(merged_graph, graph_to_merge);
      }

      if (load_data_graph_fail) {
        continue;
      }
      GUNDAM::WriteCSVGraph(merged_graph, kMergedGraphVFile,
                                          kMergedGraphEFile);
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
   * @brief Incremental evaluation for MergeGraph.
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

#endif  // EXAMPLES_ANALYTICAL_APPS_MERGE_GRAPH_MERGE_GRAPH_H_