#ifndef _EXAMPLES_ANALYTICAL_APPS_GRAPH_SAMPLE_GRAPH_SAMPLE_H_
#define _EXAMPLES_ANALYTICAL_APPS_GRAPH_SAMPLE_GRAPH_SAMPLE_H_

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

#include "graph_sample/graph_sample_context.h"

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
class GraphSample : public ParallelAppBase<FRAG_T, GraphSampleContext<FRAG_T>>,
                    public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
  using    VertexIDType = typename GraphSampleContext<FRAG_T>::   VertexIDType;
  using VertexLabelType = typename GraphSampleContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType = typename GraphSampleContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType = typename GraphSampleContext<FRAG_T>::  EdgeLabelType;
  using    DataGraph    = typename GraphSampleContext<FRAG_T>::DataGraph;

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(GraphSample<FRAG_T>, 
                          GraphSampleContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for GraphSample.
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

    DataGraph data_graph;
    GUNDAM::ReadCSVGraph(data_graph, 
                        kDataGraphPathVFile,
                        kDataGraphPathEFile);

    if (!config["SampleMethod"]) {
      std::cout << "does not specify support sample method" << std::endl;
      return;
    }
    const std::string kSampleMethod
             = config["SampleMethod"].as<std::string>();

    if (kSampleMethod != "vertex_sample"
     && kSampleMethod !=   "edge_sample"){
      std::cout<<"unknown sample method: "<<kSampleMethod<<std::endl;
      return;
    }

    if (!config["Ratios"]) {
      std::cout << "cannot get sample ratios!" << std::endl;
      return;
    }
    YAML::Node ratios_config = config["Ratios"];
    std::vector<double> ratio_list;
    for(int i=0; i < ratios_config.size(); ++i){
      ratio_list.emplace_back(ratios_config[i].as<double>());
    }

    if (kSampleMethod == "vertex_sample"){
      for (const auto& ratio : ratio_list){
        const DataGraph kSampledGraph = GUNDAM::VertexSample(data_graph, ratio);
        const std::string 
          kSampleGraphVFile = kDataGraphPrefix 
                            + "_vertex_sample_"
                            + std::to_string(ratio) 
                            + "_v.csv",
          kSampleGraphEFile = kDataGraphPrefix 
                            + "_vertex_sample_" 
                            + std::to_string(ratio) 
                            + "_e.csv";
        GUNDAM::WriteCSVGraph(kSampledGraph, 
                             kSampleGraphVFile,
                             kSampleGraphEFile);
      }
      return;
    }
    assert(kSampleMethod == "edge_sample");
    for (const auto& ratio : ratio_list){
      const DataGraph kSampledGraph = GUNDAM::EdgeSample(data_graph, ratio);
      const std::string 
        kSampleGraphVFile = kDataGraphPrefix 
                          + "_edge_sample_"
                          + std::to_string(ratio) 
                          + "_v.csv",
        kSampleGraphEFile = kDataGraphPrefix 
                          + "_edge_sample_" 
                          + std::to_string(ratio) 
                          + "_e.csv";
      GUNDAM::WriteCSVGraph(kSampledGraph, 
                            kSampleGraphVFile,
                            kSampleGraphEFile);
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
   * @brief Incremental evaluation for GraphSample.
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

#endif  // EXAMPLES_ANALYTICAL_APPS_GRAPH_SAMPLE_GRAPH_SAMPLE_H_