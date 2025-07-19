#ifndef EXAMPLES_ANALYTICAL_APPS_THREE_TUPLE_TO_CSV_THREE_TUPLE_TO_CSV_H_
#define EXAMPLES_ANALYTICAL_APPS_THREE_TUPLE_TO_CSV_THREE_TUPLE_TO_CSV_H_

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

#include "three_tuple_to_csv/three_tuple_to_csv_context.h"
#include "util/file.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#if defined _MSC_VER
#include <direct.h>
#elif defined __GNUC__
#include <sys/stat.h>
#include <sys/types.h>
#endif

#if defined(_WIN32)
#include <direct.h>
#include <io.h>
#else
#include <dirent.h>
#include <unistd.h>
#endif

namespace grape {

template <typename FRAG_T>
class ThreeTupleToCsv : public ParallelAppBase<FRAG_T, ThreeTupleToCsvContext<FRAG_T>>,
                        public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
                  
  using    VertexIDType = typename ThreeTupleToCsvContext<FRAG_T>::   VertexIDType;
  using VertexLabelType = typename ThreeTupleToCsvContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType = typename ThreeTupleToCsvContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType = typename ThreeTupleToCsvContext<FRAG_T>::  EdgeLabelType;
  using    DataGraph = typename ThreeTupleToCsvContext<FRAG_T>::DataGraph;

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(ThreeTupleToCsv<FRAG_T>, 
                          ThreeTupleToCsvContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for ThreeTupleToCsv.
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

    if (!config["InputVFile"]) {
      std::cout << "does not specify input V file" << std::endl;
      return;
    }
    const std::string kInputVFile = config["InputVFile"].as<std::string>();

    if (!config["InputEFile"]) {
      std::cout << "does not specify input E file" << std::endl;
      return;
    }
    const std::string kInputEFile = config["InputEFile"].as<std::string>();

    if (!config["OutputGraphDir"]){
      std::cout << "does not specified output data graph dir"
                << std::endl;
      return;
    }
    const std::string kOutputGraphDir 
             = config["OutputGraphDir"].as<std::string>();

    if (!config["OutputGraphName"]){
      std::cout << "does not specified output data graph name"
                << std::endl;
      return;
    }
    const std::string kOutputGraphName 
             = config["OutputGraphName"].as<std::string>();

    const std::string kOutputGraphPrefix
                    = kOutputGraphDir 
                    + "/"
                    + kOutputGraphName;
                  
    #if defined(_WIN32)
    _mkdir(kOutputGraphDir.c_str());
    #else
    mkdir(kOutputGraphDir.c_str(),
          0777);  // notice that 777 is different than 0777
    #endif

    DataGraph data_graph;

    std::ifstream v_file(kInputVFile);
    std::ifstream e_file(kInputEFile);

    int counter = 0;

    // to ensure the generated graph has continued vertex id
    std::map<VertexIDType, VertexIDType> vertex_id_dict;
    
    while (v_file) {
      std::string s;
      if (!std::getline( v_file, s )) 
        break;

      if (counter % 100000 == 0){
        std::cout << "vertex: " << counter << std::endl;
      }
      counter++;

      std::istringstream ss( s );
      std::string buf;
      std::vector <std::string> record;
      record.reserve(3);
      while (ss >> buf)
        record.emplace_back(buf);

      for (int i = 3; i < record.size(); i++){
        record[2] = record[2] + " " + record[i];
      }
      
      record[2].erase(
          std::remove( record[2].begin(), 
                       record[2].end(), '\"' ),
                       record[2].end() );

      VertexIDType    kVertexId    = GUNDAM::StringToDataType<VertexIDType   >(record[0]);
      VertexLabelType kVertexLabel = GUNDAM::StringToDataType<VertexLabelType>(record[1]);
      std::string kVertexAttribute = record[2];

      auto [ vertex_id_it,
             vertex_id_ret ] 
           = vertex_id_dict.emplace(kVertexId, 
                                     vertex_id_dict.size());
      if (!vertex_id_ret) {
        std::cout << "duplicated vertex id: " << kVertexId << std::endl;
        return;
      }

      auto [vertex_handle,
            vertex_ret] = data_graph.AddVertex(vertex_id_it->second, 
                                               kVertexLabel);
      assert(vertex_handle);
      assert(vertex_ret);
      if (!vertex_ret){
        // add fail
        std::cout << "add vertex failed: id: " << kVertexId 
                                 << " label: " << kVertexLabel << std::endl;
        vertex_id_dict.erase(vertex_id_it);
        continue;
      }
      auto [attribute_handle,
            attribute_ret] = vertex_handle->AddAttribute("attr", kVertexAttribute);
      assert(attribute_ret);
    }

    EdgeIDType edge_id_allocator = 0;
    
    counter = 0;
    while (e_file) {
      std::string s;
      if (!std::getline( e_file, s )) 
        break;

      if (counter % 100000 == 0){
        std::cout << "edge: " << counter << std::endl;
      }
      counter++;

      std::istringstream ss( s );
      std::string buf;
      std::vector <std::string> record;
      record.reserve(3);
      while (ss >> buf)
        record.emplace_back(buf);
      
      VertexIDType  kSrcVertexId = GUNDAM::StringToDataType< VertexIDType>(record[0]);
      EdgeLabelType kEdgeLabel   = GUNDAM::StringToDataType<EdgeLabelType>(record[1]);
      VertexIDType  kDstVertexId = GUNDAM::StringToDataType< VertexIDType>(record[2]);

      auto src_vertex_it = vertex_id_dict.find(kSrcVertexId);
      if (src_vertex_it == vertex_id_dict.end()) {
        // the src vertex cannot be found in added vertexes
        std::cout << "src vertex cannot be found in added vertexes: " << kSrcVertexId << std::endl;
        continue;
      }
      auto dst_vertex_it = vertex_id_dict.find(kDstVertexId);
      if (dst_vertex_it == vertex_id_dict.end()) {
        // the dst vertex cannot be found in added vertexes
        std::cout << "dst vertex cannot be found in added vertexes: " << kDstVertexId << std::endl;
        continue;
      }


      auto [edge_handle,
            edge_ret] = data_graph.AddEdge(src_vertex_it->second, 
                                           dst_vertex_it->second,
                                           kEdgeLabel,
                                           edge_id_allocator++);
      // should added successfully
      if (!edge_ret){
        // ensure the allocated edge id is continued
        edge_id_allocator--;
        std::cout << "src id: " << kSrcVertexId << std::endl
                  << "dst id: " << kDstVertexId << std::endl
                  << "edge label: " << kEdgeLabel << std::endl;
      }
      assert(edge_ret);
    }

    if (!GUNDAM::VertexIDContinued(data_graph)) {
      std::cout << "vertex id is not continued" << std::endl;
    }

    if (!GUNDAM::EdgeIDContinued(data_graph)) {
      std::cout << "edge id is not continued" << std::endl;
    }

    if (GUNDAM::HasIsolateVertex(data_graph)) {
      std::cout << "has isolated vertex" << std::endl;
    }

    const std::string kDataGraphVFile = kOutputGraphPrefix + "_v.csv",
                      kDataGraphEFile = kOutputGraphPrefix + "_e.csv";

    GUNDAM::WriteCSVGraph(data_graph, kDataGraphVFile,
                                      kDataGraphEFile);

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
   * @brief Incremental evaluation for ThreeTupleToCsv.
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

#endif  // EXAMPLES_ANALYTICAL_APPS_THREE_TUPLE_TO_CSV_THREE_TUPLE_TO_CSV_H_