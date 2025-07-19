#ifndef EXAMPLES_ANALYTICAL_APPS_GRAPH_DISCRETIZE_GRAPH_DISCRETIZE_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_GRAPH_DISCRETIZE_GRAPH_DISCRETIZE_CONTEXT_H_

#include <string>

#include "gundam/io/csvgraph.h"
#include "gundam/tool/operator/merge_graph.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/large_graph2.h"
#include "gundam/graph_type/small_graph.h"
#include "gundam/graph_type/graph.h"
#include "gundam/tool/random_walk/random_walk_path.h"
#include <yaml-cpp/yaml.h>

namespace grape{
namespace graph_discretize {

void GraphDiscretize(const std::string& yaml_file) {

//  using DataGraph = GUNDAM::LargeGraph <int32_t, int32_t, std::string,
//                                          int32_t,   int32_t, std::string>;
  using VertexIDType = uint64_t;
  using VertexLabelType = uint32_t;
  using VertexAttributeKeyType = std::string;

  using EdgeIDType = uint64_t;
  using EdgeLabelType = uint32_t;
  using EdgeAttributeKeyType = std::string;

  using DataGraph = GUNDAM::Graph<
      GUNDAM::SetVertexIDType<VertexIDType>,
      GUNDAM::SetVertexAttributeStoreType<GUNDAM::AttributeType::kGrouped>,
      // GUNDAM::SetVertexAttributeStoreType<GUNDAM::AttributeType::kSeparated>,
      GUNDAM::SetVertexLabelType<VertexLabelType>,
      GUNDAM::SetVertexLabelContainerType<GUNDAM::ContainerType::Map>,
      GUNDAM::SetVertexIDContainerType<GUNDAM::ContainerType::Map>,
      GUNDAM::SetVertexPtrContainerType<GUNDAM::ContainerType::Map>,
      GUNDAM::SetEdgeLabelContainerType<GUNDAM::ContainerType::Vector>,
      GUNDAM::SetVertexAttributeKeyType<VertexAttributeKeyType>,
      GUNDAM::SetEdgeIDType<EdgeIDType>,
      GUNDAM::SetEdgeAttributeStoreType<GUNDAM::AttributeType::kGrouped>,
      // GUNDAM::SetEdgeAttributeStoreType<GUNDAM::AttributeType::kSeparated>,
      GUNDAM::SetEdgeLabelType<EdgeLabelType>,
      GUNDAM::SetEdgeAttributeKeyType<EdgeAttributeKeyType>>;


  using VertexHandleType = typename GUNDAM::VertexHandle<DataGraph>::type;

  std::cout<<"yaml file: "<< yaml_file <<std::endl;

  YAML::Node config = YAML::LoadFile(yaml_file);

  if (!config["VFile"]){
    std::cout<<"the v file has not been been configured"<<std::endl;
    return;
  }
  const std::string kGraphVFile
    = config["VFile"].as<std::string>();

  if (!config["EFile"]){
    std::cout << "the e file has not been been configured" << std::endl;
    return;
  }
  const std::string kGraphEFile
    = config["EFile"].as<std::string>();

  int kPathNumber;

  if (!config["PathNumber"]) {
    util::Info("cant get the number of random paths");
  } else {
      kPathNumber = config["PathNumber"].as<int>();
  }

  DataGraph continuous_graph;

  if (GUNDAM::ReadCSVGraph(continuous_graph, kGraphVFile,
                          kGraphEFile) < 0) {
    std::cout << "load input graph fail: " << std::endl
              << "\t v file: " << kGraphVFile << std::endl
              << "\t e file: " << kGraphEFile << std::endl;
    return;
  }

  std::vector<DataGraph> star_vec;
  unsigned count = 0;
  for (auto vertex_it = continuous_graph.VertexBegin();
            !vertex_it.IsDone();
            vertex_it++) {
    count++;
    if (count % 100 == 0) {
      std::cout << "processed node " << count << std::endl;
    }
    VertexHandleType vertex_handle = vertex_it;

    std::vector<DataGraph> path_vec;
    std::vector<VertexIDType> id_vec;

    id_vec.push_back(vertex_handle->id());

    for (unsigned i = 0; i < kPathNumber; i++) {
      auto path = GUNDAM::RandomWalkPath(continuous_graph, vertex_handle, 3);
      path_vec.push_back(path);
    }

    DataGraph star = GUNDAM::MergeGraphSet(path_vec, id_vec);
    star_vec.push_back(star);
  }
      std::cout << "processed all nodes " << count << std::endl;

  std::vector<VertexIDType> empty_id_vec;

  DataGraph discrete_graph = GUNDAM::MergeGraphSet(star_vec, empty_id_vec);

  GUNDAM::WriteCSVGraph<true>(discrete_graph, kGraphVFile + ".dis" + std::to_string(kPathNumber),
                                        kGraphEFile + ".dis" + std::to_string(kPathNumber));

  return;
};
    
}; // graph_discretize
}; // grape
#endif // EXAMPLES_ANALYTICAL_APPS_GRAPH_DISCRETIZE_GRAPH_DISCRETIZE_CONTEXT_H_
