#include "include/gundam/graph_type/large_graph.h"
#include "include/gundam/io/csvgraph.h"
#include "include/util/log.h"

#include "src/apps/rdf_to_graph/flags.h"

#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>

#include <iostream>
#include <fstream>
#include <yaml-cpp/yaml.h>

int main(int argc, char* argv[]) {

  gflags::ParseCommandLineFlags(&argc, &argv, true);
  gflags::ShutDownCommandLineFlags();

  using VertexIDType = uint32_t;
  using   EdgeIDType = uint32_t;

  using VertexLabelType = uint32_t;
  using   EdgeLabelType = uint32_t;

  using DataGraph = GUNDAM::LargeGraph<VertexIDType, VertexLabelType, std::string,
                                         EdgeIDType,   EdgeLabelType, std::string>;

  const std::string yaml_file_str = rdf_to_graph::FLAGS_yaml_file;

  util::Info("yaml file: " + yaml_file_str);

  YAML::Node config = YAML::LoadFile(yaml_file_str);

  if (!config["RdfFile"]) {
    util::Error("does not specify the csv file");
    return -1;
  }
  const std::string kRdfFile = config["RdfFile"].as<std::string>();

  if (!config["OutputGraph"]) {
    util::Error("does not specify output graph");
    return -1;
  }
  YAML::Node output_graph_config = config["OutputGraph"];
    
  if (!output_graph_config["GraphDir"]) {
    util::Error("does not specify output graph dir");
    return -1;
  }
  const std::string kGraphDir = output_graph_config["GraphDir"].as<std::string>();
   
  if (!output_graph_config["GraphName"]) {
    util::Error("does not specify output graph name");
    return -1;
  }
  const std::string kGraphName = output_graph_config["GraphName"].as<std::string>();

  std::vector<std::string> as_vertex_label;
  if (config["AsVertexLabel"]) {
    YAML::Node as_vertex_label_config = config["AsVertexLabel"];
    for (int i = 0; i < as_vertex_label_config.size(); i++) {
      if (as_vertex_label_config[i].as<std::string>() == "") {
        util::Error("illegal edge label");
        return -1;
      }
      as_vertex_label.emplace_back(as_vertex_label_config[i].as<std::string>());
    }
    std::sort(as_vertex_label.begin(),
              as_vertex_label.end());
  }

  std::map<std::string, 
           VertexIDType> vertex_id_dict;

  const VertexLabelType kVertexLabel = 0;

  std::map<std::string, 
           VertexLabelType> vertex_label_dict = {std::pair("DefaultVertex", kVertexLabel)};

  std::map<std::string, 
           EdgeLabelType> edge_label_dict;

  DataGraph data_graph;

  if (!as_vertex_label.empty()) {
    std::cout << "edge as vertex label: " << std::endl;
    for (const auto& label : as_vertex_label) {
      std::cout << "\t" << label << std::endl;
    }
    std::ifstream rdf_file(kRdfFile);
    while (rdf_file) {
      std::string s;
      if (!std::getline( rdf_file, s )) 
        break;

      std::istringstream ss( s );
      std::string buf;
      std::vector <std::string> record;
      record.reserve(3);
      while (ss >> buf)
        record.emplace_back(buf);

      if (record.size() < 3) {
        util::Error("illegal output record column size: "
                  + std::to_string(record.size()));
        util::Error(s);
        return -1;
      }

      if (!std::binary_search(as_vertex_label.begin(),
                              as_vertex_label.end(),
                              record[1])) {
        // should be regarded as vertex label
        continue;
      }

      for (int i = 3; i < record.size(); i++) {
        record[2].append(" " + record[i]);
      }

      auto [ vertex_label_dict_it,
             vertex_label_dict_ret ]
           = vertex_label_dict.emplace(record[2], vertex_label_dict.size());

      auto src_vertex_it = vertex_id_dict.find(record[0]);
      if (src_vertex_it == vertex_id_dict.end()) {
        src_vertex_it = vertex_id_dict.emplace_hint(src_vertex_it,
                                                    record[0],
                                                    vertex_id_dict.size());
        auto [vertex_handle,
              vertex_ret] = data_graph.AddVertex(src_vertex_it->second, 
                                          vertex_label_dict_it->second);
        assert(vertex_ret);
        auto [attribute_handle,
              attribute_ret] = vertex_handle->AddAttribute("attr", record[0]);
      }
    }
  }

  size_t edge_id_allocator = 0;

  std::ifstream rdf_file(kRdfFile);
  while (rdf_file) {
    std::string s;
    if (!std::getline( rdf_file, s )) 
      break;

    std::istringstream ss( s );
    std::string buf;
    std::vector <std::string> record;
    record.reserve(3);
    while (ss >> buf)
      record.emplace_back(buf);

    if (record.size() < 3) {
      util::Error("illegal output record column size: "
                 + std::to_string(record.size()));
      util::Error(s);
      return -1;
    }

    if (std::binary_search(as_vertex_label.begin(),
                           as_vertex_label.end(),
                           record[1])) {
      // vertex label, should not be considered as edge
      continue;
    }

    for (int i = 3; i < record.size(); i++) {
      record[2].append(" " + record[i]);
    }

    auto src_vertex_it = vertex_id_dict.find(record[0]);
    if (src_vertex_it == vertex_id_dict.end()) {
      src_vertex_it = vertex_id_dict.emplace_hint(src_vertex_it,
                                                  record[0],
                                                  vertex_id_dict.size());
      auto [vertex_handle,
            vertex_ret] = data_graph.AddVertex(src_vertex_it->second, 
                                                  kVertexLabel);
      assert(vertex_ret);
      auto [attribute_handle,
            attribute_ret] = vertex_handle->AddAttribute("attr", record[0]);
    }

    auto dst_vertex_it = vertex_id_dict.find(record[2]);
    if (dst_vertex_it == vertex_id_dict.end()) {
      dst_vertex_it = vertex_id_dict.emplace_hint(dst_vertex_it,
                                                  record[2],
                                                  vertex_id_dict.size());
      auto [vertex_handle,
            vertex_ret] = data_graph.AddVertex(dst_vertex_it->second, 
                                                  kVertexLabel);
      assert(vertex_ret);
      auto [attribute_handle,
            attribute_ret] = vertex_handle->AddAttribute("attr", record[2]);
    }
    
    auto edge_label_it = edge_label_dict.find(record[1]);
    if (edge_label_it == edge_label_dict.end()) {
      edge_label_it = edge_label_dict.emplace_hint(edge_label_it,
                                                   record[1],
                                                   edge_label_dict.size() + vertex_label_dict.size());
    }

    auto [edge_handle,
          edge_ret] = data_graph.AddEdge(src_vertex_it->second, 
                                         dst_vertex_it->second,
                                         edge_label_it->second,
                                         edge_id_allocator++);

    assert(edge_ret);
  }

  GUNDAM::WriteCSVGraph(data_graph, kGraphDir + "/" + kGraphName + "_v.csv",
                                    kGraphDir + "/" + kGraphName + "_e.csv");

  std::ofstream dict_file(kGraphDir + "/" + kGraphName + "_dict.csv");
  for (const auto& [vertex_string, vertex_id] : vertex_label_dict) {
    dict_file << vertex_id << "," << vertex_string << std::endl;
  }
  for (const auto& [edge_string, edge_id] : edge_label_dict) {
    dict_file << edge_id << "," << edge_string << std::endl;
  }
  return 0;
}