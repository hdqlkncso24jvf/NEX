#include <gflags/gflags.h>
#include <gflags/gflags_declare.h>
#include <glog/logging.h>

#include "include/util/log.h"
#include "src/apps/graph_discretize/graph_discretize.h"

int main(int argc, char **argv) {
  if (argc < 2) {
    util::Error("parameter num is not correct! (0 or 1)");
    exit(-1);
  }

  std::string config_file_path = "graph_discretize.yaml";
  if (argc == 2) {
    config_file_path = argv[1];
  }

  YAML::Node config_node = YAML::LoadFile(config_file_path);

  grape::graph_discretize::GraphDiscretize(config_file_path);

  return 0;
}
