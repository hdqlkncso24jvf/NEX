#ifndef EXAMPLES_ANALYTICAL_APPS_SELECT_EDGE_SELECT_EDGE_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_SELECT_EDGE_SELECT_EDGE_CONTEXT_H_

#include <grape/app/context_base.h>
#include <grape/grape.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "util/file.h"

namespace grape {

/**
 * @brief Context for the parallel version of SelectEdge.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class SelectEdgeContext : public VertexDataContext<FRAG_T, int64_t> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  explicit SelectEdgeContext(const FRAG_T& fragment)
    : VertexDataContext<FRAG_T, int64_t>(fragment, true){
    return;
  }

  void Init(ParallelMessageManager& messages, std::string yaml_file,
            int frag_num) {
    this->yaml_file_ = yaml_file;
    this->fid_ = this->fragment().fid();
    this->frag_num_ = frag_num;
#ifdef PROFILING
    preprocess_time = 0;
    exec_time = 0;
    postprocess_time = 0;
#endif
  }

  void Output(std::ostream& os) {
#ifdef PROFILING
    VLOG(2) << "preprocess_time: " << preprocess_time << "s.";
    VLOG(2) << "exec_time: " << exec_time << "s.";
    VLOG(2) << "postprocess_time: " << postprocess_time << "s.";
#endif
  }

#ifdef PROFILING
  double preprocess_time = 0;
  double exec_time = 0;
  double postprocess_time = 0;
#endif

  using VertexIDType = uint32_t;
  using   EdgeIDType = uint32_t;

  using VertexLabelType = uint32_t;
  using   EdgeLabelType = uint32_t;

  using VertexAttributeKeyType = std::string;
  
  using   EdgeAttributeKeyType = std::string;

  static constexpr std::string_view kRulePosfix = "_v.csv";

  // using DataGraph = GUNDAM::Graph<
  //     GUNDAM::SetVertexIDType<VertexIDType>,
  //     GUNDAM::SetVertexAttributeStoreType<GUNDAM::AttributeType::kGrouped>,
  //     // GUNDAM::SetVertexAttributeStoreType<GUNDAM::AttributeType::kSeparated>,
  //     GUNDAM::SetVertexLabelType<VertexLabelType>,
  //     GUNDAM::SetVertexLabelContainerType<GUNDAM::ContainerType::Vector>,
  //     GUNDAM::SetVertexIDContainerType<GUNDAM::ContainerType::Map>,
  //     GUNDAM::SetVertexPtrContainerType<GUNDAM::ContainerType::Map>,
  //     GUNDAM::SetEdgeLabelContainerType<GUNDAM::ContainerType::Map>,
  //     GUNDAM::SetVertexAttributeKeyType<VertexAttributeKeyType>,
  //     GUNDAM::SetEdgeIDType<EdgeIDType>,
  //     GUNDAM::SetEdgeAttributeStoreType<GUNDAM::AttributeType::kGrouped>,
  //     // GUNDAM::SetEdgeAttributeStoreType<GUNDAM::AttributeType::kSeparated>,
  //     GUNDAM::SetEdgeLabelType<EdgeLabelType>,
  //     GUNDAM::SetEdgeAttributeKeyType<EdgeAttributeKeyType>>;

  using DataGraph = GUNDAM::LargeGraph<VertexIDType, VertexLabelType, std::string,
                                         EdgeIDType,   EdgeLabelType, std::string>;

  using GraphPattern =
      GUNDAM::LargeGraph2<VertexIDType, VertexLabelType, VertexAttributeKeyType,
                            EdgeIDType,   EdgeLabelType,   EdgeAttributeKeyType>;

  std::string yaml_file_;
  int fid_;
  int frag_num_;
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_SELECT_EDGE_SELECT_EDGE_CONTEXT_H_
