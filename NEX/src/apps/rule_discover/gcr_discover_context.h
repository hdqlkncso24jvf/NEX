#ifndef EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_GCR_DISCOVER_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_GCR_DISCOVER_CONTEXT_H_

#include <grape/app/context_base.h>
#include <grape/grape.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "gar/literal.h"
#include "gar/gar.h"

#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/large_graph2.h"
#include "gundam/graph_type/small_graph.h"
#include "gundam/graph_type/graph.h"

#include "gundam/graph_statistics/graph_basic_statistics.h"

#include "rule_discover/gar_discover/restriction.h"

#include "util/log.h"

namespace grape {

/**
 * @brief Context for the parallel version of GcrDiscover.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class GcrDiscoverContext : public VertexDataContext<FRAG_T, int64_t> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  explicit GcrDiscoverContext(const FRAG_T& fragment)
      : VertexDataContext<FRAG_T, int64_t>(fragment, true) {
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

  std::string yaml_file_;
  int fid_;
  int frag_num_;

  // process num for each worker
  std::map<int, int> process_num_;

  inline bool SpecifiedConfidenceBound() const {
    return this->specified_confidence_bound_;
  }

  using VertexIDType = uint64_t;
  using VertexLabelType = uint32_t;
  using VertexAttributeKeyType = std::string;

  using EdgeIDType = uint64_t;
  using EdgeLabelType = uint32_t;
  using EdgeAttributeKeyType = std::string;

  // using GraphPatternType =
  //     GUNDAM::LargeGraph2<VertexIDType, VertexLabelType, VertexAttributeKeyType,
  //                           EdgeIDType,   EdgeLabelType,   EdgeAttributeKeyType>;

  using GraphPatternType = GUNDAM::SmallGraph<VertexIDType, VertexLabelType, 
                                                EdgeIDType,   EdgeLabelType>;

  // using DataGraphType =
  //     GUNDAM::LargeGraph<VertexIDType, VertexLabelType, VertexAttributeKeyType,
  //                          EdgeIDType,   EdgeLabelType,   EdgeAttributeKeyType>;
  using DataGraphType = GUNDAM::Graph<
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

  using GraphBasicStatisticsType = GUNDAM::GraphBasicStatistics<DataGraphType>;

  using DataGraphVertexLabelType = typename GUNDAM::VertexLabel<DataGraphType>::type;
  using   DataGraphEdgeLabelType = typename GUNDAM::  EdgeLabel<DataGraphType>::type;

  using GcrType = gar::GraphAssociationRule<GraphPatternType,
                                               DataGraphType>;

  using LiteralInfoType = gar::LiteralInfo<GraphPatternType,
                                              DataGraphType>;

  using SuppType = uint64_t;
  using DepthType = uint8_t;

  using GraphPatternVertexHandleType = typename GUNDAM::VertexHandle<GraphPatternType>::type;
  using    DataGraphVertexHandleType = typename GUNDAM::VertexHandle<   DataGraphType>::type;
  
  using CandidateSetType
    = std::map<GraphPatternVertexHandleType,
      std::vector<DataGraphVertexHandleType>>;

  using RestrictionType = _gar_discover::Restriction<GraphPatternType,
                                                        DataGraphType>;

  DataGraphType data_graph_;

  std::vector<std::pair<GraphPatternType, 
                         LiteralInfoType>> predicate_set_pattern_literal_;

  std::vector<GcrType> predicate_set_;

  std::vector<std::tuple<GcrType, 
                CandidateSetType,
                        SuppType,
                        SuppType>> forest_;

  DepthType round_;

  // restrictions 
  RestrictionType restriction_;

  std::string output_gcr_dir_;

  // support bound
  uint64_t support_bound_;

  float confidence_bound_;

  bool specified_confidence_bound_;
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_GAR_DISCOVER_GAR_DISCOVER_CONTEXT_H_
