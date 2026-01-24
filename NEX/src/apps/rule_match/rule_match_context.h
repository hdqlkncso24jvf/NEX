#ifndef EXAMPLES_ANALYTICAL_APPS_RULE_MATCH_RULE_MATCH_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_RULE_MATCH_RULE_MATCH_CONTEXT_H_

#include <grape/app/context_base.h>
#include <grape/grape.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "util/file.h"

#include "gar/csv_gar.h"
#include "gar/gar.h"

#include "gundam/graph_type/graph.h"
#include "gundam/graph_type/small_graph.h"

namespace grape {

/**
 * @brief Context for the parallel version of RuleMatch.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class RuleMatchContext : public VertexDataContext<FRAG_T, int64_t> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  explicit RuleMatchContext(const FRAG_T& fragment)
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

  using VertexIDType = uint32_t;
  using   EdgeIDType = uint32_t;

  using VertexLabelType = uint32_t;
  using   EdgeLabelType = uint32_t;

  // using Pattern   = GUNDAM::LargeGraph2<VertexIDType, VertexLabelType, std::string,
  //                                         EdgeIDType,   EdgeLabelType, std::string>;
  
  using   Pattern = GUNDAM::SmallGraph<VertexIDType, VertexLabelType, 
                                         EdgeIDType,   EdgeLabelType>;

  // using DataGraph = GUNDAM::LargeGraph<VertexIDType, VertexLabelType, std::string,
  //                                        EdgeIDType,   EdgeLabelType, std::string>;

  // using Pattern = GUNDAM::SmallGraph<VertexIDType, VertexLabelType, 
  //                                      EdgeIDType,   EdgeLabelType>;

  using DataGraph = GUNDAM::Graph<
      GUNDAM::SetVertexIDType<VertexIDType>,
      GUNDAM::SetVertexAttributeStoreType<GUNDAM::AttributeType::kGrouped>,
      // GUNDAM::SetVertexAttributeStoreType<GUNDAM::AttributeType::kSeparated>,
      GUNDAM::SetVertexLabelType<VertexLabelType>,
      GUNDAM::SetVertexLabelContainerType<GUNDAM::ContainerType::Map>,
      GUNDAM::SetVertexIDContainerType<GUNDAM::ContainerType::Map>,
      GUNDAM::SetVertexPtrContainerType<GUNDAM::ContainerType::Vector>,
      GUNDAM::SetEdgeLabelContainerType<GUNDAM::ContainerType::Vector>,
      GUNDAM::SetVertexAttributeKeyType<std::string>,
      GUNDAM::SetEdgeIDType<EdgeIDType>,
      GUNDAM::SetEdgeAttributeStoreType<GUNDAM::AttributeType::kGrouped>,
      // GUNDAM::SetEdgeAttributeStoreType<GUNDAM::AttributeType::kSeparated>,
      GUNDAM::SetEdgeLabelType<EdgeLabelType>,
      GUNDAM::SetEdgeAttributeKeyType<std::string>>;

  using GarType = gar::GraphAssociationRule<Pattern, DataGraph>;

  inline void AddRule(const GarType& rule,
                  const std::string& rule_name) {
    this->rule_set_for_current_worker_.emplace_back(rule, rule_name);
    return;
  }

  inline GarType& GetRule(size_t gar_idx) {
    assert(gar_idx >= 0 
        && gar_idx < this->rule_set_for_current_worker_.size());
    return this->rule_set_for_current_worker_[gar_idx].first;
  }

  inline const GarType& GetRule(size_t gar_idx) const {
    assert(gar_idx >= 0 
        && gar_idx < this->rule_set_for_current_worker_.size());
    return this->rule_set_for_current_worker_[gar_idx].first;
  }

  inline const std::string& GetRuleName(size_t gar_idx) const {
    assert(gar_idx >= 0 
        && gar_idx < this->rule_set_for_current_worker_.size());
    return this->rule_set_for_current_worker_[gar_idx].second;
  }

  inline size_t RuleSetSize() const {
    return this->rule_set_for_current_worker_.size();
  }

  std::string yaml_file_;
  int fid_;
  int frag_num_;

  std::string rule_type_;

  DataGraph data_graph_;

  std::ofstream log_file_;

 private:
  std::vector<std::pair<GarType, std::string>> rule_set_for_current_worker_;
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_RULE_MATCH_RULE_MATCH_CONTEXT_H_
