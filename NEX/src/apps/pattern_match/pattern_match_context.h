#ifndef EXAMPLES_ANALYTICAL_APPS_PATTERN_MATCH_PATTERN_MATCH_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_PATTERN_MATCH_PATTERN_MATCH_CONTEXT_H_

#include <grape/app/context_base.h>
#include <grape/grape.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <queue>

#include "gundam/graph_type/small_graph.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/graph.h"

#include "gundam/match/match.h"

namespace grape {

/**
 * @brief Context for the parallel version of PatternMatch.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class PatternMatchContext : public VertexDataContext<FRAG_T, int64_t> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  explicit PatternMatchContext(const FRAG_T& fragment)
           : VertexDataContext<FRAG_T, int64_t>(fragment, true){
    return;
  }

  void Init(ParallelMessageManager& messages, std::string yaml_file,
            int frag_num) {
    this->yaml_file_ = yaml_file;
    this->fid_ = this->fragment().fid();
    this->frag_num_ = frag_num;
    // #######################
    // #  program parameter  #
    // #######################
    this->ResetParameter();
    this->current_processing_pattern_idx_ = 0;
    this->total_match_time_ = 0;
    this->   supp_limit_ = -1;
    this->   time_limit_ = -1.0;
    this->   time_limit_per_supp_ = -1.0;

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

  using MatchSetType = GUNDAM::MatchSet<Pattern, DataGraph,
                       GUNDAM::ContainerType::Set,
                       GUNDAM::SortType::Default>;

  using CandidateSet = std::map<typename GUNDAM::VertexHandle< Pattern >::type, 
                    std::vector<typename GUNDAM::VertexHandle<DataGraph>::type>>;

  inline void ToNextPattern() {
    assert(this->current_processing_pattern_idx_ 
         < this->pattern_set_.size());
    this->current_processing_pattern_idx_++;
    return;
  }

  inline bool HasPatternToProcess() const {
    return this->current_processing_pattern_idx_ 
         < this->pattern_set_.size();
  }

  inline bool HasProcessedAllPattern() const {
    return !this->HasPatternToProcess();
  }

  inline Pattern& CurrentPattern() {
    assert(this->current_processing_pattern_idx_ >= 0 
        && this->current_processing_pattern_idx_ < this->pattern_set_.size());
    return this->pattern_set_[this->current_processing_pattern_idx_].first;
  }

  inline bool CurrentPatternCollectMatchResult() const {
    return this->CurrentMatchFile() != "";
  }

  inline const std::string& CurrentMatchFile() const {
    assert(this->current_processing_pattern_idx_ >= 0 
        && this->current_processing_pattern_idx_ < this->pattern_path_set_.size());
    return this->pattern_path_set_[this->current_processing_pattern_idx_].second;
  }

  inline std::string CurrentPatternName() const {
    return "# pattern # vfile: " + std::get<0>(this->pattern_path_set_[this->current_processing_pattern_idx_].first)
                  + " # efile: " + std::get<1>(this->pattern_path_set_[this->current_processing_pattern_idx_].first)
                  + " # pivot: " + GUNDAM::ToString(this->CurrentPivot()->id());
                  + " #";
  }

  inline const typename GUNDAM::VertexHandle<Pattern>::type& CurrentPivot() const {
    assert(this->current_processing_pattern_idx_ >= 0 
        && this->current_processing_pattern_idx_ < this->pattern_set_.size());
    assert(this->pattern_set_[this->current_processing_pattern_idx_].second);
    return this->pattern_set_[this->current_processing_pattern_idx_].second;
  }

  inline const size_t PatternSetSize() const {
    return this->pattern_set_.size();
  }

  inline void ResetParameter() {
    this->match_set_.Clear();
    this->candidate_set_initialized_ = false;
    this->remained_pivot_size_.clear();
    this->remained_pivot_size_.resize(this->frag_num_, 0);
    this->current_allocated_pivot_set_idx_ = 0;
    this->final_result_ = 0;
    this->candidate_set_.clear();
    if (this->work_load_balance_) {
      // in this case, the data_graph_pivot_range_ is allocated
      // from the coordinator, needs to be reset in each round for
      // different patterns
      assert(this->data_graph_pivot_range_.empty());
      this->data_graph_pivot_range_.clear();
    }
    return;
  }

  inline void AddMatchTime(double time) {
    this->total_match_time_ += time;
    return;
  }

  inline double TotalMatchTime() const {
    return this->total_match_time_;
  }

  std::string yaml_file_;
  int fid_;
  int frag_num_;

  DataGraph data_graph_;

  std::vector<typename GUNDAM::VertexHandle<DataGraph>::type> 
    data_graph_pivot_range_;

  CandidateSet candidate_set_;

  std::vector<std::pair<
              std::tuple<std::string,
                         std::string,
                         std::string>, std::string>> pattern_path_set_;

  std::vector<std::pair<Pattern, typename GUNDAM::VertexHandle<Pattern>::type>>  pattern_set_;

  bool is_incremental_,
       is_isomorphism_;
       
  bool candidate_set_initialized_;

  size_t final_result_;

  std::ofstream time_log_file_;

  // parameter for workload balance
  bool work_load_balance_;
  int32_t balance_period_ms_;
  int32_t pivoted_candidate_balance_size_;

  // to be stored in each worker, for more efficient transform
  std::vector<typename GUNDAM::VertexHandle<DataGraph>::type> 
    sorted_vertex_handle_set_;

  std::vector<size_t> remained_pivot_size_;

  size_t current_allocated_pivot_set_idx_;

  int64_t supp_limit_;

  double time_limit_,
         time_limit_per_supp_;

  MatchSetType match_set_;

 private:
  size_t current_processing_pattern_idx_;

  double total_match_time_;
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_PATTERN_MATCH_PATTERN_MATCH_CONTEXT_H_
