#ifndef EXAMPLES_ANALYTICAL_APPS_PROB_GAR_CHASE_PROB_GAR_CHASE_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_PROB_GAR_CHASE_PROB_GAR_CHASE_CONTEXT_H_

#include <grape/app/context_base.h>
#include <grape/grape.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <queue>

#include "prob_gar/prob_gar.h"

#include "gundam/graph_type/small_graph.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/graph.h"

namespace grape {

/**
 * @brief Context for the parallel version of ProbGarChase.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class ProbGarChaseContext : public VertexDataContext<FRAG_T, int64_t> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  explicit ProbGarChaseContext(const FRAG_T& fragment)
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
    this->current_processing_prob_gar_idx_ = 0;
    this->total_match_time_ = 0;

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

  // using Pattern = GUNDAM::LargeGraph<VertexIDType, VertexLabelType, std::string,
  //                                      EdgeIDType,   EdgeLabelType, std::string>;

  using DataGraph = GUNDAM::LargeGraph<VertexIDType, VertexLabelType, std::string,
                                         EdgeIDType,   EdgeLabelType, std::string>;

  // using DataGraph = GUNDAM::Graph<
  //     GUNDAM::SetVertexIDType<VertexIDType>,
  //     GUNDAM::SetVertexAttributeStoreType<GUNDAM::AttributeType::kGrouped>,
  //     // GUNDAM::SetVertexAttributeStoreType<GUNDAM::AttributeType::kSeparated>,
  //     GUNDAM::SetVertexLabelType<VertexLabelType>,
  //     GUNDAM::SetVertexLabelContainerType<GUNDAM::ContainerType::Map>,
  //     GUNDAM::SetVertexIDContainerType<GUNDAM::ContainerType::Map>,
  //     GUNDAM::SetVertexPtrContainerType<GUNDAM::ContainerType::Vector>,
  //     GUNDAM::SetEdgeLabelContainerType<GUNDAM::ContainerType::Vector>,
  //     GUNDAM::SetVertexAttributeKeyType<std::string>,
  //     GUNDAM::SetEdgeIDType<EdgeIDType>,
  //     GUNDAM::SetEdgeAttributeStoreType<GUNDAM::AttributeType::kGrouped>,
  //     // GUNDAM::SetEdgeAttributeStoreType<GUNDAM::AttributeType::kSeparated>,
  //     GUNDAM::SetEdgeLabelType<EdgeLabelType>,
  //     GUNDAM::SetEdgeAttributeKeyType<std::string>>;

  using ProbGarType = prob_gar::ProbGraphAssociationRule<Pattern, DataGraph>;

  using CandidateSet = std::map<typename GUNDAM::VertexHandle< Pattern >::type, 
                    std::vector<typename GUNDAM::VertexHandle<DataGraph>::type>>;

  inline void ToNextProbGarSet() {
    assert(this->current_processing_prob_gar_idx_ 
         < this->all_prob_gar_set_.size());
    this->current_processing_prob_gar_idx_++;
    return;
  }

  inline bool HasProbGarToProcess() const {
    return this->current_processing_prob_gar_idx_ 
         < this->all_prob_gar_set_.size();
  }

  inline bool HasProcessedAllProbGar() const {
    return !this->HasProbGarToProcess();
  }

  inline std::vector<ProbGarType>& CurrentProbGarSet() {
    assert(this->current_processing_prob_gar_idx_ >= 0 
        && this->current_processing_prob_gar_idx_ < this->all_prob_gar_set_.size());
    return this->all_prob_gar_set_[this->current_processing_prob_gar_idx_];
  }

  inline std::string CurrentProbGarSetName() const {
    std::string prob_gar_set_name;
    for (const auto& prob_gar_path 
         : this->all_prob_gar_path_set_[this->current_processing_prob_gar_idx_]) {
      prob_gar_set_name += "# prob gar # vfile: " + std::get<0>(prob_gar_path)
                                   + " # efile: " + std::get<1>(prob_gar_path)
                                   + " # xfile: " + std::get<2>(prob_gar_path)
                                   + " # yfile: " + std::get<3>(prob_gar_path)
                                   + " # pfile: " + std::get<4>(prob_gar_path)
                                   + " #";
    }
    return prob_gar_set_name; 
  }

  inline const size_t ProbGarSetSize() const {
    return this->all_prob_gar_set_.size();
  }

  inline void ResetParameter() {
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

  std::ofstream time_log_file_;

  std::vector<
  std::vector<std::tuple<std::string,
                         std::string,
                         std::string,
                         std::string,
                         std::string>>> all_prob_gar_path_set_;

  std::vector<std::vector<ProbGarType>> all_prob_gar_set_;

 private:
  size_t current_processing_prob_gar_idx_;

  double total_match_time_;
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_PROB_GAR_CHASE_PROB_GAR_CHASE_CONTEXT_H_
