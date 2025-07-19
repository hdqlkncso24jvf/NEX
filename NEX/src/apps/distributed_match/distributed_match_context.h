#ifndef EXAMPLES_ANALYTICAL_APPS_DISTRIBUTEDMATCH_DISTRIBUTEDMATCH_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_DISTRIBUTEDMATCH_DISTRIBUTEDMATCH_CONTEXT_H_

#include <grape/app/context_base.h>
#include <grape/grape.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "gundam/io/csvgraph.h"
#include "gundam/algorithm/dp_iso.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/small_graph.h"

namespace grape {

/**
 * @brief Context for the parallel version of Distributed Match.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class DistributedMatchContext : public VertexDataContext<FRAG_T, int64_t> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  using Pattern = GUNDAM::SmallGraph<uint32_t, uint32_t, uint32_t, uint32_t>;
  using PatternVertexPtr = typename GUNDAM::VertexHandle<Pattern>::type;
  using DataGraphVertexPtr =
      typename GUNDAM::VertexHandle<FragmentGraphWithIndex<const FRAG_T>>::type;
  using CandidateSetContainer =
      std::map<PatternVertexPtr, std::vector<DataGraphVertexPtr>>;
  using MatchMap = std::map<PatternVertexPtr, DataGraphVertexPtr>;
  using MessageType =
      std::pair<std::map<typename Pattern::VertexType::IDType,
                         typename FragmentGraphWithIndex<
                             const FRAG_T>::VertexType::IDType>,
                std::map<typename Pattern::VertexType::IDType, int>>;
  using DistributedMap = std::map<
      PatternVertexPtr,
      typename FragmentGraphWithIndex<const FRAG_T>::VertexType::IDType>;
  using DataMatchSet = std::set<
      typename FragmentGraphWithIndex<const FRAG_T>::VertexType::IDType>;
  using MatchVertexBelongTo = std::map<PatternVertexPtr, int>;

  explicit DistributedMatchContext(const FRAG_T& fragment)
      : VertexDataContext<FRAG_T, int64_t>(fragment, true) {
    return;
  }

  void Init(ParallelMessageManager& messages, std::string pattern_v_file,
            std::string pattern_e_file) {
    this->pattern_e_file_ = pattern_e_file;
    this->pattern_v_file_ = pattern_v_file;
    this->supp_.clear();
    this->fid = this->fragment().fid();
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
    if (fid == 0) LOG(INFO) << "last supp = " << this->supp_.size();
  }

#ifdef PROFILING
  double preprocess_time = 0;
  double exec_time = 0;
  double postprocess_time = 0;
#endif
  std::string pattern_e_file_;
  std::string pattern_v_file_;
  std::set<vid_t> supp_;
  int fid;

  Pattern pattern;
  FragmentGraphWithIndex<const FRAG_T> index_graph;
  CandidateSetContainer distributed_candidate_set;
  int round_num = 0;
};
}  // namespace grape

#endif
