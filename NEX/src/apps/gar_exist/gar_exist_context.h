#ifndef EXAMPLES_ANALYTICAL_APPS_GAR_EXIST_GAR_EXIST_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_GAR_EXIST_GAR_EXIST_CONTEXT_H_

#include <grape/app/context_base.h>
#include <grape/grape.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "util/file.h"

namespace grape {



/**
 * @brief Context for the parallel version of GarExist.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class GarExistContext : public VertexDataContext<FRAG_T, int64_t> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  explicit GarExistContext(const FRAG_T& fragment)
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

  static constexpr std::string_view kRulePosfix = "_v.csv";

  // using Pattern = GUNDAM::SmallGraph<uint32_t, uint32_t, uint32_t, uint32_t>;
  using Pattern   = GUNDAM::LargeGraph2<VertexIDType, VertexLabelType, std::string,
                                          EdgeIDType,   EdgeLabelType, std::string>;
  using DataGraph = GUNDAM::LargeGraph2<VertexIDType, VertexLabelType, std::string,
                                          EdgeIDType,   EdgeLabelType, std::string>;

  using GARPatternToDataGraph 
      = gar::GraphAssociationRule<Pattern, 
                                      DataGraph>;
  using GARPatternToPattern
      = gar::GraphAssociationRule<Pattern, 
                                        Pattern>;

  std::string yaml_file_;
  int fid_;
  int frag_num_;
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_GAR_EXIST_GAR_EXIST_CONTEXT_H_
