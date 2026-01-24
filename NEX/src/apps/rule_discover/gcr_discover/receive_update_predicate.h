#ifndef EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_UPDATE_PREDICATE_H_
#define EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_UPDATE_PREDICATE_H_

#include <vector>
#include <string>

#include "protocol.h"

namespace grape {

namespace _gcr_discover {

/* ############################################## *
 * ##  received all legal predicates from the  ## *
 * ##  coordinator, collect all of them,       ## *
 * ##  each worker selects a subset of it to   ## *
 * ##  build a forest                          ## *
 * ############################################## */
template <typename ContextType>
std::string ReceiveUpdatePredicate(ContextType& ctx) {
  assert(ctx.fid_ == kCoordinatorID);

  using GcrType = typename ContextType::GcrType;

  ctx.forest_       .reserve((ctx.predicate_set_pattern_literal_.size() / ctx.frag_num_) + 1);
  ctx.predicate_set_.reserve( ctx.predicate_set_pattern_literal_.size() );
  for (size_t predicate_idx = 0;
              predicate_idx < ctx.predicate_set_pattern_literal_.size();
              predicate_idx++) {
    GcrType gcr(ctx.predicate_set_pattern_literal_[predicate_idx].first);
    gcr.AddY(ctx.predicate_set_pattern_literal_[predicate_idx].second);
    ctx.predicate_set_.emplace_back(gcr);
    if (predicate_idx % ctx.frag_num_ != ctx.fid_ ) {
      continue;
    }
    std::string literal_info_str;
    literal_info_str << ctx.predicate_set_pattern_literal_[predicate_idx].second;
    // is belong to this forest
    ctx.forest_.emplace_back();
    
    std::get<0>(ctx.forest_.back()) = gcr;
  }

  return "";
}

}; // _gcr_discover

}; // grape

#endif // EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_UPDATE_PREDICATE_H_