#ifndef EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_DELIVER_PREDICATE_H_
#define EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_DELIVER_PREDICATE_H_

#include <vector>
#include <string>

#include "protocol.h"

#include "include/gar/gar.h"
#include "include/gcr/gcr_supp.h"

namespace grape {

namespace _gcr_discover {

/* ################################################### *
 * ##  received the delivered predicates, evaluate  ## *
 * ##  all of them, send the legal ones back to     ## *
 * ##  the coordinator                              ## *
 * ################################################### */
template <typename ContextType>
std::string ReceiveDeliverPredicate(ContextType& ctx) {
  using GcrType = typename ContextType::GcrType;

  omp_lock_t message_lock;
  omp_init_lock(&message_lock);

  decltype(ctx.predicate_set_pattern_literal_) coordinator_predicate_set;
  
  std::string message;
  #pragma omp parallel for schedule(dynamic) 
  for (size_t predicate_idx = 0;
              predicate_idx < ctx.predicate_set_pattern_literal_.size();
              predicate_idx++) {

    auto& predicate = ctx.predicate_set_pattern_literal_[predicate_idx];
    auto& [pattern, literal] = predicate;

    GcrType gcr(pattern);
    gcr.AddY(literal);
    auto [x_supp, xy_supp] = gcr::GcrSupp(gcr, ctx.data_graph_);

    if (xy_supp < ctx.support_bound_) {
      // not legal
      continue;
    }

    if (ctx.fid_ == kCoordinatorID) {
      omp_set_lock(&message_lock);
      coordinator_predicate_set.emplace_back(std::move(predicate));
      omp_unset_lock(&message_lock);
      continue;
    }

    // legal
    omp_set_lock(&message_lock);
    message << predicate;
    omp_unset_lock(&message_lock);
  }

  ctx.predicate_set_pattern_literal_.clear();
  if (ctx.fid_ == kCoordinatorID) {
    ctx.predicate_set_pattern_literal_.swap(coordinator_predicate_set);
  }
  return message;
}

}; // _gcr_discover

}; // grape


#endif // EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_DELIVER_PREDICATE_H_