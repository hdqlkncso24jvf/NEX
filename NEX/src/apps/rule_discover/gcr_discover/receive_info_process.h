#ifndef EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_INFO_PROCESS_H_
#define EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_INFO_PROCESS_H_

#include <vector>
#include <string>

#include "protocol.h"

namespace grape {

namespace _gcr_discover {

/* ############################################ *
 * ##  deliever the generated predicates to  ## *
 * ##  all workers to evalute                ## *
 * ############################################ */
template <typename ContextType>
std::vector<std::string> ReceiveInfoProcess(ContextType& ctx) {
  assert(ctx.fid_ == kCoordinatorID);
  std::vector<std::string> msg_to_workers(ctx.frag_num_, "");

  std::vector<uint32_t> worker_id_set;
  for (const auto& [worker_id, process_num] : ctx.process_num_) {
    for (int i = 0; i < process_num; i++) { 
      worker_id_set.emplace_back(worker_id);
    }
  }
  std::random_shuffle ( worker_id_set.begin(), 
                        worker_id_set.end() );

  decltype(ctx.predicate_set_pattern_literal_) coordinator_predicate_set;
  coordinator_predicate_set.reserve(ctx.predicate_set_pattern_literal_.size());
  for (auto& predicate : ctx.predicate_set_pattern_literal_) {
    const auto& to_worker_id 
                 = worker_id_set[std::rand() 
                 % worker_id_set.size()];
    if (to_worker_id != kCoordinatorID) {
      msg_to_workers[to_worker_id] << predicate;
      continue;
    }
    coordinator_predicate_set.emplace_back(std::move(predicate));
  }
  ctx.predicate_set_pattern_literal_.swap(coordinator_predicate_set);
  return msg_to_workers;
}

}; // _gcr_discover

}; // grape


#endif // EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_INFO_PROCESS_H_