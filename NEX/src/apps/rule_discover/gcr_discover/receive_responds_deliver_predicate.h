#ifndef EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_RESPONDS_DELIVER_PREDICATE_H_
#define EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_RESPONDS_DELIVER_PREDICATE_H_

#include <vector>
#include <string>

#include "protocol.h"

namespace grape {

namespace _gcr_discover {

/* ############################################## *
 * ##  collect all legal predicates, and send  ## *
 * ##  them to all workers                     ## *
 * ############################################## */
template <typename ContextType>
std::string ReceiveRespondsDeliverPredicate(ContextType& ctx) {
  assert(ctx.fid_ == kCoordinatorID);
  std::string msg;
  
  for (auto& predicate : ctx.predicate_set_pattern_literal_) {
    msg << predicate;
  }
  return msg;
}

}; // _gcr_discover

}; // grape


#endif // EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_RESPONDS_DELIVER_PREDICATE_H_