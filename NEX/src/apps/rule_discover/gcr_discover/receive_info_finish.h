#ifndef EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_INFO_FINISH_H_
#define EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_INFO_FINISH_H_

#include <vector>
#include <string>

#include "protocol.h"

namespace grape {

namespace _gcr_discover {

/* ################################################ *
 * ##  told all workers to begin the next round  ## *
 * ################################################ */
template <typename ContextType>
std::string ReceiveInfoFinish(ContextType& ctx) {
  return "";
}

}; // _gcr_discover

}; // grape

#endif // EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_INFO_FINISH_H_