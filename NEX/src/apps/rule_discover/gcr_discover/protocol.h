#ifndef EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_PROTOCOL_H_
#define EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_PROTOCOL_H_

#include <string>
#include "include/gar/literal_info.h"

namespace grape {

namespace _gcr_discover {

// the fragment calculate confidence and deliever the file name of the gars
// to other workers
static constexpr int kCoordinatorID = 0;

// from ordinary workers to kCoordinatorFragID
//    info the kCoordinatorFragID how many process each worker have
static constexpr std::string_view kInfoProcessPrefix = "#info_process";

static constexpr std::string_view         kDeliverPredicatePrefix =          "#deliver_predicate";
static constexpr std::string_view kRespondsDeliverPredicatePrefix = "#responds_deliver_predicate";
static constexpr std::string_view          kUpdatePredicatePrefix =           "#update_predicate";

static constexpr std::string_view  kInfoBeginPrefix = "#info_begin";
static constexpr std::string_view kInfoFinishPrefix = "#info_finish";


template <typename GraphPatternType,
          typename    DataGraphType>
std::string& operator<<(std::string& out_string,
        const std::pair<GraphPatternType,
       gar::LiteralInfo<GraphPatternType,
                           DataGraphType>>& predicate) {

  out_string = std::move(out_string) + " <predicate";

  std::string pattern_str;
  pattern_str << predicate.first;

  out_string = std::move(out_string) + pattern_str;

  std::string literal_str;
  literal_str << predicate.second;

  out_string = std::move(out_string) + literal_str;

  out_string = std::move(out_string) + " >"; 

  return out_string;
}

template <typename GraphPatternType,
          typename    DataGraphType>
std::string& operator>>(std::string& in_string,
              std::pair<GraphPatternType,
       gar::LiteralInfo<GraphPatternType,
                           DataGraphType>>& predicate) {

  using DataGraphLiteralType = gar::LiteralInfo<GraphPatternType,
                                                   DataGraphType>;

  using PatternVertexIDType = typename GraphPatternType
                                           ::VertexType
                                               ::IDType;

  std::stringstream ss;
  ss << in_string;

  std::string str;

  ss >> str;
  assert(str == "<predicate");

  // deserilize graph pattern
  getline(ss, str);
  str >> predicate.first;
  ss.clear();
  ss << str;

  getline(ss, str);
  str >> predicate.second;
  ss.clear();
  ss << str;

  ss >> str;
  assert(str == ">");  // predicate

  return in_string;
}

}; // _gcr_discover

}; // grape


#endif // EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_PROTOCOL_H_