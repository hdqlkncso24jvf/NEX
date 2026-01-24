#ifndef EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_EXPAND_LITERAL_H_
#define EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_EXPAND_LITERAL_H_

#include "gundam/graph_type/graph.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/large_graph2.h"
#include "gundam/graph_type/small_graph.h"
#include "gundam/graph_statistics/graph_basic_statistics.h"

#include "gundam/tool/connected.h"
#include "gundam/tool/connected_component.h"
#include "gundam/tool/topological/star/is_star.h"
#include "gundam/tool/topological/path/is_path.h"

#include "gundam/type_getter/vertex_label.h"
#include "gundam/type_getter/edge_label.h"

namespace grape {

namespace _gcr_discover {

// #################
// #  optimize me  #
// #################
// expand rhs literal
template <typename GraphPatternType,
          typename    DataGraphType>
std::vector<
gar::LiteralInfo<GraphPatternType,
                    DataGraphType>> ExpandLiteral(
                 GraphPatternType& pattern,
   const GUNDAM::GraphBasicStatistics<
                    DataGraphType>& graph_basic_statistics,
   const _gar_discover
    ::Restriction<GraphPatternType,
                     DataGraphType>& restriction) {
                                
  using PatternVertexHandleType      = typename GUNDAM::VertexHandle<GraphPatternType>::type;
  using PatternVertexIDType          = typename GUNDAM::VertexID    <GraphPatternType>::type;
  using PatternVertexConstHandleType = typename GUNDAM::VertexHandle<const GraphPatternType>::type;

  using LiteralInfoType = gar::LiteralInfo<GraphPatternType,
                                              DataGraphType>;

  std::vector<LiteralInfoType> expanded_literal_info;

  std::vector<GraphPatternType> connected_component_set
                      = GUNDAM::ConnectedComponent(pattern);

  /* ############################ *
   * ##  add constant literal  ## *
   * ############################ */
  if (restriction.consider_constant_literal()) {
    for (const auto& connected_component
                   : connected_component_set) {
      assert(GUNDAM::IsPath(connected_component));
      // enumerate only constant literal at end point
      auto [ end_point_0,
             end_point_1 ] = GUNDAM::PathEndPoints(connected_component);
      assert(end_point_0);
      assert(end_point_1);
      
      std::vector<PatternVertexConstHandleType> end_point_set 
                                            = { end_point_0,
                                                end_point_1 };

      for (const auto& end_point : end_point_set) {
        // iterator
        auto attr_it  = graph_basic_statistics.legal_attr_set().find(end_point->label());
        if ( attr_it == graph_basic_statistics.legal_attr_set().end()) {
          // this label does not have legal attr
          continue;
        }
        for (const auto& [attr_key, attr_value_container] : attr_it->second) {
          // enumerate A contained in x
          // A = attr_key
          for (const auto& [attr_value, 
                            attr_value_counter] 
                          : attr_value_container) {
            // enumerate the c for x.A
            expanded_literal_info.emplace_back(end_point->id(), 
                                              attr_key, 
                                              attr_value.first,  // value_str
                                              attr_value.second);// value_type
          }
        }
      }
    } 

  }
  if (connected_component_set.size() == 1) {
    // has only one path, enumerate constant literal only
    return expanded_literal_info;
  }

  /* ################################## *
   * ##  enumerate variable literal  ## *
   * ################################## */

  if (restriction.consider_variable_literal()) {

    std::vector<
    std::vector<PatternVertexConstHandleType>> two_path_end_point_set;

    for (const auto& connected_component
                   : connected_component_set) {
      assert(GUNDAM::IsPath(connected_component));
      // enumerate only constant literal at end point
      auto [ end_point_0,
             end_point_1 ] = GUNDAM::PathEndPoints(connected_component);

      two_path_end_point_set.emplace_back(std::vector{end_point_0,
                                                      end_point_1});
    }

    assert(two_path_end_point_set.size() == 2);

    for (const auto& end_point_0 : two_path_end_point_set[0]) {
      for (const auto& end_point_1 : two_path_end_point_set[1]) {
        assert(end_point_0->id() != end_point_1->id());
        if (end_point_0->id() > end_point_1->id()) {
          continue;
        }
        auto x_attr_ret  = graph_basic_statistics.legal_attr_set().find(end_point_0->label());
        if ( x_attr_ret == graph_basic_statistics.legal_attr_set().end()) {
          continue;
        }
        auto y_attr_ret  = graph_basic_statistics.legal_attr_set().find(end_point_1->label());
        if ( y_attr_ret == graph_basic_statistics.legal_attr_set().end()) {
          continue;
        }
        for (const auto& [x_attr_key, 
                          x_attr_value_set] : x_attr_ret->second) {
          // enumerate B, e.g. the attribute key of y
          for (const auto& [y_attr_key, 
                            y_attr_value_set] : y_attr_ret->second) {
            if (x_attr_key != y_attr_key) {
              // only add variable literal with same key
              continue;
            }
            expanded_literal_info.emplace_back(end_point_0->id(), x_attr_key, 
                                               end_point_1->id(), y_attr_key);
          }
        }
      }
    }

  }

  return expanded_literal_info;
}

};  // namespace _gcr_discover

}  // namespace grape

#endif  // _EXPAND_LITERAL_H