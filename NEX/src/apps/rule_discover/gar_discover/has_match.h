#ifndef EXAMPLES_ANALYTICAL_HAS_MATCH_
#define EXAMPLES_ANALYTICAL_HAS_MATCH_

#include <algorithm>

#include "gar/gar_supp.h"

#include "gundam/tool/connected_component.h"
#include "gundam/tool/sub_graph_of.h"
#include "gundam/tool/has_edge.h"

namespace grape {

namespace _gar_discover {

namespace _has_match {
static constexpr float does_not_return_confidence = 0.0;
};

template <typename GraphPatternType,
          typename    DataGraphType, 
          typename CandidateSetContainer>
std::pair<uint64_t, float> HasMatch(GraphPatternType& pattern,
 const std::vector<gar::LiteralInfo<GraphPatternType,
                                       DataGraphType>>& literal_set,
             const gar::LiteralInfo<GraphPatternType,
                                       DataGraphType>&  rhs_literal,
                                       DataGraphType& data_graph,
       const std::map<typename GUNDAM::EdgeLabel<
                                       DataGraphType>::type,
                      typename GUNDAM::EdgeLabel<
                                       DataGraphType>::type>& normal_to_ml_edge_label,
                  const CandidateSetContainer& pattern_candidate_set,
                  const CandidateSetContainer& pattern_candidate_set_removed_nec,
                  const uint64_t supp_bound,
                  const bool only_verify_is_legal,
    const GUNDAM::GraphBasicStatistics<DataGraphType>& graph_statistic,
                  const Restriction<GraphPatternType,
                                       DataGraphType>& restrictions,
                  const double time_limit,
                  const double time_limit_per_supp) {
                  
  const bool kConsiderConfidenceBound 
          = !only_verify_is_legal
          && restrictions.specified_confidence_bound();

  assert(literal_set.size() >= 1);
  #ifndef NDEBUG 
  bool _has_same = false;
  for (const auto& literal : literal_set) {
    if (literal == rhs_literal) {
      assert(!_has_same); // should only have one same
      _has_same = true;
    }
  }
  assert(_has_same);
  #endif // NDEBUG 
  
  if (restrictions.gcr()) {
    /* ######################################################## *
     * ##  pruning:                                          ## *
     * ##    since gcr utilize homomorphism and consider     ## *
     * ##    star-pattern only, when a path without literal  ## *
     * ##    can be contained in another path, this gcr      ## *
     * ##    has been evalauted before and satisfy the       ## *
     * ##    support bound. Need to be evaluated again       ## *
     * ######################################################## */
    // first collect id set of vertexes that are contained in
    // the literals
    std::vector<typename GUNDAM::VertexID<GraphPatternType>::type > 
      vertex_id_set_in_all_literals
                     = rhs_literal.vertex_id_set();
    for (const auto& literal : literal_set) {
      auto vertex_id_set = literal.vertex_id_set();
      vertex_id_set_in_all_literals.insert(
      vertex_id_set_in_all_literals.end(),
        std::make_move_iterator(vertex_id_set.begin()),
        std::make_move_iterator(vertex_id_set.end()));
    }
    std::sort(vertex_id_set_in_all_literals.begin(),
              vertex_id_set_in_all_literals.end());

    vertex_id_set_in_all_literals.erase(
        std::unique(vertex_id_set_in_all_literals.begin(), 
                    vertex_id_set_in_all_literals.end()), 
                    vertex_id_set_in_all_literals.end());

    std::vector<GraphPatternType> 
          connected_components(GUNDAM::ConnectedComponent(pattern));

    for (auto& connected_component : connected_components) {
      const auto [end_vertex_handle_set,
              central_vertex_handle] = GUNDAM::StarEndPoints<true>(connected_component);

      GUNDAM::Match<GraphPatternType,
                    GraphPatternType> partial_match;

      if (kConsiderConfidenceBound) {
        if (central_vertex_handle) {
          partial_match.AddMap(connected_component.FindVertex(central_vertex_handle->id()),
                               connected_component.FindVertex(central_vertex_handle->id()));
        }
        else {
          assert(end_vertex_handle_set.size() == 2);
        }
      }

      for (const auto& vertex_id : vertex_id_set_in_all_literals) {
        if (!connected_component.FindVertex(vertex_id)) {
          continue;
        }
        partial_match.AddMap(connected_component.FindVertex(vertex_id),
                             connected_component.FindVertex(vertex_id));
      }

      auto ret = GUNDAM::MatchUsingMatch<
                 GUNDAM::MatchSemantics::kHomomorphism>(connected_component,
                                                        connected_component,
                                                        partial_match);

      assert(ret >= 1);

      if (ret >= 2) {
        // has been considered before, satisfy support bound but not
        // confidence bound (if specified)
        return std::pair(supp_bound, _has_match::does_not_return_confidence);
      }
    }
  }

  gar::GraphAssociationRule<GraphPatternType,
                                DataGraphType> temp_gar(pattern);
  temp_gar.AddY(rhs_literal);

  const auto kRhsLiteralStandAloneInfo
          = (*temp_gar.y_literal_set()
                      .begin())->stand_alone_info();

  CandidateSetContainer temp_candidate_set,
                        temp_candidate_set_removed_nec;

  auto temp_candidate_set_ptr 
     = std::addressof(pattern_candidate_set),
       temp_candidate_set_removed_nec_ptr 
     = std::addressof(pattern_candidate_set_removed_nec);

  if (!restrictions.specified_rhs_literal_set().empty()) {
    // specified rhs literal set
    if (restrictions.specified_support_set_for_rhs_literal(kRhsLiteralStandAloneInfo)) {
      assert(kRhsLiteralStandAloneInfo.VertexNum() == 2);
      const auto& candidate_set_for_vertex_in_rhs_literal 
                = restrictions
                 .candidate_set_for_vertex_in_rhs_literal(kRhsLiteralStandAloneInfo);
                
      temp_candidate_set             = pattern_candidate_set;
      temp_candidate_set_removed_nec = pattern_candidate_set_removed_nec;
      
      const auto x_handle = pattern.FindVertex(rhs_literal.x_id());
      const auto y_handle = pattern.FindVertex(rhs_literal.y_id());

      std::vector<typename GUNDAM::VertexHandle<GraphPatternType>::type> 
             literal_vertex_handle_set{x_handle, y_handle};

      assert(literal_vertex_handle_set.size() 
          == candidate_set_for_vertex_in_rhs_literal.size());

      // for each vertex contained in rhs literal, intersect its
      // candidate set with the vertex set apeared in the specified
      // support set
      for (size_t vertex_idx = 0; 
                  vertex_idx < literal_vertex_handle_set.size(); 
                  vertex_idx++) {
        const auto& vertex_handle = literal_vertex_handle_set[vertex_idx];
        const auto& specified_candidate_set_for_vertex 
                            = candidate_set_for_vertex_in_rhs_literal[vertex_idx];

         auto  vertex_handle_candidate_set_it  = temp_candidate_set.find(vertex_handle);
        assert(vertex_handle_candidate_set_it != temp_candidate_set.end());
        auto& vertex_handle_candidate_set 
            = vertex_handle_candidate_set_it->second;

         auto  vertex_handle_candidate_set_removed_nec_it  = temp_candidate_set_removed_nec.find(vertex_handle);
        assert(vertex_handle_candidate_set_removed_nec_it != temp_candidate_set_removed_nec.end());
        auto& vertex_handle_candidate_set_removed_nec 
            = vertex_handle_candidate_set_removed_nec_it->second;

        std::vector<typename GUNDAM::VertexHandle<DataGraphType>::type> intersetct_candidate;
        intersetct_candidate.reserve(
                  std::min(vertex_handle_candidate_set.size(),
                    specified_candidate_set_for_vertex.size()));
        std::set_intersection(vertex_handle_candidate_set.begin(),
                              vertex_handle_candidate_set. end (),
                       specified_candidate_set_for_vertex.begin(),
                       specified_candidate_set_for_vertex. end (),
                              std::back_inserter(intersetct_candidate));
        intersetct_candidate.swap(vertex_handle_candidate_set);
        intersetct_candidate.clear();

        intersetct_candidate.reserve(
                  std::min(vertex_handle_candidate_set_removed_nec.size(),
                               specified_candidate_set_for_vertex.size()));
        std::set_intersection(vertex_handle_candidate_set_removed_nec.begin(),
                              vertex_handle_candidate_set_removed_nec. end (),
                                  specified_candidate_set_for_vertex.begin(),
                                  specified_candidate_set_for_vertex. end (),
                              std::back_inserter(intersetct_candidate));
        intersetct_candidate.swap(vertex_handle_candidate_set_removed_nec);
        intersetct_candidate.clear();
      }
      temp_candidate_set_ptr             = std::addressof(temp_candidate_set);
      temp_candidate_set_removed_nec_ptr = std::addressof(temp_candidate_set_removed_nec);
    }
  }

  assert(temp_candidate_set_ptr);
  assert(temp_candidate_set_removed_nec_ptr);

  const CandidateSetContainer& processed_pattern_candidate_set 
                                         = *temp_candidate_set_ptr,
                               processed_pattern_candidate_set_removed_nec
                                         = *temp_candidate_set_removed_nec_ptr;

  if (restrictions.gcr()) {
    // when has only one non-constant literal, can evaluate
    // it through the simulation
    size_t lhs_non_constant_literal_counter = 0,
               lhs_constant_literal_counter = 0,
               lhs_variable_literal_counter = 0;
    gar::LiteralInfo<GraphPatternType,
                        DataGraphType> non_constant_literal;
    assert(non_constant_literal.literal_type() == gar::LiteralType::kNoneLiteral);
    for (const auto& literal : literal_set) {
      assert(literal.literal_type() != gar::LiteralType::kNoneLiteral);
      if (literal.literal_type() != gar::LiteralType::kConstantLiteral) {
        non_constant_literal = literal;
      }
      /* ################################## *
       * ##  statistic the lhs literals  ## *
       * ################################## */
      if (kConsiderConfidenceBound) {
        if (literal == rhs_literal) {
          // does not add rhs literal to lhs
          continue;
        }
      }
      if (literal.literal_type() == gar::LiteralType::kConstantLiteral) {
        lhs_constant_literal_counter++;
        continue;
      }
      lhs_non_constant_literal_counter++;
      if (literal.literal_type() == gar::LiteralType::kVariableLiteral) {
        lhs_variable_literal_counter++;
        continue;
      }
    }

    if ( lhs_non_constant_literal_counter == 0
     || (lhs_non_constant_literal_counter == 1
     && !kConsiderConfidenceBound)
  /* || (lhs_non_constant_literal_counter == 1 // only support one variable literal in lhs now 
          && lhs_variable_literal_counter == 1) */) {
      // can utilize simulation
      const CandidateSetContainer* filted_constant_candidate_set_ptr
       = std::addressof(processed_pattern_candidate_set);

      CandidateSetContainer candidate_set_satisfy_constant_rules;
      if (lhs_constant_literal_counter > 0) {
        // has constant_literal in lhs, filt 
        candidate_set_satisfy_constant_rules = processed_pattern_candidate_set;
        for (const auto& literal : literal_set) {
          if (literal.literal_type() != gar::LiteralType::kConstantLiteral) {
            continue;
          }
          if (kConsiderConfidenceBound) {
            // does not add rhs literal to lhs
            if (literal == rhs_literal) {
              continue;
            }
          }
          // filt out all candidate vertexes not satisfy the constant rules
          const auto x_handle = pattern.FindVertex(literal.x_id());
          const auto x_attr_key   = literal.x_attr_key();
          const auto x_value_str  = literal.c_str();
          const auto x_value_type = literal.data_type();
           auto  x_candidate_set_it  = candidate_set_satisfy_constant_rules.find(x_handle);
          assert(x_candidate_set_it != candidate_set_satisfy_constant_rules.end());
          auto& x_candidate_set 
              = x_candidate_set_it->second;
          std::vector<typename GUNDAM::VertexHandle<DataGraphType>::type> satisfy_literal_x_candidate_set;
          satisfy_literal_x_candidate_set.reserve(
                          x_candidate_set.size());
          for (const auto& dst_handle : x_candidate_set) {
            const auto x_dst_attr_handle = dst_handle->FindAttribute(x_attr_key);
            if (!x_dst_attr_handle) {
              // does not has attribute, not satisfy this literal
              continue;
            }
            // has this attribute
            if (x_dst_attr_handle->value_type() != x_value_type
             || x_dst_attr_handle->value_str()  != x_value_str) {
              // has different type or value, not satisfy the literal
              continue;
            }
            // satisfy literal
            satisfy_literal_x_candidate_set.emplace_back(dst_handle);
          }
          x_candidate_set.swap(satisfy_literal_x_candidate_set);
        }
        GUNDAM::_dp_iso_using_match
              ::RefineCandidateSet(pattern, data_graph, 
                                   candidate_set_satisfy_constant_rules);
        filted_constant_candidate_set_ptr = std::addressof(candidate_set_satisfy_constant_rules);
      }
      const CandidateSetContainer& filted_constant_candidate_set
                                = *filted_constant_candidate_set_ptr;

      // const CandidateSetContainer* filted_variable_candidate_set_ptr
      //  = std::addressof(filted_constant_candidate_set);

      // CandidateSetContainer candidate_set_satisfy_variable_rules;
      // if (lhs_variable_literal_counter > 0) {
      //   // support only one lhs_variable_literal now
      //   assert(lhs_variable_literal_counter == 1);
      //   // evaluate the value space of the variable, filt out all edges
      //   for (const auto& literal : literal_set) {
      //     if (kConsiderConfidenceBound) {
      //       if (literal == rhs_literal) {
      //         // does not add rhs literal to lhs
      //         continue;
      //       }
      //     }
      //     if (literal.literal_type() != gar::LiteralType::kVariableLiteral) {
      //       continue;
      //     }
          
      //     const auto x_handle   = pattern.FindVertex(literal.x_id());
      //     const auto x_attr_key = literal.x_attr_key();
          
      //     const auto y_handle   = pattern.FindVertex(literal.y_id());
      //     const auto y_attr_key = literal.y_attr_key();

      //     assert(graph_basic_static.has_legal_attr_value(x_handle->label(),
      //                                                    x_attr_key));
      //     assert(graph_basic_static.has_legal_attr_value(y_handle->label(),
      //                                                    y_attr_key));

      //     assert(x_handle->label() == y_handle->label());
      //     assert(x_attr_key == y_attr_key);
          
      //     const auto& legal_attr_value_set 
      //               = graph_statistic
      //                .legal_attr_value_set(x_handle->label(),
      //                                      x_attr_key);

      //     CandidateSetContainer temp_candidate_set_satisfy_variable_literal;

      //     for (const auto& [value_type_pair, 
      //                       counter] : legal_attr_value_set) {
      //       const auto& value_str  = value_type_pair.first;
      //       const auto& value_type = value_type_pair.second;

      //       CandidateSetContainer temp_candidate_set_satisfy_legal_value
      //                 (filted_constant_candidate_set);

      //       std::vector<typename GUNDAM::VertexHandle<GraphPatternType>::type> 
      //         vertex_handle_set{x_handle, y_handle};

      //       for (const auto& vertex_handle : vertex_handle_set) {
      //          auto  candidate_set_it  = temp_candidate_set_satisfy_legal_value.find(vertex_handle;
      //         assert(candidate_set_it != temp_candidate_set_satisfy_legal_value.end());
      //         auto&  candidate_set = candidate_set_it->second;
      //         std::vector<typename GUNDAM::VertexHandle<DataGraphType>::type> satisfy_constant_literal_candidate_set;
      //         satisfy_constant_literal_candidate_set.reserve(
      //                                  candidate_set.size());
      //         for (const auto& dst_handle : candidate_set) {
      //           const auto dst_attr_handle = dst_handle->FindAttribute(x_attr_key);
      //           if (!dst_attr_handle) {
      //             // does not has attribute, not satisfy this literal
      //             continue;
      //           }
      //           // has this attribute
      //           if (dst_attr_handle->value_type() != value_type
      //            || dst_attr_handle->value_str()  != value_str) {
      //             // has different type or value, not satisfy the literal
      //             continue;
      //           }
      //           // satisfy literal
      //           satisfy_constant_literal_candidate_set.emplace_back(dst_handle);
      //         }
      //       }
            
      //       GUNDAM::_dp_iso_using_match
      //             ::RefineCandidateSet(pattern, data_graph, 
      //                                  temp_candidate_set_satisfy_legal_value);

      //       for (const auto& [vertex_handle,
      //                         vertex_candidate] : temp_candidate_set_satisfy_legal_value) {

      //       }
      //     }
      //   }
      // }
      // const CandidateSetContainer& filted_variable_candidate_set
      //                           = *filted_variable_candidate_set_ptr;
      
      if (rhs_literal.literal_type() == gar::LiteralType::kConstantLiteral) {
        // all literals are constant literal
        assert(non_constant_literal.literal_type() == gar::LiteralType::kNoneLiteral);
        assert(         rhs_literal.literal_type() == gar::LiteralType::kConstantLiteral);
        const auto x_handle = pattern.FindVertex(rhs_literal.x_id());
        assert(x_handle);
         auto  x_candidate_set_it  = filted_constant_candidate_set.find(x_handle);
        assert(x_candidate_set_it != filted_constant_candidate_set.end());
         auto& x_candidate_set 
             = x_candidate_set_it->second;
        if (!kConsiderConfidenceBound) {
          return std::pair(std::min(x_candidate_set.size(), supp_bound),
                    _has_match::does_not_return_confidence);
        }
        if (x_candidate_set.empty()) {
          return std::pair(0, 0);
        }
         auto  x_supp_counter = x_candidate_set.size();
        assert(x_supp_counter > 0);
        size_t xy_supp_counter = 0;
        const auto x_attr_key   = rhs_literal.x_attr_key();
        const auto x_value_str  = rhs_literal.c_str();
        const auto x_value_type = rhs_literal.data_type();
        for (const auto& dst_handle : x_candidate_set) {
          const auto x_dst_attr_handle = dst_handle->FindAttribute(x_attr_key);
          if (!x_dst_attr_handle) {
            // does not has attribute, not satisfy this literal
            continue;
          }
          if (x_dst_attr_handle->value_type() != x_value_type
           || x_dst_attr_handle->value_str()  != x_value_str) {
            // has different type or value, not satisfy the rhs_literal
            continue;
          }
          // satisfy the rhs_literal
          xy_supp_counter++;
        }
        assert(x_supp_counter > 0);
        return std::pair(xy_supp_counter, ((float) xy_supp_counter) 
                                        / ((float)  x_supp_counter));
      }
      assert(non_constant_literal == rhs_literal);
      switch (rhs_literal.literal_type()) {
       case gar::LiteralType::kVariableLiteral: {
         auto  x_handle = pattern.FindVertex(rhs_literal.x_id());
        assert(x_handle);
         auto  y_handle = pattern.FindVertex(rhs_literal.y_id());
        assert(y_handle);

        const auto& x_attr_key = rhs_literal.x_attr_key();
        const auto& y_attr_key = rhs_literal.y_attr_key();

        assert( filted_constant_candidate_set.find(x_handle)
             != filted_constant_candidate_set.end() );
        const auto&  x_candidate_set 
              = filted_constant_candidate_set.find(x_handle)->second;

        assert( filted_constant_candidate_set.find(y_handle)
             != filted_constant_candidate_set.end() );
        const auto&  y_candidate_set 
              = filted_constant_candidate_set.find(y_handle)->second;

        uint64_t x_supp_counter = x_candidate_set.size()
                                * y_candidate_set.size();
        if (x_supp_counter == 0) {
          return std::pair(0, 0);
        }

        std::map<std::string, typename DataGraphType
                                 ::VertexCounterType> x_histogram,
                                                      y_histogram;

        for (const auto& x_candidate : x_candidate_set) {
          const auto x_attr_handle = x_candidate->FindAttribute(x_attr_key);
          if (!x_attr_handle) {
            // does not have this literal
            continue;
          }
          x_histogram[x_attr_handle->value_str()]++;
        }
        for (const auto& y_candidate : y_candidate_set) {
          const auto y_attr_handle = y_candidate->FindAttribute(y_attr_key);
          if (!y_attr_handle) {
            // does not have this literal
            continue;
          }
          y_histogram[y_attr_handle->value_str()]++;
        }
        /* ##################################### *
         * ## : optimize me             ## *
         * ##     compare the two histograms  ## *
         * ##     in the merge sort manner    ## *
         * ##################################### */
        uint64_t xy_supp_counter = 0;
        for (const auto& [x_str, x_counter] : x_histogram) {
          auto y_histogram_it =  y_histogram.find(x_str);
          if ( y_histogram_it == y_histogram.end() ) {
            continue;
          }
          xy_supp_counter += x_counter * y_histogram_it->second;
          if (kConsiderConfidenceBound) {
            // does not end when reach support bound
            continue;
          }
          if ( xy_supp_counter >= supp_bound ) {
            // has reached the support bound
            assert(!kConsiderConfidenceBound);
            return std::pair(supp_bound, _has_match::does_not_return_confidence);
          }
        }
        if (!kConsiderConfidenceBound) {
          assert(xy_supp_counter < supp_bound);
          return std::pair(xy_supp_counter, _has_match::does_not_return_confidence);
        }
        assert(x_supp_counter > 0);
        return std::pair(xy_supp_counter, ((float) xy_supp_counter) 
                                        / ((float)  x_supp_counter));
       }
       case gar::LiteralType::kEdgeLiteral: {
         auto  x_handle = pattern.FindVertex(rhs_literal.x_id());
        assert(x_handle);
         auto  y_handle = pattern.FindVertex(rhs_literal.y_id());
        assert(y_handle);

        assert( filted_constant_candidate_set.find(x_handle)
             != filted_constant_candidate_set.end() );
        const auto& x_vertex_x_supp_candidate_set 
                           = filted_constant_candidate_set.find(x_handle)->second;

        assert( filted_constant_candidate_set.find(y_handle)
             != filted_constant_candidate_set.end() );
        const auto& y_vertex_x_supp_candidate_set 
                           = filted_constant_candidate_set.find(y_handle)->second;

        uint64_t x_supp_counter = 0;
        if (kConsiderConfidenceBound) {
          // other wise, does not need to process x_supp_counter
          if (!restrictions.specified_support_set_for_rhs_literal(kRhsLiteralStandAloneInfo)) {
            x_supp_counter = x_vertex_x_supp_candidate_set.size()
                           * y_vertex_x_supp_candidate_set.size();
          }
          else {
            for (const auto& support : restrictions.support_set_for_rhs_literal(kRhsLiteralStandAloneInfo)) {
              assert(support.size() == 2);
              if (!std::binary_search(x_vertex_x_supp_candidate_set.begin(),
                                      x_vertex_x_supp_candidate_set.end(),
                                      support[0])) {
                continue;
              }
              if (!std::binary_search(y_vertex_x_supp_candidate_set.begin(),
                                      y_vertex_x_supp_candidate_set.end(),
                                      support[1])) {
                continue;
              }
              x_supp_counter++;
            }
          }
        }

        uint64_t xy_supp_counter = 0;

        if (!restrictions.specified_support_set_for_rhs_literal(kRhsLiteralStandAloneInfo)) {
          for (auto vertex_it = data_graph.VertexBegin(x_handle->label());
                   !vertex_it.IsDone();
                    vertex_it++) {
            typename GUNDAM::VertexHandle<DataGraphType>::type
                      vertex_handle = vertex_it;
            if (!std::binary_search(x_vertex_x_supp_candidate_set.begin(),
                                    x_vertex_x_supp_candidate_set.end(),
                                      vertex_handle)) {
              continue;
            }
            for (auto out_edge_it = vertex_it->OutEdgeBegin();
                     !out_edge_it.IsDone();
                      out_edge_it++) {
              if (out_edge_it->label() != rhs_literal.edge_label()) {
                continue;
              }
              if (out_edge_it->dst_handle()->label() != y_handle->label()) {
                continue;
              }
              if (!std::binary_search(y_vertex_x_supp_candidate_set.begin(),
                                      y_vertex_x_supp_candidate_set.end(),
                                      out_edge_it->dst_handle())) {
                continue;
              }
              xy_supp_counter++;
            }
          }
        }
        else {
          for (const auto& support : restrictions.support_set_for_rhs_literal(kRhsLiteralStandAloneInfo)) {
            assert(support.size() == 2);
            if (!std::binary_search(x_vertex_x_supp_candidate_set.begin(),
                                    x_vertex_x_supp_candidate_set.end(),
                                    support[0])) {
              continue;
            }
            if (!std::binary_search(y_vertex_x_supp_candidate_set.begin(),
                                    y_vertex_x_supp_candidate_set.end(),
                                    support[1])) {
              continue;
            }
            if (!GUNDAM::HasEdge<DataGraphType>(
                                support[0],
                                rhs_literal.edge_label(),
                                support[1])) {
              continue;
            }
            xy_supp_counter++;
          }
        }

        if (!kConsiderConfidenceBound) {
          return std::pair(std::min(xy_supp_counter, supp_bound), 
                         _has_match::does_not_return_confidence);
        }
        if (x_supp_counter == 0) {
          return std::pair(xy_supp_counter, 0);
        }
        assert(x_supp_counter > 0);
        return std::pair(xy_supp_counter, ((float) xy_supp_counter) 
                                        / ((float)  x_supp_counter));
       }
       default:
        assert(false);
        break;
      }
      assert(false);
      return std::pair(0, _has_match::does_not_return_confidence);
    }
  }

  // optimization for edge_literal
  bool gar_pattern_added_edge = false;
  GraphPatternType pattern_with_edge_literal(pattern);
  // there is an edge literal in literal_set, then add it as an edge
  // into the pattern for gar
  typename GUNDAM::EdgeID<GraphPatternType>::type 
    new_edge_id = GUNDAM::MaxEdgeId(pattern_with_edge_literal);
  new_edge_id++;
  for (const auto& literal : literal_set) {
    if (kConsiderConfidenceBound) {
      // does not add rhs literal to lhs
      if (literal == rhs_literal) {
        continue;
      }
    }
    if (literal.literal_type() == gar::LiteralType::kEdgeLiteral) {
      auto [ edge_handle, edge_ret ]
           = pattern_with_edge_literal.AddEdge(literal.x_id(),
                                               literal.y_id(),
                                               literal.edge_label(),
                                               new_edge_id++);
      assert(edge_ret);
      gar_pattern_added_edge = true;
    }
    if (literal.literal_type() == gar::LiteralType::kMlLiteral) {
       auto  ml_edge_label_it  = normal_to_ml_edge_label.find(literal.edge_label());
      assert(ml_edge_label_it != normal_to_ml_edge_label.end());
      auto [ edge_handle, edge_ret ]
           = pattern_with_edge_literal.AddEdge(literal.x_id(),
                                               literal.y_id(),
                                               ml_edge_label_it->second,
                                               new_edge_id++);
      assert(edge_ret);
      gar_pattern_added_edge = true;
    }
  }
  gar::GraphAssociationRule<GraphPatternType,
                               DataGraphType> gar(pattern_with_edge_literal);
  gar.AddY(rhs_literal);
  // add both x and y literal into x
  for (const auto& literal : literal_set) {
    if (literal.literal_type() == gar::LiteralType::kEdgeLiteral){
      // edge literal has already been added into the
      // pattern
      continue;
    }
    if (kConsiderConfidenceBound) {
      // does not add rhs literal to lhs
      if (literal == rhs_literal) {
        continue;
      }
    }
    gar.AddX(literal);
  }
  auto& gar_pattern = gar.pattern();
  CandidateSetContainer gar_pattern_candidate_set;
  for (const auto& processed_pattern_candidate 
                 : processed_pattern_candidate_set) {
    auto pattern_vertex_handle = processed_pattern_candidate.first;
    // should not be nullptr
    assert(pattern_vertex_handle);
    auto gar_pattern_vertex_handle 
       = gar_pattern.FindVertex(pattern_vertex_handle->id());
    // should not be nullptr
    assert(gar_pattern_vertex_handle);
    auto [ gar_pattern_candidate_set_it,
           gar_pattern_candidate_set_ret ]
         = gar_pattern_candidate_set.emplace(gar_pattern_vertex_handle,
                                       processed_pattern_candidate.second);
    // should added successfully
    assert(gar_pattern_candidate_set_ret);
  }
  assert( processed_pattern_candidate_set.size()
            ==  gar_pattern_candidate_set.size() );

  CandidateSetContainer gar_pattern_candidate_set_removed_nec;
  for (const auto& processed_pattern_candidate 
                 : processed_pattern_candidate_set_removed_nec) {
    auto pattern_vertex_handle = processed_pattern_candidate.first;
    // should not be nullptr
    assert(pattern_vertex_handle);
    auto gar_pattern_vertex_handle 
       = gar_pattern.FindVertex(pattern_vertex_handle->id());
    // should not be nullptr
    assert(gar_pattern_vertex_handle);
    auto [ gar_pattern_candidate_set_removed_nec_it,
           gar_pattern_candidate_set_removed_nec_ret ]
         = gar_pattern_candidate_set_removed_nec.emplace(gar_pattern_vertex_handle,
                                                   processed_pattern_candidate.second);
    // should added successfully
    assert(gar_pattern_candidate_set_removed_nec_ret);
  }
  assert( processed_pattern_candidate_set_removed_nec.size()
             == gar_pattern_candidate_set_removed_nec.size() );

  if (gar_pattern_added_edge) {
    // further refine the candidate set for more efficient processing
    if (!GUNDAM::_dp_iso_using_match::RefineCandidateSet(gar_pattern, 
                                                          data_graph, 
                               gar_pattern_candidate_set_removed_nec)) {
      return std::pair(0, _has_match::does_not_return_confidence);
    }
    if (!GUNDAM::_dp_iso_using_match::RefineCandidateSet(gar_pattern, 
                                                          data_graph, 
                                           gar_pattern_candidate_set)) {
      return std::pair(0, _has_match::does_not_return_confidence);
    }
  }

  uint64_t x_supp_counter = 0;

  std::function<bool(const GUNDAM::Match<GraphPatternType, 
                                            DataGraphType>&)>
    satisfy_x_supp_callback 
        = [&x_supp_counter,
           &supp_bound,
           &kConsiderConfidenceBound](
                     const GUNDAM::Match<GraphPatternType,
                                            DataGraphType>& match) {
    assert(kConsiderConfidenceBound
        || x_supp_counter < supp_bound);
    x_supp_counter++;
    if (kConsiderConfidenceBound) {
      // need to satisfy the confidence bound, cannot
      // stop when reach supp_bound
      return true;
    }
    // can stop when reached the support bound
    if (x_supp_counter == supp_bound) {
      // has reached supp_bound
      // stop matching
      return false;
    }
    return true;
  };

  uint64_t xy_supp_counter = 0;

  std::function<bool(const GUNDAM::Match<GraphPatternType, 
                                            DataGraphType>&)> 
    satisfy_xy_supp_callback 
        = [&xy_supp_counter,
           &supp_bound,
           &kConsiderConfidenceBound](
                     const GUNDAM::Match<GraphPatternType,
                                            DataGraphType>& match) {
    assert(kConsiderConfidenceBound
        || xy_supp_counter < supp_bound);
    xy_supp_counter++;
    if (kConsiderConfidenceBound) {
      // need to satisfy the confidence bound, cannot
      // stop when reach supp_bound
      return true;
    }
    // can stop when reached the support bound
    if (xy_supp_counter == supp_bound) {
      // has reached supp_bound
      // stop matching
      return false;
    }
    return true;
  };

  assert(kRhsLiteralStandAloneInfo
     == (*gar.y_literal_set()
             .begin())->stand_alone_info());

  if (!restrictions.specified_support_set_for_rhs_literal(kRhsLiteralStandAloneInfo)) {
    GUNDAM::Match<GraphPatternType,
                    DataGraphType> match_state;
    if (restrictions.gcr()) {
      // gcr, use homomorphism
      auto [x_supp, xy_supp] = gar::GarSupp<GUNDAM::MatchSemantics::kHomomorphism>(
                                            gar, data_graph, match_state,
                                            gar_pattern_candidate_set,
                                            gar_pattern_candidate_set_removed_nec,
                                            satisfy_x_supp_callback, 
                                            satisfy_xy_supp_callback, 
                                            time_limit,
                                            time_limit_per_supp);
      assert(xy_supp == xy_supp_counter);
      assert( x_supp ==  x_supp_counter);
      if (!kConsiderConfidenceBound) {
        assert( x_supp_counter == xy_supp_counter);
        return std::pair(xy_supp_counter, _has_match::does_not_return_confidence);
      }
      if (x_supp_counter == 0) {
        return std::pair(xy_supp_counter, 0);
      }
      assert(x_supp_counter > 0);
      return std::pair(xy_supp_counter, ((float) xy_supp_counter) 
                                      / ((float)  x_supp_counter));
    }
    // gar or horn rule, use isomorphism
    auto [x_supp, xy_supp] = gar::GarSupp<GUNDAM::MatchSemantics::kIsomorphism>(
                                          gar, data_graph, match_state,
                                          gar_pattern_candidate_set,
                                          gar_pattern_candidate_set_removed_nec,
                                          satisfy_x_supp_callback, 
                                          satisfy_xy_supp_callback, 
                                          time_limit,
                                          time_limit_per_supp);
    assert(xy_supp == xy_supp_counter);
    assert( x_supp ==  x_supp_counter);
    if (!kConsiderConfidenceBound) {
      assert( x_supp_counter == xy_supp_counter);
      return std::pair(xy_supp_counter, _has_match::does_not_return_confidence);
    }
    if (x_supp_counter == 0) {
      return std::pair(xy_supp_counter, 0);
    }
    assert(x_supp_counter > 0);
    return std::pair(xy_supp_counter, ((float) xy_supp_counter) 
                                    / ((float)  x_supp_counter));
  }

  // specified support set 
  const auto& support_set = restrictions.support_set_for_rhs_literal(kRhsLiteralStandAloneInfo);
  assert(!support_set.empty());

  const auto kRhsLiteralInfo
         = (*gar.y_literal_set()
                .begin())->info();

  auto begin_time = std::time(NULL);
  for (const auto& support : support_set) {
    assert(kRhsLiteralStandAloneInfo.VertexNum() == support.size());
    assert(kRhsLiteralStandAloneInfo.VertexNum() == 2);

    if (time_limit > 0 
     && time_limit < (std::time(NULL) - begin_time)) {
      // has reached time limit
      break;
    }

    GUNDAM::Match<GraphPatternType,
                     DataGraphType> partial_match_state;

    assert(gar.pattern().FindVertex(kRhsLiteralInfo.x_id()));
    assert(gar.pattern().FindVertex(kRhsLiteralInfo.y_id()));
    partial_match_state.AddMap(gar.pattern().FindVertex(kRhsLiteralInfo.x_id()),
                               support[0]);
    partial_match_state.AddMap(gar.pattern().FindVertex(kRhsLiteralInfo.y_id()),
                               support[1]);
    
    if (restrictions.gcr()) {
      // gcr, use homomorphism
      auto [x_supp, xy_supp] = gar::GarSupp<GUNDAM::MatchSemantics::kHomomorphism>(
                                            gar, data_graph, partial_match_state,
                                            gar_pattern_candidate_set,
                                            gar_pattern_candidate_set_removed_nec,
                                            satisfy_x_supp_callback, 
                                            satisfy_xy_supp_callback, 
                                            time_limit_per_supp,
                                            time_limit_per_supp);
      assert(xy_supp <= x_supp);
      assert(xy_supp <= 1 && x_supp <= 1);
      continue;
    }
    // gar or horn rule, use isomorphism
    auto [x_supp, xy_supp] = gar::GarSupp<GUNDAM::MatchSemantics::kIsomorphism>(
                                          gar, data_graph, partial_match_state,
                                          gar_pattern_candidate_set,
                                          gar_pattern_candidate_set_removed_nec,
                                          satisfy_x_supp_callback, 
                                          satisfy_xy_supp_callback, 
                                          time_limit_per_supp,
                                          time_limit_per_supp);
    assert(xy_supp <= x_supp);
    assert(xy_supp <= 1 && x_supp <= 1);
  }
  if (!kConsiderConfidenceBound) {
    assert( x_supp_counter == xy_supp_counter);
    return std::pair(xy_supp_counter, _has_match::does_not_return_confidence);
  }
  if (x_supp_counter == 0) {
    return std::pair(xy_supp_counter, 0);
  }
  assert(x_supp_counter > 0);
  return std::pair(xy_supp_counter, ((float) xy_supp_counter) 
                                  / ((float)  x_supp_counter));
}

template <typename GraphPatternType,
          typename    DataGraphType, 
          typename CandidateSetContainer>
auto HasMatch(const GraphPatternType& pattern,
              const std::vector<gar::LiteralInfo<GraphPatternType,
                                                    DataGraphType>>& literal_set,
                          const gar::LiteralInfo<GraphPatternType,
                                                    DataGraphType>&  rhs_literal,
                      DataGraphType& data_graph,
              const std::map<typename DataGraphType::EdgeType::LabelType,
                             typename DataGraphType::EdgeType::LabelType>& normal_to_ml_edge_label,
        const CandidateSetContainer& pattern_candidate_set,
              const uint64_t supp_bound,
              const bool only_verify_is_legal,
              const double time_limit,
              const double time_limit_per_supp) {
  return HasMatch(pattern,
                  literal_set,
                  rhs_literal,
                  data_graph,
                  normal_to_ml_edge_label,
                  pattern_candidate_set,
                  pattern_candidate_set,
                  supp_bound,
                  only_verify_is_legal,
                  time_limit,
                  time_limit_per_supp);
}

}  // namespace _gar_discover

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_HAS_MATCH_