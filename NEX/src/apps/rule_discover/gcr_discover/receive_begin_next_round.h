#ifndef EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_BEGIN_NEXT_ROUND_H_
#define EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_BEGIN_NEXT_ROUND_H_

#include <vector>
#include <string>
#include <omp.h>

#include "include/gar/gar.h"
#include "include/gundam/include/gundam/util/permutation.h"
#include "rule_discover/gar_discover/export_gar.h"

#include "protocol.h"

namespace grape {

namespace _gcr_discover {

// template<typename GraphPatternType,
//          typename    DataGraphType> 
// std::vector<
// gar::GraphAssociationRule<
//          GraphPatternType,
//             DataGraphType>>
// MergeCentral(const gar::GraphAssociationRule<
//                             GraphPatternType,
//                                DataGraphType>& gcr,
//                       const GraphPatternType& path,
//                       const gar::LiteralInfo<
//                             GraphPatternType,
//                                DataGraphType>& literal_info,
//         const GUNDAM::Match<GraphPatternType,
//                             GraphPatternType>& path_central_to_gcr_central) {
  
//   using GcrType = gar::GraphAssociationRule<
//                            GraphPatternType,
//                               DataGraphType>;

//   assert(path_central_to_gcr_central.size() == 1);

//   std::vector<GcrType> merged_gcr_set;

//   GraphPatternType merged_pattern = gcr.pattern();
  
//   auto path_to_merged_pattern_match 
//              = MergeGraph(merged_pattern,
//                           path,
//                           path_central_to_gcr_central);

//   for (auto map_it = path_to_merged_pattern_match.MapBegin();
//            !map_it.IsDone();
//             map_it++) {
    
//   }

//   GcrType merged_gcr(merged_pattern);

//   merged_gcr.AddX(literal_info);
  
//   merged_gcr_set.emplace_back(merged_gcr);

//   for (auto ) {

//   }

//   // GraphPatternType merged_pattern = gcr.pattern();
  
//   // MergeGraph(merged_pattern,
//   //            path,
//   //            path_to_gcr_match);

//   return merged_gcr_set;
// }

template<typename GraphPatternType,
         typename    DataGraphType> 
std::vector<
gar::GraphAssociationRule<
         GraphPatternType,
            DataGraphType>> 
  MergePredicate(const gar::GraphAssociationRule<
                                GraphPatternType,
                                   DataGraphType>& gcr,
                 const gar::GraphAssociationRule<
                                GraphPatternType,
                                   DataGraphType>& predicate) {

  using GcrType = gar::GraphAssociationRule<
                           GraphPatternType,
                              DataGraphType>;

  std::cout << "gcr.pattern().CountVertex(): "
            <<  gcr.pattern().CountVertex() << std::endl;

  auto       gcr_cc_set = GUNDAM::ConnectedComponent(      gcr.pattern());
  auto predicate_cc_set = GUNDAM::ConnectedComponent(predicate.pattern());

  using GraphPatternVertexHandleType 
      = typename GUNDAM::VertexHandle<const GraphPatternType>::type;

  std::cout << "gcr_cc_set.size(): "
            <<  gcr_cc_set.size() << std::endl;

  std::vector<GraphPatternVertexHandleType> gcr_end_set,
                                            gcr_central_set;
  for (const auto& gcr_cc : gcr_cc_set) {
    auto [end_points,
          central] = GUNDAM::StarEndPoints(gcr_cc);
    gcr_central_set.emplace_back(central);
    if (end_points[0] == end_points[1]) { 
      gcr_end_set.emplace_back(end_points[0]);
      continue;
    }
    gcr_end_set.emplace_back(end_points[0]);
    gcr_end_set.emplace_back(end_points[1]);
  }
  std::cout << "gcr_central_set.size(): "
            <<  gcr_central_set.size() << std::endl;
  assert(gcr_central_set.size() == 2);
  // std::sort(gcr_central_set.begin(),
  //           gcr_central_set.end());

  std::vector<GraphPatternVertexHandleType> predicate_end_set,
                                            predicate_central_set;
  for (const auto& predicate_cc : predicate_cc_set) {
    auto [end_points,
          central] = GUNDAM::StarEndPoints(predicate_cc);
    predicate_central_set.emplace_back(central);
    if (end_points[0] == end_points[1]) { 
      predicate_end_set.emplace_back(end_points[0]);
      continue;
    }
    predicate_end_set.emplace_back(end_points[0]);
    predicate_end_set.emplace_back(end_points[1]);
  }

  auto permutation_set = GUNDAM::Permutation(gcr_central_set,
                                       predicate_central_set.size());

  std::vector<
  gar::GraphAssociationRule<
       GraphPatternType,
          DataGraphType>> merged_gcr_set;

  for (const auto& permutation : permutation_set) {
    GUNDAM::Match<const GraphPatternType,
                        GraphPatternType> predicate_to_gcr_pattern_match;
    GraphPatternType merged_pattern(gcr.pattern());
    for (size_t predicate_central_idx = 0;
                predicate_central_idx < predicate_central_set.size();
                predicate_central_idx++) {
      // map predicate_central_set[predicate_central_idx]
      //            to permutation[predicate_central_idx]
      predicate_to_gcr_pattern_match
        .AddMap(predicate.pattern().FindVertex(predicate_central_set[predicate_central_idx]->id()),
                   merged_pattern  .FindVertex(          permutation[predicate_central_idx]->id()));
    }

    auto predicate_to_merged_pattern_match 
              = GUNDAM::MergeGraph<
                GUNDAM::MergeGraphReturnType::match_from_graph_1>(
                            merged_pattern,
                            predicate.pattern(),
                            predicate_to_gcr_pattern_match);

    GcrType merged_gcr(std::move(merged_pattern));

    for (auto y_literal_ptr : gcr.y_literal_set()) {
      merged_gcr.AddY(y_literal_ptr->info());
    }
    for (auto x_literal_ptr : gcr.x_literal_set()) {
      merged_gcr.AddX(x_literal_ptr->info());
    }
    auto predicate_literal_info
     = (*predicate.y_literal_set().begin())->info();
    auto predicate_literal_info_on_merged_graph
       = predicate_literal_info.MapTo(
                    predicate_to_merged_pattern_match);
    merged_gcr.AddX(predicate_literal_info_on_merged_graph);
    
    merged_gcr_set.emplace_back(std::move(merged_gcr));
  }
  
  return merged_gcr_set;
}

/* ######################## *
 * ##  begin this round  ## *
 * ######################## */
template <typename ContextType>
std::string ReceiveBeginNextRound(ContextType& ctx) {
  assert(ctx.fid_ == kCoordinatorID);
  std::string msg;

  using CandidateSetType = typename ContextType
                             ::CandidateSetType;

  using  GcrType = typename ContextType
                              ::GcrType;

  using DepthType = typename ContextType
                             ::DepthType;
  using  SuppType = typename ContextType
                              ::SuppType;

  std::vector<GcrType> gcr_to_export;
  std::vector<
  std::tuple<SuppType,
             SuppType>> gcr_supp_to_export;

  omp_lock_t stack_lock;
  omp_init_lock(&stack_lock);
  
  #pragma omp parallel for schedule(dynamic) 
  for (int forest_idx = 0;
           forest_idx < ctx.forest_.size();
           forest_idx++) {

    auto& gcr = std::get<0>(ctx.forest_[forest_idx]);

    CandidateSetType candidate_set;
    if (!GUNDAM::_dp_iso_using_match
               ::InitCandidateSet<GUNDAM::MatchSemantics::kHomomorphism>(
            gcr.pattern(),
            ctx.data_graph_, 
            candidate_set)) {
      continue;
    }
  
    if (!GUNDAM::_dp_iso_using_match
               ::RefineCandidateSet(
            gcr.pattern(),
            ctx.data_graph_, 
            candidate_set)) {
      continue;
    }

    auto [supp_x, supp_xy] = gcr::GcrSupp(gcr, ctx.data_graph_, 
                                               candidate_set);

    std::get<1>(ctx.forest_[forest_idx]) = std::move(candidate_set);
    std::get<2>(ctx.forest_[forest_idx]) = supp_x;
    std::get<3>(ctx.forest_[forest_idx]) = supp_xy;

    omp_set_lock(&stack_lock);
    gcr_to_export.emplace_back(gcr);
    gcr_supp_to_export.emplace_back(supp_x, 
                                    supp_xy);
    omp_unset_lock(&stack_lock);
  }

  for (const auto& [gcr, candidate_set, supp_x, supp_xy] : ctx.forest_) {
    // dfs on each rule tree
    std::stack<
    std::tuple<GcrType, 
      CandidateSetType,
             DepthType,
              SuppType,
              SuppType>> dfs_stack; 

    std::cout << "gcr.pattern().CountVertex(): "
              <<  gcr.pattern().CountVertex() << std::endl;

    dfs_stack.emplace(std::tuple(gcr, candidate_set, 
                                 1, supp_x, supp_xy)) ;

    const size_t kDepthLimit = ctx.round_;

    while (!dfs_stack.empty()) {
      auto [ current_gcr,
             current_candidate_set,
             current_depth,
             current_supp_x,
             current_supp_xy ] = std::move(dfs_stack.top());

      std::cout << "current_gcr.pattern().CountVertex(): "
                <<  current_gcr.pattern().CountVertex() << std::endl;

      dfs_stack.pop();

      if (current_depth >= kDepthLimit) {
        assert(current_depth == kDepthLimit);
        continue;
      }

      std::vector<GcrType> expanded_gcr_set;
      for (const auto& predicate : ctx.predicate_set_) {
        auto current_expanded_gcr_set = MergePredicate( current_gcr,
                                                        predicate );
        expanded_gcr_set.insert(expanded_gcr_set.end(),
         std::make_move_iterator(current_expanded_gcr_set.begin()),
         std::make_move_iterator(current_expanded_gcr_set.end()));
      }

      // for each vertex in current level, merge it 
      // with all predicates 
      #pragma omp parallel for schedule(dynamic) 
      for (size_t expanded_gcr_idx = 0;
                  expanded_gcr_idx < expanded_gcr_set.size();
                  expanded_gcr_idx++) {
        auto& expanded_gcr 
            = expanded_gcr_set[expanded_gcr_idx];
        CandidateSetType expanded_candidate_set;
        if (!GUNDAM::_dp_iso_using_match
                   ::InitCandidateSet<GUNDAM::MatchSemantics::kHomomorphism>(
                expanded_gcr.pattern(),
                ctx.data_graph_, 
                expanded_candidate_set)) {
          continue;
        }

        for (const auto& [pattern_vertex,
                          candidate] : current_candidate_set) {
          assert(expanded_candidate_set.find(expanded_gcr.pattern().FindVertex(pattern_vertex->id()))
              != expanded_candidate_set. end());
          expanded_candidate_set[pattern_vertex] = candidate;
        }
  
        if (!GUNDAM::_dp_iso_using_match
                   ::RefineCandidateSet(
                expanded_gcr.pattern(),
                ctx.data_graph_, 
                expanded_candidate_set)) {
          continue;
        }

        auto [supp_x, supp_xy] = gcr::GcrSupp(expanded_gcr,
                                              ctx.data_graph_, 
                                              expanded_candidate_set);
        if (supp_xy < ctx.support_bound_) { 
          // not satisfy the requries like support bound 
          // would not be considered in the dfs tree
          continue;
        }
        const float kConfidence = ((float) supp_xy) 
                                / ((float) supp_x );
        if (ctx.SpecifiedConfidenceBound() 
         && kConfidence >= ctx.confidence_bound_) {
          // satisfy both the support bound and confidence bound
          // dose not need further expand 
          omp_set_lock(&stack_lock);
          gcr_to_export.emplace_back(std::move(expanded_gcr));
          gcr_supp_to_export.emplace_back(supp_x, 
                                          supp_xy);
          omp_unset_lock(&stack_lock);
          continue;
        }
        if (current_depth == kDepthLimit) {
          // has reached the depth limit, 
          // does not need further expanding
          continue;
        }
        omp_set_lock(&stack_lock);
        dfs_stack.emplace(std::move(expanded_gcr),
                          std::move(expanded_candidate_set),
                          current_depth + 1,
                          std::move(supp_x), 
                          std::move(supp_xy));
        omp_unset_lock(&stack_lock);
      }
    }
  }

  _gar_discover::ExportGARSet(gcr_to_export,
                              ctx.output_gcr_dir_,
                              ctx.round_,
                              ctx.fid_);

  return "";
}

}; // _gcr_discover

}; // grape

#endif // EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_RECEIVE_BEGIN_NEXT_ROUND_H_