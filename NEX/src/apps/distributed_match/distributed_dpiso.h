#ifndef DISTRIBUTED_DPISO_H_
#define DISTRIBUTED_DPISO_H_
#include <map>
#include <set>

#include "gundam/algorithm/dp_iso.h"
#include "gundam/type_getter/vertex_handle.h"
#include "gundam/type_getter/edge_handle.h"

namespace grape {
enum EdgeState { kIn, kOut };
enum MatchSemantics { kIsomorphism, kHomomorphism };
// using (edge_label,dst_label) Filter
template <class Pattern, class DataGraphWithIndex, class PatternVertexPtr,
          class DataGraphVertexPtr>
inline bool NeighborLabelFrequencyFiltering(
      PatternVertexPtr pattern_vertex_ptr,
    DataGraphVertexPtr    data_vertex_ptr) {
  if (!GUNDAM::_dp_iso::NeighborLabelFrequencyFiltering(pattern_vertex_ptr,
                                                           data_vertex_ptr)) {
    return false;
  }
  using VertexLabelType = typename Pattern::VertexType::LabelType;
  using   EdgeLabelType = typename Pattern::  EdgeType::LabelType;
  using   PatternVertexConstPtr = typename GUNDAM::VertexHandle<Pattern>::type;
  using DataGraphVertexConstPtr = typename GUNDAM::VertexHandle<DataGraphWithIndex>::type;
  using IndexKeyType = std::pair<EdgeLabelType, VertexLabelType>;
  using DataGraphVertexType = typename DataGraphWithIndex::FragmentVertexType;
  std::map<IndexKeyType, size_t> out_edge_degree_count, in_edge_degree_count;
  std::map<IndexKeyType, std::set<PatternVertexConstPtr>>
      out_adj_vertex_with_index, in_adj_vertex_with_index;
  for (auto edge_it = pattern_vertex_ptr->OutEdgeBegin(); 
           !edge_it.IsDone();
            edge_it++) {
    PatternVertexConstPtr dst_ptr = edge_it->dst_handle();
    IndexKeyType index_key{edge_it->label(), dst_ptr->label()};
    out_edge_degree_count[index_key]++;
    out_adj_vertex_with_index[index_key].emplace(dst_ptr);
  }
  for (auto edge_it = pattern_vertex_ptr->InEdgeBegin(); 
           !edge_it.IsDone();
            edge_it++) {
    PatternVertexConstPtr src_ptr = edge_it->src_handle();
    IndexKeyType index_key{edge_it->label(), src_ptr->label()};
    in_edge_degree_count[index_key]++;
    in_adj_vertex_with_index[index_key].emplace(src_ptr);
  }
  bool satisfy_filter_flag = true;
  for (const auto &[index_key, edge_count] : out_edge_degree_count) {
    if (!satisfy_filter_flag) break;
    if (data_vertex_ptr->CountOutEdge(index_key.first, index_key.second) <
        edge_count) {
      satisfy_filter_flag = false;
      break;
    }
  }
  for (const auto &[index_key, edge_count] : out_adj_vertex_with_index) {
    if (!satisfy_filter_flag) break;
    if (data_vertex_ptr->CountOutVertex(index_key.first, index_key.second) <
        edge_count.size()) {
      satisfy_filter_flag = false;
      break;
    }
  }
  for (const auto &[index_key, edge_count] : in_edge_degree_count) {
    if (!satisfy_filter_flag) break;
    if (data_vertex_ptr->CountInEdge(index_key.first, index_key.second) <
        edge_count) {
      satisfy_filter_flag = false;
      break;
    }
  }
  for (const auto &[index_key, edge_count] : in_adj_vertex_with_index) {
    if (!satisfy_filter_flag) break;
    if (data_vertex_ptr->CountInVertex(index_key.first, index_key.second) <
        edge_count.size()) {
      satisfy_filter_flag = false;
      break;
    }
  }
  return satisfy_filter_flag;
}

// init candidate set in distributed-dpiso
template <class Pattern, class DataGraphWithIndex, class CandidateSet>
inline bool InitCandidateSet(Pattern &pattern,
                             DataGraphWithIndex &index_graph,
                             CandidateSet &candidate_set) {
  using VertexLabelType = typename Pattern::VertexType::LabelType;
  using EdgeLabelType = typename Pattern::EdgeType::LabelType;
  using PatternVertexConstPtr = typename GUNDAM::VertexHandle<Pattern>::type;
  using DataGraphVertexConstPtr = typename GUNDAM::VertexHandle<DataGraphWithIndex>::type;
  using IndexKeyType = std::pair<EdgeLabelType, VertexLabelType>;
  using DataGraphVertexType = typename DataGraphWithIndex::FragmentVertexType;
  for (auto vertex_it = pattern.VertexBegin(); !vertex_it.IsDone();
       vertex_it++) {
    PatternVertexConstPtr pattern_vertex_ptr = vertex_it;
    auto &l = candidate_set[pattern_vertex_ptr];
    for (auto data_vertex_iter =
             index_graph.VertexBegin(pattern_vertex_ptr->label());
         !data_vertex_iter.IsDone(); data_vertex_iter++) {
      DataGraphVertexConstPtr data_vertex_ptr = data_vertex_iter;
      // when data_vertex is outer_vertex,not filter
      if (!index_graph.fragment->IsOuterVertex(data_vertex_iter->vertex())) {
        if (!GUNDAM::_dp_iso::DegreeFiltering(pattern_vertex_ptr,
                                              data_vertex_ptr)) {
          continue;
        }
        if (!NeighborLabelFrequencyFiltering<Pattern, DataGraphWithIndex>(
                pattern_vertex_ptr, data_vertex_ptr)) {
          continue;
        }
      }
      l.emplace_back(data_vertex_ptr);
    }
    if (l.empty()) {
      return false;
    }
  }
  for (auto &it : candidate_set) {
    sort(it.second.begin(), it.second.end());
  }
  return true;
}

template <enum EdgeState edge_state, 
          class QueryVertexPtr,
          class TargetVertexPtr, class CandidateSet>
inline void GetAdjNotMatchedVertexSingleDirection(
          QueryVertexPtr &query_vertex_ptr, 
    const CandidateSet &candidate_set,
    const std::map<QueryVertexPtr, TargetVertexPtr> &match_state,
    const std::map<QueryVertexPtr, int> &match_vertex_belong_to,
    std::set<QueryVertexPtr> &next_query_set) {
  for (auto edge_iter =
           (edge_state == EdgeState::kIn ? query_vertex_ptr->InEdgeBegin()
                                         : query_vertex_ptr->OutEdgeBegin());
       !edge_iter.IsDone(); edge_iter++) {
    QueryVertexPtr query_opp_vertex_ptr =
        edge_state == EdgeState::kIn ? edge_iter->src_handle()
                                     : edge_iter->dst_handle();
    if (match_state.count(query_opp_vertex_ptr) == 0) {
      next_query_set.insert(query_opp_vertex_ptr);
    }
  }
}
template <class QueryVertexPtr, class TargetVertexPtr, class CandidateSet>
inline void GetAdjNotMatchedVertex(
    QueryVertexPtr &query_vertex_ptr, 
    const CandidateSet &candidate_set,
    const std::map<QueryVertexPtr, TargetVertexPtr> &match_state,
    const std::map<QueryVertexPtr, int> &match_vertex_belong_to,
    const int32_t now_frag_id, std::set<QueryVertexPtr> &next_query_set) {
  if (match_vertex_belong_to.find(query_vertex_ptr)->second != now_frag_id) {
    return;
  }
  GetAdjNotMatchedVertexSingleDirection<EdgeState::kOut>(
      query_vertex_ptr, candidate_set, match_state, match_vertex_belong_to,
      next_query_set);
  GetAdjNotMatchedVertexSingleDirection<EdgeState::kIn>(
      query_vertex_ptr, candidate_set, match_state, match_vertex_belong_to,
      next_query_set);
}
template <class QueryVertexPtr, class TargetVertexPtr, class CandidateSet>
inline QueryVertexPtr NextMatchVertex(
    const CandidateSet &candidate_set,
          std::map<QueryVertexPtr, TargetVertexPtr> &match_state,
          std::map<QueryVertexPtr, int> &match_vertex_belong_to,
    const int32_t now_frag_id) {
  std::set<QueryVertexPtr> next_query_set;
  for (auto &[query_vertex_ptr, target_vertex_ptr] : match_state) {
    GetAdjNotMatchedVertex(query_vertex_ptr, candidate_set, match_state,
                           match_vertex_belong_to, now_frag_id, next_query_set);
  }
  if (next_query_set.empty()) {
    // not find next match vertex ,should send to other fragment,return empty
    // ptr.
    return QueryVertexPtr();
  }

  QueryVertexPtr res;
  size_t min = SIZE_MAX;
  for (const auto &query_vertex_ptr : next_query_set) {
    size_t candidate_count =
        candidate_set.find(query_vertex_ptr)->second.size();
    if (candidate_count < min) {
      res = query_vertex_ptr;
      min = candidate_count;
    }
  }
  return res;
}

// need rewrite JoinableCheck
// note: match_state data structure is
// map<PatternVertexConstPtr,DataGraphWithIndex::VertexIDType>
template <enum EdgeState edge_state, 
          class Pattern,
          class DataGraphWithIndex, 
          class MatchMap>
inline bool JoinableCheck(DataGraphWithIndex &index_graph,
         typename GUNDAM::VertexHandle<Pattern>::type &query_vertex_ptr,
         typename GUNDAM::VertexHandle<DataGraphWithIndex>::type &target_vertex_ptr,
                          MatchMap &match_state) {
  using PatternEdgePtr = typename GUNDAM::EdgeHandle<Pattern>::type;
  using DataGraphEdgePtr = typename GUNDAM::EdgeHandle<DataGraphWithIndex>::type;
  using PatternEdgeMapPair = std::pair<typename Pattern::EdgeType::LabelType,
                                       typename GUNDAM::VertexHandle<Pattern>::type>;
  using   PatternVertexPtr = typename GUNDAM::VertexHandle<Pattern>::type;
  using DataGraphVertexPtr = typename GUNDAM::VertexHandle<DataGraphWithIndex>::type;
  using DataGraphEdgeMapPair =
      std::pair<typename DataGraphWithIndex::EdgeType::LabelType,
                typename GUNDAM::VertexHandle<DataGraphWithIndex>::type>;
  std::set<typename DataGraphWithIndex::EdgeType::IDType> used_edge;
  for (auto query_edge_iter =
           ((edge_state == EdgeState::kIn) ? query_vertex_ptr->InEdgeBegin()
                                           : query_vertex_ptr->OutEdgeBegin());
       !query_edge_iter.IsDone(); query_edge_iter++) {
    PatternEdgePtr query_edge_ptr = query_edge_iter;
    PatternVertexPtr query_opp_vertex_ptr =
        (edge_state == EdgeState::kIn) ? query_edge_ptr->src_handle()
                                       : query_edge_ptr->dst_handle();
    auto match_iter = match_state.find(query_opp_vertex_ptr);
    if (match_iter == match_state.end()) continue;

    // change1: match_iter->second ->
    // index_graph.FindVertex(match_iter->second)
    DataGraphVertexPtr query_opp_match_vertex_ptr =
        index_graph.FindVertex(match_iter->second);
    if (query_opp_match_vertex_ptr.IsNull()) {
      continue;
    }
    if (index_graph.fragment->IsOuterVertex(
            query_opp_match_vertex_ptr->vertex()) &&
        index_graph.fragment->IsOuterVertex(target_vertex_ptr->vertex())) {
      continue;
    }

    bool find_target_flag = false;
    for (auto target_edge_iter =
             ((edge_state == EdgeState::kIn)
                  ? target_vertex_ptr->InEdgeBegin(query_edge_ptr->label(),
                                                    query_opp_match_vertex_ptr)
                  : target_vertex_ptr->OutEdgeBegin(
                        query_edge_ptr->label(), query_opp_match_vertex_ptr));
         !target_edge_iter.IsDone(); target_edge_iter++) {
      if (used_edge.count(target_edge_iter->id())) continue;
      find_target_flag = true;
      used_edge.insert(target_edge_iter->id());
      break;
    }
    if (!find_target_flag) return false;
  }
  return true;
}

// need rewrite IsJoinable
// note: match_state data structure is
// map<PatternVertexConstPtr,DataGraphWithIndex::VertexIDType>
// note: target_matched data structure is
// std::set<DataGraphWithIndex::VertexIDType>
template <enum MatchSemantics match_semantics, 
          class Pattern,
          class DataGraphWithIndex, 
          class PatternVertexPtr,
          class DataGraphVertexPtr, class MatchMap, class DataGraphVertexSet>
inline bool IsJoinable(DataGraphWithIndex &index_graph,
                       PatternVertexPtr query_vertex_ptr,
                       DataGraphVertexPtr target_vertex_ptr,
                       const MatchMap &match_state,
                       const DataGraphVertexSet &target_matched) {
  if (match_semantics == MatchSemantics::kIsomorphism &&
      target_matched.count(target_vertex_ptr->id()) > 0) {
    return false;
  }
  if (!JoinableCheck<EdgeState::kIn, Pattern, DataGraphWithIndex>(
          index_graph, query_vertex_ptr, target_vertex_ptr, match_state)) {
    return false;
  }
  if (!JoinableCheck<EdgeState::kOut, Pattern, DataGraphWithIndex>(
          index_graph, query_vertex_ptr, target_vertex_ptr, match_state)) {
    return false;
  }
  return true;
}
// not need rewrite UpdateState,RestoreState,UpdateCandidateSet
// note: match_state data structure is
// map<PatternVertexConstPtr,DataGraphWithIndex::VertexIDType>
// note: target_matched data structure is
// std::set<DataGraphWithIndex::VertexIDType>

template <class Pattern, class DataGraphWithIndex>
inline decltype(auto) MapToMessage(
    const std::map<typename GUNDAM::VertexHandle<Pattern>::type,
                   typename DataGraphWithIndex::VertexType::IDType>
        &match_state,
    const std::map<typename GUNDAM::VertexHandle<Pattern>::type, int>
        &match_vertex_belong_to) {
  std::map<typename Pattern::VertexType::IDType,
           typename DataGraphWithIndex::VertexType::IDType>
      change_result;
  std::map<typename Pattern::VertexType::IDType, int> change_belong_to;
  using ReturnType =
      std::pair<std::map<typename Pattern::VertexType::IDType,
                         typename DataGraphWithIndex::VertexType::IDType>,
                std::map<typename Pattern::VertexType::IDType, int>>;
  for (const auto &it : match_state) {
    change_result.emplace(it.first->id(), it.second);
  }
  for (const auto &it : match_vertex_belong_to) {
    change_belong_to.emplace(it.first->id(), it.second);
  }
  return ReturnType{std::move(change_result), std::move(change_belong_to)};
}

template <class Pattern, class DataGraphWithIndex>
inline decltype(auto) MessageToMap(
    const Pattern &pattern,
    const std::pair<std::map<typename Pattern::VertexType::IDType,
                             typename DataGraphWithIndex::VertexType::IDType>,
                    std::map<typename Pattern::VertexType::IDType, int>>
        &message) {
  std::map<typename GUNDAM::VertexHandle<Pattern>::type,
           typename DataGraphWithIndex::VertexType::IDType>
      change_result;
  std::map<typename GUNDAM::VertexHandle<Pattern>::type, int> change_belong_to;
  using ReturnType =
      std::pair<std::map<typename GUNDAM::VertexHandle<Pattern>::type,
                         typename DataGraphWithIndex::VertexType::IDType>,
                std::map<typename GUNDAM::VertexHandle<Pattern>::type, int>>;
  for (const auto &it : message.first) {
    change_result.emplace(pattern.FindVertex(it.first), it.second);
  }
  for (const auto &it : message.second) {
    change_belong_to.emplace(pattern.FindVertex(it.first), it.second);
  }
  return ReturnType{std::move(change_result), std::move(change_belong_to)};
}

template <class MatchMap, class MatchVertexBelongTo>
inline int PartialMatchShouldBeSendTo(
    const MatchMap &match_state,
    const MatchVertexBelongTo &match_vertex_belong_to) {
  for (auto &[temp_query_vertex_ptr, temp_target_vertex_ptr] : match_state) {
    auto  query_vertex_ptr =  temp_query_vertex_ptr;
    auto target_vertex_ptr = temp_target_vertex_ptr;
    for (auto edge_it = query_vertex_ptr->OutEdgeBegin(); 
             !edge_it.IsDone(); edge_it++) {
      if (match_state.find(edge_it->dst_handle()) == match_state.end()) {
        return match_vertex_belong_to.find(query_vertex_ptr)->second;
      }
    }
    for (auto edge_it = query_vertex_ptr->InEdgeBegin();
             !edge_it.IsDone(); edge_it++) {
      if (match_state.find(edge_it->src_handle()) == match_state.end()) {
        return match_vertex_belong_to.find(query_vertex_ptr)->second;
      }
    }
  }
  return -1;
}
template <class QueryVertexPtr, class TargetVertexPtr, class MatchStateMap,
          class TargetVertexSet, class MatchVertxBelongTo>
inline void UpdateState(QueryVertexPtr query_vertex_ptr,
                        TargetVertexPtr target_vertex_ptr,
                        int target_vertex_fragment_id,
                        MatchStateMap &match_state,
                        TargetVertexSet &target_matched,
                        MatchVertxBelongTo &match_vertex_belong_to) {
  GUNDAM::_dp_iso::UpdateState(query_vertex_ptr, target_vertex_ptr->id(),
                               match_state, target_matched);
  match_vertex_belong_to.emplace(query_vertex_ptr, target_vertex_fragment_id);
}
template <class QueryVertexPtr, class TargetVertexPtr, class MatchStateMap,
          class TargetVertexSet, class MatchVertxBelongTo>
inline void RestoreState(QueryVertexPtr query_vertex_ptr,
                         TargetVertexPtr target_vertex_ptr,
                         int target_vertex_fragment_id,
                         MatchStateMap &match_state,
                         TargetVertexSet &target_matched,
                         MatchVertxBelongTo &match_vertex_belong_to) {
  GUNDAM::_dp_iso::RestoreState(query_vertex_ptr, target_vertex_ptr->id(),
                                match_state, target_matched);
  match_vertex_belong_to.erase(query_vertex_ptr);
}
// need rewrite main function
// note: match_state data structure is
// map<PatternVertexConstPtr,DataGraphWithIndex::VertexIDType>
// note: target_matched data structure is
// std::set<DataGraphWithIndex::VertexIDType>
// note:message container data structure is
// std::vector<std::vector<std::map<Pattern::VertexIDType,DataGraphWithIndex::VertexIDType>>>
template <enum MatchSemantics match_semantics = MatchSemantics::kIsomorphism,
          class Pattern, class DataGraphWithIndex, class CandidateSet,
          class MatchMap, class DataGraphVertexSet, class MatchVertexBelongTo,
          class MessageContainer, class MatchCallback, class PruneCallback>
bool _DistributedMatch(Pattern &pattern,
                       DataGraphWithIndex &index_graph,
                       const CandidateSet &candidate_set, 
                       MatchMap &match_state,
                       DataGraphVertexSet &target_matched,
                       MatchVertexBelongTo &match_vertex_belong_to,
                       MessageContainer &message_container,
                       size_t &result_count, MatchCallback user_callback,
                       PruneCallback prune_callback) {
  if (prune_callback(match_state)) {
    return true;
  }
  if (match_state.size() == pattern.CountVertex()) {
    result_count++;
    return user_callback(match_state);
  }
  using   PatternVertexPtr = typename GUNDAM::VertexHandle<Pattern>::type;
  using DataGraphVertexPtr = typename GUNDAM::VertexHandle<DataGraphWithIndex>::type;

  PatternVertexPtr next_query_vertex_ptr =
      NextMatchVertex(candidate_set, match_state, match_vertex_belong_to,
                      index_graph.fragment->fid());
  if (next_query_vertex_ptr.IsNull()) {
    int frag_id =
        PartialMatchShouldBeSendTo(match_state, match_vertex_belong_to);
    message_container[frag_id].push_back(
        std::move(MapToMessage<Pattern, DataGraphWithIndex>(
            match_state, match_vertex_belong_to)));
    return true;
  }
  for (const DataGraphVertexPtr &next_target_vertex_ptr :
       candidate_set.find(next_query_vertex_ptr)->second) {
    if (prune_callback(match_state)) {
      return true;
    }
    if (IsJoinable<match_semantics, Pattern, DataGraphWithIndex>(
            index_graph, next_query_vertex_ptr, next_target_vertex_ptr,
            match_state, target_matched)) {
      CandidateSet temp_candidate_set = candidate_set;
      UpdateState(
          next_query_vertex_ptr, next_target_vertex_ptr,
          index_graph.fragment->GetFragId(next_target_vertex_ptr->vertex()),
          match_state, target_matched, match_vertex_belong_to);
      if (!index_graph.fragment->IsOuterVertex(
              next_target_vertex_ptr->vertex())) {
        GUNDAM::_dp_iso::UpdateCandidateSet<Pattern, DataGraphWithIndex>(
            next_query_vertex_ptr, next_target_vertex_ptr, temp_candidate_set,
            match_state, target_matched);
        if (!_DistributedMatch<match_semantics, Pattern, DataGraphWithIndex>(
                pattern, index_graph, temp_candidate_set, match_state,
                target_matched, match_vertex_belong_to, message_container,
                result_count, user_callback, prune_callback)) {
          return false;
        }
      } else {
        message_container[index_graph.fragment->GetFragId(
                              next_target_vertex_ptr->vertex())]
            .push_back(MapToMessage<Pattern, DataGraphWithIndex>(
                match_state, match_vertex_belong_to));
      }
      RestoreState(
          next_query_vertex_ptr, next_target_vertex_ptr,
          index_graph.fragment->GetFragId(next_target_vertex_ptr->vertex()),
          match_state, target_matched, match_vertex_belong_to);
    }
  }

  return true;
}
}  // namespace grape

#endif