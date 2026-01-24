#ifndef EXAMPLES_ANALYTICAL_APPS_DISTRIBUTEDMATCH_DISTRIBUTEDMATCH_H_
#define EXAMPLES_ANALYTICAL_APPS_DISTRIBUTEDMATCH_DISTRIBUTEDMATCH_H_

#include <grape/grape.h>
#include <omp.h>

#include "../fragment2gundam.h"
#include "../fragment_graph.h"
#include "../fragment_graph_with_index.h"
#include "../timer.h"
#include "distributed_dpiso.h"
#include "distributed_match/distributed_match_context.h"
#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"
namespace grape {
template <typename FRAG_T>
class DistributedMatch
    : public ParallelAppBase<FRAG_T, DistributedMatchContext<FRAG_T>>,
      public ParallelEngine {
 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(DistributedMatch<FRAG_T>,
                          DistributedMatchContext<FRAG_T>, FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  using Pattern = typename context_t::Pattern;
  using PatternVertexPtr = typename context_t::PatternVertexPtr;
  using DataGraphVertexPtr = typename context_t::DataGraphVertexPtr;
  using CandidateSetContainer = typename context_t::CandidateSetContainer;
  using MatchMap = typename context_t::MatchMap;
  using MessageType = typename context_t::MessageType;
  using DistributedMap = typename context_t::DistributedMap;
  using DataMatchSet = typename context_t::DataMatchSet;
  using MatchVertexBelongTo = typename context_t::MatchVertexBelongTo;

  /**
   * @brief Partial evaluation for Distributed Match.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();
    messages.InitChannels(thread_num());
    timer_next("build index and load pattern");

    ctx.index_graph.fragment = &frag;
    ctx.index_graph.BuildIndex();
    std::cout << "fragment vertex number: "
              << ctx.index_graph.fragment->GetVerticesNum() << std::endl;
    std::cout << "inner vertex number: "
              << ctx.index_graph.fragment->InnerVertices().size() << std::endl;
    std::cout << "outer vertex number: "
              << ctx.index_graph.fragment->OuterVertices().size() << std::endl;
    GUNDAM::ReadCSVGraph(ctx.pattern, ctx.pattern_v_file_, ctx.pattern_e_file_);
    // cal this fragment's supp
    auto supp_ptr = ctx.pattern.FindVertex(1);
    std::vector<DataGraphVertexPtr> supp2;
    CandidateSetContainer candidate_set;
    timer_next("run dpiso");
    GUNDAM::_dp_iso::InitCandidateSet<GUNDAM::MatchSemantics::kIsomorphism>(
        ctx.pattern, ctx.index_graph, candidate_set);
    GUNDAM::_dp_iso::RefineCandidateSet(ctx.pattern, ctx.index_graph,
                                        candidate_set);
    for (const auto& target_ptr : candidate_set.find(supp_ptr)->second) {
      CandidateSetContainer temp_candidate_set = candidate_set;
      vertex_t target_vertex;
      frag.GetVertex(target_ptr->id(), target_vertex);
      if (frag.IsOuterVertex(target_vertex)) continue;
      MatchMap match_state;
      match_state.emplace(supp_ptr, target_ptr);
      auto user_callback = [&supp_ptr, &supp2](auto& match_state) {
        supp2.push_back(match_state[supp_ptr]);
        return false;
      };
      auto prune_callback = [](auto& match_state) { return false; };
      GUNDAM::DPISO(ctx.pattern, ctx.index_graph, temp_candidate_set,
                    match_state, user_callback, prune_callback);
    }
    for (const auto& it : supp2) {
      ctx.supp_.insert(it->id());
    }
    LOG(INFO) << "fid = " << frag.fid() << " part supp = " << ctx.supp_.size();
    timer_next("run distributed match");
    InitCandidateSet(ctx.pattern, ctx.index_graph,
                     ctx.distributed_candidate_set);

    std::vector<std::vector<MessageType>> message_container(frag.fnum());
    for (const auto& target_ptr : ctx.distributed_candidate_set[supp_ptr]) {
      if (ctx.supp_.count(target_ptr->id())) {
        continue;
      }
      if (frag.IsOuterVertex(target_ptr->vertex())) continue;
      DistributedMap dis_map;
      DataMatchSet target_matched;
      MatchVertexBelongTo match_vertex_belong_to;
      auto temp_candidate_set = ctx.distributed_candidate_set;
      GUNDAM::_dp_iso::UpdateState(supp_ptr, target_ptr->id(), dis_map,
                                   target_matched);
      match_vertex_belong_to.emplace(supp_ptr, frag.fid());

      GUNDAM::_dp_iso::UpdateCandidateSet<
          Pattern, FragmentGraphWithIndex<const fragment_t>>(
          supp_ptr, target_ptr, temp_candidate_set, dis_map, target_matched);

      size_t result_count = 0;
      auto user_callback = [&supp_ptr, &supp2](auto& match_state) {
        return false;
      };
      auto prune_callback = [](auto& match_state) { return false; };
      _DistributedMatch(ctx.pattern, ctx.index_graph, temp_candidate_set,
                        dis_map, target_matched, match_vertex_belong_to,
                        message_container, result_count, user_callback,
                        prune_callback);
    }
    for (size_t send_fid = 0; send_fid < frag.fnum(); send_fid++) {
      if (send_fid == frag.fid()) {
        continue;
      }
      InArchive send_message;
      for (const auto& message : message_container[send_fid]) {
        send_message << message;
      }
      messages.SendRawMsgByFid(send_fid, std::move(send_message));
    }
    ctx.round_num = 0;
    messages.ForceContinue();
#ifdef PROFILING
    ctx.exec_time -= GetCurrentTime();
#endif

#ifdef PROFILING
    ctx.exec_time += GetCurrentTime();
    ctx.postprocess_time -= GetCurrentTime();
#endif
    // messages.ForceContinue();

#ifdef PROFILING
    ctx.postprocess_time += GetCurrentTime();
#endif
  }

  /**
   * @brief Incremental evaluation for DPISO.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void IncEval(const fragment_t& frag, context_t& ctx,
               message_manager_t& messages) {
    // auto inner_vertices = frag.InnerVertices();

    auto& channels = messages.Channels();
    using VertexIDType = typename fragment_t::vid_t;

    using MessageReciever = std::vector<MessageType>;
    ctx.round_num++;
    LOG(INFO) << "IncEval Round" << ctx.round_num;
    if (ctx.round_num <= ctx.pattern.CountVertex()) {
      MessageReciever message_receiver;
      messages.ParallelProcess<MessageType>(
          1, [&message_receiver](int tid, MessageType msg) {
            message_receiver.push_back(std::move(msg));
          });
      LOG(INFO) << "fid = " << frag.fid()
                << " message num = " << message_receiver.size();
      auto supp_ptr = ctx.pattern.FindVertex(1);
      std::vector<std::vector<MessageType>> message_container(frag.fnum());
      for (const auto& msg : message_receiver) {
        auto [match_state, match_vertex_belong_to] =
            MessageToMap<Pattern, FragmentGraphWithIndex<const fragment_t>>(
                ctx.pattern, msg);
        DataMatchSet target_matched;
        if (ctx.supp_.count(match_state[supp_ptr])) {
          continue;
        }
        for (const auto& it : match_state) {
          target_matched.insert(it.second);
        }
        size_t result_count = 0;
        auto user_callback = [&supp_ptr, &ctx](auto& match_state) {
          ctx.supp_.insert(match_state[supp_ptr]);
          return false;
        };
        auto prune_callback = [](auto& match_state) { return false; };
        _DistributedMatch(
            ctx.pattern, ctx.index_graph, ctx.distributed_candidate_set,
            match_state, target_matched, match_vertex_belong_to,
            message_container, result_count, user_callback, prune_callback);
      }
      if (ctx.round_num < ctx.pattern.CountVertex()) {
        for (size_t send_fid = 0; send_fid < frag.fnum(); send_fid++) {
          if (send_fid == frag.fid()) {
            continue;
          }
          InArchive send_message;
          for (const auto& message : message_container[send_fid]) {
            send_message << message;
          }
          messages.SendRawMsgByFid(send_fid, std::move(send_message));
        }
      }
      messages.ForceContinue();
      if (ctx.round_num == ctx.pattern.CountVertex()) {
        // send result to process 0
        InArchive send_message;
        send_message << ctx.supp_;
        messages.SendRawMsgByFid(0, std::move(send_message));
      }
    } else if (ctx.round_num == ctx.pattern.CountVertex() + 1) {
      messages.ParallelProcess<decltype(ctx.supp_)>(
          1, [&ctx](int tid, decltype(ctx.supp_)& msg) {
            for (const auto& it : msg) {
              ctx.supp_.insert(it);
            }
          });
    }
#ifdef PROFILING
    ctx.preprocess_time -= GetCurrentTime();
#endif

#ifdef PROFILING
    ctx.preprocess_time += GetCurrentTime();
    ctx.exec_time -= GetCurrentTime();
#endif

#ifdef PROFILING
    ctx.exec_time += GetCurrentTime();
    ctx.postprocess_time -= GetCurrentTime();
#endif
#ifdef PROFILING
    ctx.postprocess_time += GetCurrentTime();
#endif
  }
};
}  // namespace grape
#endif