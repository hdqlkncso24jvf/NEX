
#ifndef EXAMPLES_ANALYTICAL_APPS_DPISO_DPISO_H_
#define EXAMPLES_ANALYTICAL_APPS_DPISO_DPISO_H_

#include <grape/grape.h>
#include <omp.h>

#include "../fragment2gundam.h"
#include "../fragment_graph.h"
#include "../fragment_graph_with_index.h"
#include "../timer.h"
#include "dpiso/dpiso_context.h"
#include "grape/serialization/in_archive.h"
#include "grape/serialization/out_archive.h"
#include "gundam/io/csvgraph.h"
#include "gundam/algorithm/dp_iso.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/small_graph.h"

#include "gundam/type_getter/vertex_handle.h"

namespace grape {

template <typename FRAG_T>
class DPISO : public ParallelAppBase<FRAG_T, DPISOContext<FRAG_T>>,
              public ParallelEngine {
 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(DPISO<FRAG_T>, DPISOContext<FRAG_T>, FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  using Pattern = typename context_t::Pattern;
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

  /**
   * @brief Partial evaluation for DPISO.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void PEval(const fragment_t& frag, context_t& ctx,
             message_manager_t& messages) {
    auto inner_vertices = frag.InnerVertices();
    messages.InitChannels(thread_num());
    auto begin = timer();
    std::cout << "inner_vertices size: " << inner_vertices.size() << std::endl;
    timer_next("build index");

    ctx.index_graph.fragment = &frag;
    assert(ctx.index_graph.fragment);
    ctx.index_graph.BuildIndex();

    std::cout << "fragment vertex number: "
              << ctx.index_graph.fragment->GetVerticesNum() << std::endl;
    std::cout << "inner vertex number: "
              << ctx.index_graph.fragment->InnerVertices().size() << std::endl;
    std::cout << "outer vertex number: "
              << ctx.index_graph.fragment->OuterVertices().size() << std::endl;

    timer_next("load pattern");
    auto end = timer();
    GUNDAM::ReadCSVGraph(ctx.pattern, ctx.pattern_v_file_, ctx.pattern_e_file_);
    auto supp_ptr = ctx.pattern.FindVertex(1);
    std::vector<
        typename GUNDAM::VertexHandle<FragmentGraphWithIndex<const fragment_t>>::type>
        supp2;

    std::cout << "pattern vertex size: " << ctx.pattern.CountVertex()
              << std::endl;
    LOG(INFO) << "fid = " << frag.fid() << " supp = " << supp2.size();

    using CandidateSetContainer =
        std::map<typename GUNDAM::VertexHandle<Pattern>::type,
     std::vector<typename GUNDAM::VertexHandle<FragmentGraphWithIndex<const fragment_t>>::type>>;
    timer_next("run dpiso");

    CandidateSetContainer candidate_set;
    GUNDAM::_dp_iso::InitCandidateSet<GUNDAM::MatchSemantics::kIsomorphism>(
        ctx.pattern, ctx.index_graph, candidate_set);
    GUNDAM::_dp_iso::RefineCandidateSet(ctx.pattern, ctx.index_graph,
                                        candidate_set);
    using MatchMap = std::map<
        typename GUNDAM::VertexHandle<Pattern>::type,
        typename GUNDAM::VertexHandle<FragmentGraphWithIndex<const fragment_t>>::type>;
    std::sort(candidate_set[supp_ptr].begin(), candidate_set[supp_ptr].end(),
              [](auto& a, auto& b) { return a->id() < b->id(); });
    omp_lock_t add_lock;
    omp_init_lock(&add_lock);
    auto& supp_ptr_candidate = candidate_set[supp_ptr];
#pragma omp parallel for schedule(dynamic)
    for (fid_t i = frag.fid(); i < supp_ptr_candidate.size();
         i += frag.fnum()) {
      CandidateSetContainer temp_candidate_set = candidate_set;
      MatchMap match_state;
      match_state.emplace(supp_ptr, supp_ptr_candidate[i]);
      auto user_callback = [&supp_ptr, &supp2, &add_lock](auto& match_state) {
        omp_set_lock(&add_lock);
        supp2.push_back(match_state[supp_ptr]);
        omp_unset_lock(&add_lock);
        return false;
      };
      auto prune_callback = [](auto& match_state) { return false; };
      GUNDAM::DPISO(ctx.pattern, ctx.index_graph, temp_candidate_set,
                    match_state, user_callback, prune_callback);
    }

    LOG(INFO) << "fid = " << frag.fid() << " supp = " << supp2.size();
    // timer_next("send message");
    /*
    if (frag.fid() != 0) {
      InArchive send_message;
      for (const auto& vertex_ptr : supp2) {
        send_message << vertex_ptr->id();
      }
      messages.SendRawMsgByFid(0, std::move(send_message));
    } else {
      for (const auto& vertex_ptr : supp2) {
        ctx.supp_.insert(vertex_ptr->id());
      }
    }
    */

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
    /*
    if (frag.fid() == 0) {
      // messages.ParallelProcess<VertexIDType>(
      //   thread_num(),
      //   [&ctx](int tid, VertexIDType msg) { ctx.supp_.insert(msg); });
    }
    */

#ifdef PROFILING
    ctx.preprocess_time -= GetCurrentTime();
#endif
    // std::cout << "IncEval!" << std::endl;
    // ctx.next_modified.parallel_clear(thread_num());

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

#endif  // EXAMPLES_ANALYTICAL_APPS_SSSP_SSSP_H_
