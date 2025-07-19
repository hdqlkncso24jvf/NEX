#ifndef EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_GCR_DISCOVER_H_
#define EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_GCR_DISCOVER_H_

#include <grape/grape.h>
#include <omp.h>

#include <functional>

#include "gundam/tool/operator/merge_graph.h"

#include "gundam/match/match.h"
#include "gundam/match/matchresult.h"

#include "gundam/io/csvgraph.h"

#include "gundam/graph_type/graph.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/large_graph2.h"
#include "gundam/graph_type/small_graph.h"

#include "gundam/util/string.h"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../timer.h"

#include "rule_discover/gcr_discover_context.h"
#include "rule_discover/gcr_discover/protocol.h"
#include "rule_discover/gcr_discover/expand_literal.h"
#include "rule_discover/gcr_discover/receive_info_process.h"
#include "rule_discover/gcr_discover/receive_info_finish.h"
#include "rule_discover/gcr_discover/receive_deliver_predicate.h"
#include "rule_discover/gcr_discover/receive_responds_deliver_predicate.h"
#include "rule_discover/gcr_discover/receive_update_predicate.h"
#include "rule_discover/gcr_discover/receive_begin_next_round.h"

#include "gar/csv_gar.h"
#include "gar/gar.h"
#include "gar/gar_supp.h"
#include "gcr/gcr_supp.h"

#include "util/file.h"
#include "util/log.h"
#include "util/load_rule.h"
#include "util/load_graph.h"
#include "util/load_pattern.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

namespace grape {

using namespace _gcr_discover;

template <typename FRAG_T>
class GcrDiscover : public ParallelAppBase<FRAG_T, GcrDiscoverContext<FRAG_T>>,
                    public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
  using     VertexIDType = typename GcrDiscoverContext<FRAG_T>::    VertexIDType;
  using  VertexLabelType = typename GcrDiscoverContext<FRAG_T>:: VertexLabelType;
  using       EdgeIDType = typename GcrDiscoverContext<FRAG_T>::      EdgeIDType;
  using    EdgeLabelType = typename GcrDiscoverContext<FRAG_T>::   EdgeLabelType;
  using    DataGraphType = typename GcrDiscoverContext<FRAG_T>::   DataGraphType;
  using GraphPatternType = typename GcrDiscoverContext<FRAG_T>::GraphPatternType;
  using         RuleType = typename GcrDiscoverContext<FRAG_T>::         GcrType;

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(GcrDiscover<FRAG_T>, 
                          GcrDiscoverContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for GcrDiscover.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void PEval(const fragment_t& frag, 
                    context_t& ctx,
            message_manager_t& messages) {
    
    messages.InitChannels(thread_num());
    auto begin = timer();
    timer_next("build index");

    util::Info( "yaml file: "
           + ctx.yaml_file_ );

    YAML::Node config = YAML::LoadFile(ctx.yaml_file_);
    
    // ##################################
    // ##   load path of data graph    ##
    // ##################################
    if (!config["DataGraphPath"]) {
      util::Error("cannot get data graph path");
      return;
    }
    auto [graph, load_graph_ret] = util::LoadGraph<DataGraphType>(
                                           config["DataGraphPath"]);
    if (!load_graph_ret) {
      util::Error("load data graph fail!");
      return;
    }
    ctx.data_graph_ = std::move(graph);

    ctx.support_bound_ = 1;
    if (config["SupportBound"]) {
      ctx.support_bound_ = config["SupportBound"].as<decltype(ctx.support_bound_)>();
    }

    ctx.specified_confidence_bound_ = false;
    if (config["ConfidenceBound"]) {
      ctx.specified_confidence_bound_ = true;
      ctx.          confidence_bound_ = config["ConfidenceBound"].as<float>();

      if (ctx.confidence_bound_ < 0.0
       || ctx.confidence_bound_ > 1.0) {
        util::Error("illegal ConfidenceBound: " + std::to_string(ctx.confidence_bound_));
        return;
      }
    }

    if (!config["Round"]) {
      util::Error("does not specified Round!");
    }
    ctx.round_ = config["Round"].as<int>();

    if (!config["PatternPath"]) {
      util::Error("does not specify pattern path for gcr");
      return;
    }

    if (!config["OutputGcrDir"]) {
      util::Error("cannot get output gar dir");
      return;
    }
    ctx.output_gcr_dir_ = config["OutputGcrDir"].as<std::string>();

    util::Mkdir(ctx.output_gcr_dir_);

    if (ctx.fid_ != kCoordinatorID) {
      // send back to kCoordinatorID
      std::string msg(kInfoProcessPrefix);
      msg += " " + std::to_string(ctx.fid_)
          + " " + std::to_string(omp_get_num_procs());
      auto& channel_0 = messages.Channels()[0];
      channel_0.SendToFragment(kCoordinatorID, msg);
      return;
    }

    // process all literals
    auto path_set = util::LoadPatternSet<GraphPatternType>(config["PatternPath"]);

    util::Info("path_set.size(): " + std::to_string(path_set.size()));

    // ################################################
    // ##   load types of literal to be considered   ##
    // ################################################
    if (config["LiteralTypes"]) {
      ctx.restriction_.set_consider_no_literal();
      YAML::Node literal_types_config = config["LiteralTypes"];
      for (int i = 0; i < literal_types_config.size(); i++) {
        const std::string literal_type 
                        = literal_types_config[i].as<std::string>();
        if (!ctx.restriction_.ConsiderLiteral(literal_type)) {
          // illegal literal
          return;
        }
      }
    }

    std::vector<GraphPatternType> pattern_set;
    pattern_set.reserve(path_set.size() * path_set.size()
                      + path_set.size());
    for (const auto& [path_0, path_0_name] : path_set) {
      for (const auto& [path_1, path_1_name] : path_set) {
        std::vector<GraphPatternType> input_path_set = {GraphPatternType(path_0), 
                                                        GraphPatternType(path_1)};
        pattern_set.emplace_back(GUNDAM::MergeGraphSet(input_path_set));
      }
    }

    for (auto& [path, path_name] : path_set) {
      pattern_set.emplace_back(std::move(path));
    }

    const GUNDAM::GraphBasicStatistics<DataGraphType>
      graph_statistics(ctx.data_graph_);

    for (auto& pattern : pattern_set) {
      auto literal_info_set = ExpandLiteral(pattern, graph_statistics, ctx.restriction_);
      for (const auto& literal_info : literal_info_set) {
        ctx.predicate_set_pattern_literal_.emplace_back(pattern, literal_info);
      }
    }

    std::random_shuffle ( ctx.predicate_set_pattern_literal_.begin(), 
                          ctx.predicate_set_pattern_literal_.end() );

    std::string msg(kInfoProcessPrefix);
    msg += " " + std::to_string(ctx.fid_)
         + " " + std::to_string(omp_get_num_procs());
    auto& channel_0 = messages.Channels()[0];
    channel_0.SendToFragment(kCoordinatorID, msg);

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

    return;
  }

  /**
   * @brief Incremental evaluation for GcrDiscover.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void IncEval(const fragment_t& frag, 
                      context_t& ctx,
              message_manager_t& messages) {

    std::vector<std::vector<std::string>> gar_file_set;

    bool receive_info_process               = false,
         receive_deliver_predicate          = false,
         receive_responds_deliver_predicate = false,
         receive_update_predicate           = false,
         receive_info_begin_round           = false, 
         receive_info_finish                = false, 
         receive_message                    = false;

    messages.ParallelProcess<std::string>(
        // thread_num(),
        1, [&ctx, 
            &receive_info_process,
            &receive_deliver_predicate,
            &receive_responds_deliver_predicate,
            &receive_update_predicate,
            &receive_info_begin_round, 
            &receive_info_finish, 
            &receive_message](int tid, std::string msg) {
          std::cout << "fid: " << ctx.fid_ << " receive message: " << msg
                    << std::endl;
          auto res_info_process 
            = std::mismatch(kInfoProcessPrefix.begin(),
                            kInfoProcessPrefix. end (), msg.begin());
          auto res_deliver_predicate 
            = std::mismatch(kDeliverPredicatePrefix.begin(),
                            kDeliverPredicatePrefix. end (), msg.begin());
          auto res_responds_deliver_predicate
            = std::mismatch(kRespondsDeliverPredicatePrefix.begin(),
                            kRespondsDeliverPredicatePrefix. end (), msg.begin());
          auto res_update_predicate
            = std::mismatch(kUpdatePredicatePrefix.begin(),
                            kUpdatePredicatePrefix. end (), msg.begin());
          auto res_info_begin_round 
            = std::mismatch(kInfoBeginPrefix.begin(),
                            kInfoBeginPrefix. end (), msg.begin());
          auto res_info_finish 
            = std::mismatch(kInfoFinishPrefix.begin(),
                            kInfoFinishPrefix. end (), msg.begin());
          receive_message = true;
          if (res_info_process.first == kInfoProcessPrefix.end()){
            assert(ctx.fid_ == kCoordinatorID);
            msg = msg.substr(kInfoProcessPrefix.size());
            // kInfoProcessPrefix is the prefix of msg.
            receive_info_process = true;
            std::string fid_str, proc_num_str;
            std::stringstream ss;
            ss << msg;
            ss >> fid_str;
            ss >> proc_num_str;
            util::Info(" fid: " + fid_str + " proc_num: " + proc_num_str);
            ctx.process_num_.emplace(std::stoi(fid_str), 
                                     std::stoi(proc_num_str));
          } else if (res_deliver_predicate.first == kDeliverPredicatePrefix.end()) {
            msg = msg.substr(kDeliverPredicatePrefix.size());
            // kDeliverPredicatePrefix is the prefix of msg.
            receive_deliver_predicate = true;
            while (!msg.empty()) {
              ctx.predicate_set_pattern_literal_.emplace_back();
              msg >>  ctx.predicate_set_pattern_literal_.back();
            }
            util::Info("predicate set to evaluate: "
                      + std::to_string(ctx.predicate_set_pattern_literal_.size()));
          } else if (res_responds_deliver_predicate.first == kRespondsDeliverPredicatePrefix.end()) {
            assert(ctx.fid_ == kCoordinatorID);
            msg = msg.substr(kRespondsDeliverPredicatePrefix.size());
            // kRespondsDeliverPredicatePrefix is the prefix of msg.
            receive_responds_deliver_predicate = true;
            while (!msg.empty()) {
              assert(ctx.fid_ != kCoordinatorID);
              ctx.predicate_set_pattern_literal_.emplace_back();
              msg >>  ctx.predicate_set_pattern_literal_.back();
            }
            util::Info("total legal predicates: "
                      + std::to_string(ctx.predicate_set_pattern_literal_.size()));
          } else if (res_update_predicate.first == kUpdatePredicatePrefix.end()) {
            msg = msg.substr(kUpdatePredicatePrefix.size());
            // kUpdatePredicatePrefix is the prefix of msg.
            receive_update_predicate = true;
            while (!msg.empty()) {
              assert(ctx.fid_ != kCoordinatorID);
              ctx.predicate_set_pattern_literal_.emplace_back();
              msg >>  ctx.predicate_set_pattern_literal_.back();
            }
            util::Info("total legal predicates: "
                      + std::to_string(ctx.predicate_set_pattern_literal_.size()));
          } else if (res_info_finish.first == kInfoFinishPrefix.end()) {
            msg = msg.substr(kInfoFinishPrefix.size());
            // kInfoProcessPrefix is the prefix of msg.
            receive_info_finish = true;
          } else {
            // unknown message type
            assert(false);
          }
        });

    if (receive_info_process) {
      assert (ctx.fid_ == kCoordinatorID);
      /* ############################################ *
       * ##  deliever the generated predicates to  ## *
       * ##  all workers to evalute                ## *
       * ############################################ */
      auto msg_to_workers = ReceiveInfoProcess(ctx);
      auto& channel_0 = messages.Channels()[0];
      for (size_t worker_id = 0; 
                  worker_id < msg_to_workers.size();
                  worker_id++) {
        channel_0.SendToFragment(worker_id, std::string(kDeliverPredicatePrefix)
                                          + std:: move (msg_to_workers[worker_id]));
      }
      return;
    }

    if (receive_deliver_predicate) {
      /* ################################################### *
       * ##  received the delivered predicates, evaluate  ## *
       * ##  all of them, send the legal ones back to     ## *
       * ##  the coordinator                              ## *
       * ################################################### */
      auto& channel_0 = messages.Channels()[0];
      channel_0.SendToFragment(kCoordinatorID, std::string(kRespondsDeliverPredicatePrefix)
                                                   + ReceiveDeliverPredicate(ctx));
      return;
    }

    if (receive_responds_deliver_predicate) {
      /* ############################################## *
       * ##  collect all legal predicates, and send  ## *
       * ##  them to all workers                     ## *
       * ############################################## */
      assert (ctx.fid_ == kCoordinatorID);
      auto msg = ReceiveRespondsDeliverPredicate(ctx);
      auto& channel_0 = messages.Channels()[0];
      for (size_t dst_fid = 0; dst_fid < ctx.frag_num_; dst_fid++) {
        if (dst_fid == kCoordinatorID) {
          channel_0.SendToFragment(dst_fid, std::string(kUpdatePredicatePrefix));
          continue;
        }
        channel_0.SendToFragment(dst_fid, std::string(kUpdatePredicatePrefix) + msg);
      }
      return;
    }

    if (receive_update_predicate) {
      /* ############################################## *
       * ##  received all legal predicates from the  ## *
       * ##  coordinator, collect all of them,       ## *
       * ##  each worker selects a subset of it to   ## *
       * ##  build a forest                          ## *
       * ############################################## */
      auto& channel_0 = messages.Channels()[0];
      channel_0.SendToFragment(kCoordinatorID, std::string(kInfoFinishPrefix)
                                                   + ReceiveUpdatePredicate(ctx));
      return;
    }

    if (receive_info_finish) {
      /* ########################################## *
       * ##  info all workers to begin the next  ## *
       * ##  round processing                    ## *
       * ########################################## */
      assert (ctx.fid_ == kCoordinatorID);
      auto msg = ReceiveInfoFinish(ctx);
      auto& channel_0 = messages.Channels()[0];
      for (size_t dst_fid = 0; dst_fid < ctx.frag_num_; dst_fid++) {
        channel_0.SendToFragment(dst_fid, std::string(kInfoBeginPrefix) + msg);
      }
      return;
    }

    /* ################################################# *
     * ##  begin the next round matching,             ## *
     * ##  first use all predicates to expand local   ## *
     * ##  forest, then evaluate the generated nodes  ## *
     * ################################################# */
    assert(receive_info_begin_round);
    auto msg = ReceiveBeginNextRound(ctx);

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

    return;
  }
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_GCR_DISCOVER_GCR_DISCOVER_H_