#ifndef EXAMPLES_ANALYTICAL_APPS_RULE_MATCH_RULE_MATCH_H_
#define EXAMPLES_ANALYTICAL_APPS_RULE_MATCH_RULE_MATCH_H_

#include <grape/grape.h>
#include <omp.h>

#include <functional>

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

#include "rule_match/rule_match_context.h"

#include "gar/csv_gar.h"
#include "gar/gar.h"
#include "gar/gar_supp.h"
#include "gcr/gcr_supp.h"

#include "util/log.h"
#include "util/load_rule.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

namespace grape {

template <typename FRAG_T>
class RuleMatch : public ParallelAppBase<FRAG_T, RuleMatchContext<FRAG_T>>,
                  public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
  static constexpr std::string_view kWorkerRuleMatchPosfix = "@worker";

  static constexpr std::string_view kRulePosfix    = RuleMatchContext<FRAG_T>::kRulePosfix;
  static constexpr std::string_view kRuleSetPosfix = RuleMatchContext<FRAG_T>::kRuleSetPosfix;

  static constexpr std::string_view kSuppPosfix    = RuleMatchContext<FRAG_T>::kSuppPosfix;
  static constexpr std::string_view kSuppSetPosfix = RuleMatchContext<FRAG_T>::kSuppSetPosfix;
                  
  using    VertexIDType = typename RuleMatchContext<FRAG_T>::   VertexIDType;
  using VertexLabelType = typename RuleMatchContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType = typename RuleMatchContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType = typename RuleMatchContext<FRAG_T>::  EdgeLabelType;
  using    DataGraph    = typename RuleMatchContext<FRAG_T>::DataGraph;
  using      Pattern    = typename RuleMatchContext<FRAG_T>::  Pattern;
  using     RuleType    = typename RuleMatchContext<FRAG_T>::GarType;

  // the fragment calculate confidence and deliever the file name of the gars
  // to other workers
  static constexpr int kCoordinatorFragID = 0;

  // from ordinary workers to kCoordinatorFragID
  //    info the kCoordinatorFragID how many process each worker have
  static constexpr std::string_view kInfoProcessPrefix = "#info_process";
  // from kCoordinatorFragID to ordinary worker
  //    deliever the file name of the gars that they needs to process
  static constexpr std::string_view kDeliverGarFileNamePrefix = "#deliver_gar_file_name";
  // from ordinary worker to kCoordinatorFragID
  //    inform that all support has been calcluated, it would be ok to calcluate the
  //    confidence
  static constexpr std::string_view kInfoFinishPrefix = "#info_finish";

  // from kTimmerFragID to ordinary workers
  //    info the ordinary workers begin to run the matching
  static constexpr std::string_view kBeginMatchPrefix = "#begin_match";

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(RuleMatch<FRAG_T>, 
                          RuleMatchContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for RuleMatch.
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

    std::cout<<"yaml file: "<<ctx.yaml_file_<<std::endl;

    YAML::Node config = YAML::LoadFile(ctx.yaml_file_);
    
    // ##################################
    // ##   load path of data graph    ##
    // ##################################
    if (!config["DataGraphPath"]) {
      util::Error("cannot get data graph path");
      return;
    }
    YAML::Node data_graph_path_config = config["DataGraphPath"];

    if (!data_graph_path_config["VFile"]) {
      util::Error("does not specify v-file for data graph");
      return;
    }
    const std::string kDataGraphVFile
      = data_graph_path_config["VFile"].as<std::string>();

    if (!data_graph_path_config["EFile"]) {
      util::Error("does not specify e-file for data graph");
      return;
    }
    const std::string kDataGraphEFile
      = data_graph_path_config["EFile"].as<std::string>();

    if (GUNDAM::ReadCSVGraph(ctx.data_graph_, kDataGraphVFile,
                                              kDataGraphEFile) < 0) {
      util::Error("load data graph fail!");
      return;
    }

    if (!config["RuleType"]) {
      util::Error("does not specified rule type");
      return;
    }
    ctx.rule_type_ = config["RuleType"].as<std::string>();
    
    if (ctx.rule_type_ != "gcr"
     && ctx.rule_type_ != "gar") {
      util::Error("unsupported rule type: " + ctx.rule_type_); 
      return;
    }

    if (config["LogFile"]) {
      /* ######################## *
       * ##  for detailed log  ## *
       * ######################## */
      assert(!ctx.log_file_.is_open());
      const std::string kLogFileName = config["LogFile"].as<std::string>();
      ctx.log_file_.open(kLogFileName, std::ios::app);
      if (!ctx.log_file_.is_open()) {
        util::Error("open log file failed: " + kLogFileName);
        return;
      }
      if (ctx.fid_ == kCoordinatorFragID) {
        ctx.log_file_ << "log of: " << ctx.yaml_file_ << std::endl;
      }
    }

    if (!config["RulePath"]) {
      util::Error("cannot get rule path");
      return;
    }
    YAML::Node rule_path_config = config["RulePath"];

    std::vector<std::pair<RuleType, std::string>> rule_set;

    if (rule_path_config.IsSequence()) {
      for (int i = 0; i < rule_path_config.size(); i++) {
        auto temp_rule_set = util::LoadRule<Pattern, DataGraph>(rule_path_config[i]);
        std::move(temp_rule_set.begin(), 
                  temp_rule_set.end(), std::back_inserter(rule_set));
      }
    }
    else {
      rule_set = util::LoadRule<Pattern, DataGraph>(rule_path_config);
    }

    util::Info("rule loaded, number: " + std::to_string(rule_set.size()));

    // ##############################################
    // ##   allocate rules to different workers    ##
    // ##############################################

    for (size_t rule_idx = 0; 
                rule_idx < rule_set.size(); 
                rule_idx++) {
      std::cout << "ctx.frag_num_: " << ctx.frag_num_ << std::endl;
      std::cout << "ctx.fid_     : " << ctx.fid_      << std::endl;
      if (rule_idx % ctx.frag_num_ != ctx.fid_) {
        continue;
      }
      ctx.AddRule(std::move(rule_set[rule_idx].first ),
                  std::move(rule_set[rule_idx].second));
    }

    std::cout << "ctx.RuleSetSize(): " << ctx.RuleSetSize() << std::endl;

    std::string msg(kInfoProcessPrefix);
    msg += " " + std::to_string(ctx.fid_)
         + " " + std::to_string(omp_get_num_procs());
    auto& channel_0 = messages.Channels()[0];
    channel_0.SendToFragment(kCoordinatorFragID, msg);

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
   * @brief Incremental evaluation for RuleMatch.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void IncEval(const fragment_t& frag, 
                      context_t& ctx,
              message_manager_t& messages) {

    std::vector<std::vector<std::string>> gar_file_set;

    bool receive_info_process = false,
         receive_begin_match  = false, 
         receive_info_finish  = false, 
         receive_message      = false;

    std::vector<std::tuple<std::string, // rule_name
                           size_t,      //  x_supp
                           size_t>      // xy_supp
               > result;

    messages.ParallelProcess<std::string>(
        // thread_num(),
        1, [&ctx, 
            &result,
            &receive_info_process,
            &receive_begin_match,
            &receive_info_finish, 
            &receive_message](int tid, std::string msg) {
          std::cout << "fid: " << ctx.fid_ << " receive message: " << msg
                    << std::endl;
          auto res_info_process 
            = std::mismatch(kInfoProcessPrefix.begin(),
                            kInfoProcessPrefix. end (), msg.begin());
          auto res_begin_match
            = std::mismatch(kBeginMatchPrefix.begin(),
                            kBeginMatchPrefix. end (), msg.begin());
          auto res_info_finish 
            = std::mismatch(kInfoFinishPrefix.begin(),
                            kInfoFinishPrefix. end (), msg.begin());
          receive_message = true;
          if (res_info_process.first == kInfoProcessPrefix.end()){
            msg = msg.substr(kInfoProcessPrefix.size());
            // kInfoProcessPrefix is the prefix of msg.
            receive_info_process = true;
            std::string fid_str, proc_num_str;
            std::stringstream ss;
            ss << msg;
            ss >> fid_str;
            ss >> proc_num_str;
          } else if (res_begin_match.first == kBeginMatchPrefix.end()) {
            receive_begin_match = true;
          } else if (res_info_finish.first == kInfoFinishPrefix.end()) {
            msg = msg.substr(kInfoFinishPrefix.size());
            // kInfoProcessPrefix is the prefix of msg.
            receive_info_finish = true;
            std::vector<std::string> info = GUNDAM::util::split(msg, '#');
            std::cout << "info.size(): " << info.size() << std::endl;
            assert((info.size() - 1) % 3 == 0);
            result.reserve(result.size() + info.size() / 3);
            for (size_t i = 1; i < info.size() ; i += 3) {
              result.emplace_back(info[i],
                                  GUNDAM::StringToDataType<size_t>(info[i + 1]),
                                  GUNDAM::StringToDataType<size_t>(info[i + 2]));
            }
          } else {
            // unknown message type
            assert(false);
          }
        });

    if (!receive_message) {
      return;
    }

    if (receive_info_process) {
      timer_next("run rule match");
      auto& channel_0 = messages.Channels()[0];
      for (int dst_fid = 0; dst_fid < ctx.frag_num_; dst_fid++) {
        std::string msg(kBeginMatchPrefix);
        channel_0.SendToFragment(dst_fid, std::move(msg));
      }
      return;
    }

    if (receive_info_finish) {
      if (!ctx.log_file_.is_open()) {
        return;
      }
      ctx.log_file_ << "runtime: " << timer_now() << std::endl;
      ctx.log_file_ << "rule_name" << "\t" <<  "x_supp"
                                   << "\t" << "xy_supp" << std::endl;
      for (const auto& [rule_name, x_supp, xy_supp] : result) {
        ctx.log_file_ << rule_name << "\t" <<  x_supp
                                   << "\t" << xy_supp << std::endl;
      }
      return;
    }

    std::string return_info;

    omp_lock_t return_info_lock;
    omp_init_lock(&return_info_lock);

    #pragma omp parallel for schedule(dynamic) 
    for (size_t rule_idx = 0; 
                rule_idx < ctx.RuleSetSize();
                rule_idx++) {

            auto& rule      = ctx.GetRule    (rule_idx);
      const auto& rule_name = ctx.GetRuleName(rule_idx);

      size_t x_supp = 0,
            xy_supp = 0;
    
      if (ctx.rule_type_ == "gcr") {
        std::tie(x_supp, xy_supp) = gcr::GcrSupp(rule, ctx.data_graph_);
      }
      else if (ctx.rule_type_ == "gar") {
        std::tie(x_supp, xy_supp) = gar::GarSupp(rule, ctx.data_graph_);
      }
      else {
        assert(false);
      }
      omp_set_lock(&return_info_lock);
      return_info = std::move(return_info) 
            + "#" + rule_name
            + "#" + std::to_string( x_supp)
            + "#" + std::to_string(xy_supp);
      omp_unset_lock(&return_info_lock);
    }

    std::string msg(kInfoFinishPrefix);
    msg = std::move(msg) 
        + std::move(return_info);
    auto& channel_0 = messages.Channels()[0];
    channel_0.SendToFragment(kCoordinatorFragID, msg);

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

#endif  // EXAMPLES_ANALYTICAL_APPS_RULE_MATCH_RULE_MATCH_H_