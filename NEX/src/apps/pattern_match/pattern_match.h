#ifndef EXAMPLES_ANALYTICAL_APPS_PATTERN_MATCH_PATTERN_MATCH_H_
#define EXAMPLES_ANALYTICAL_APPS_PATTERN_MATCH_PATTERN_MATCH_H_

#include <grape/grape.h>
#include <omp.h>

#include <functional>

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "pattern_match/pattern_match_context.h"

#include "gundam/io/csvgraph.h"
#include "gundam/algorithm/match_using_match.h"

#include "util/yaml_config.h"
#include "util/log.h"

#include "../timer.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

namespace grape {

template <typename FRAG_T>
class PatternMatch : public ParallelAppBase<FRAG_T, PatternMatchContext<FRAG_T>>,
                 public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
  static constexpr std::string_view kWorkerPatternMatchPosfix = "@worker";

  static constexpr std::string_view kRulePosfix    = PatternMatchContext<FRAG_T>::kRulePosfix;
  static constexpr std::string_view kRuleSetPosfix = PatternMatchContext<FRAG_T>::kRuleSetPosfix;

  static constexpr std::string_view kSuppPosfix    = PatternMatchContext<FRAG_T>::kSuppPosfix;
  static constexpr std::string_view kSuppSetPosfix = PatternMatchContext<FRAG_T>::kSuppSetPosfix;
                  
  using    VertexIDType = typename PatternMatchContext<FRAG_T>::   VertexIDType;
  using VertexLabelType = typename PatternMatchContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType = typename PatternMatchContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType = typename PatternMatchContext<FRAG_T>::  EdgeLabelType;
  using    DataGraph    = typename PatternMatchContext<FRAG_T>::DataGraph;
  using      Pattern    = typename PatternMatchContext<FRAG_T>::  Pattern;

  using   PatternVertexHandle = typename GUNDAM::VertexHandle<  Pattern>::type;
  using DataGraphVertexHandle = typename GUNDAM::VertexHandle<DataGraph>::type;

  using CandidateSet = typename PatternMatchContext<FRAG_T>::CandidateSet;

  static constexpr auto compare_by_vertex_id 
  =[](const DataGraphVertexHandle& vertex_handle_0,
      const DataGraphVertexHandle& vertex_handle_1) {
    return vertex_handle_0->id() 
         < vertex_handle_1->id();
  };

  // the fragment calculate confidence and deliever the file name of the patterns
  // to other workers
  static constexpr int kTimmerFragID = 0;

  // from ordinary workers to kTimmerFragID
  //    info the kTimmerFragID how many process each worker have
  static constexpr std::string_view kInfoProcessPrefix = "#info_process";
  // from ordinary workers to kTimmerFragID
  //    inform that all support has been calcluated, it would be ok to calcluate the
  //    confidence
  static constexpr std::string_view kInfoFinishPrefix = "#info_finish";

  // ###########################################
  // ##   protocol without workload balance   ##
  // ###########################################
  // from kTimmerFragID to ordinary workers
  //    info the ordinary workers begin to run the matching
  static constexpr std::string_view kBeginMatchPrefix = "#begin_match";

  // ########################################
  // ##   protocol with workload balance   ##
  // ########################################
  // from kTimmerFragID to ordinary workers
  //    alloc pivot set to each worker
  static constexpr std::string_view kAllocPivotSetPrefix = "#alloc_pivot_set";
  // from ordinary workers to kTimmerFragID
  //    info how many pivots are processed
  static constexpr std::string_view kInfoStatusPrefix = "#info_status_set";

  // #################################
  // ##   protocol with multi pattern   ##
  // #################################
  // from kTimmerFragID to ordinary workers
  //    info them to reset parameter
  static constexpr std::string_view kResetParameterPrefix = "#reset_parameter";

  static constexpr size_t kInitialPivotSetNum = 5;

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(PatternMatch<FRAG_T>, 
                          PatternMatchContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for PatternMatch.
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
    timer_next("load graph");

    util::Info("yaml file: " + ctx.yaml_file_);

    YAML::Node config = YAML::LoadFile(ctx.yaml_file_);

    if (!config["DataGraphPath"]) {
      util::Error("cannot get data graph path");
      return;
    }
    YAML::Node data_graph_path_config = config["DataGraphPath"];

    if (!data_graph_path_config["VFile"]) {
      util::Error("cannot get data graph v file");
      return;
    }
    const std::string kDataGraphPathVFile 
          = data_graph_path_config["VFile"].as<std::string>();

    if (!data_graph_path_config["EFile"]) {
      util::Error("cannot get data graph e file");
      return;
    }
    const std::string kDataGraphPathEFile 
          = data_graph_path_config["EFile"].as<std::string>();

    ctx.is_isomorphism_ = true;
    if (config["MatchSemantics"]) {
      if (config["MatchSemantics"].as<std::string>() == "iso"
       || config["MatchSemantics"].as<std::string>() == "isomorphism") {
        ctx.is_isomorphism_ = true;
      }
      else if (config["MatchSemantics"].as<std::string>() == "homo"
            || config["MatchSemantics"].as<std::string>() == "homomorphism") {
        ctx.is_isomorphism_ = false;
      }
      else {
        util::Error("illegal MatchSemantics: " 
                   + config["MatchSemantics"].as<std::string>());
        return;
      }
    }

    if (config["TimeLogFile"]) {
      assert(!ctx.time_log_file_.is_open());
      const std::string kTimeLogFileName = config["TimeLogFile"].as<std::string>();
      if (ctx.fid_ == kTimmerFragID) {
        ctx.time_log_file_.open(kTimeLogFileName, std::ios::app);
        ctx.time_log_file_ << "time log of: " << ctx.yaml_file_ << std::endl;
        if (!ctx.time_log_file_.is_open()) {
          util::Error("open time log file failed: " 
                     + kTimeLogFileName);
          return;
        }
      }
    }

    if (!config["PatternPath"]) {
      util::Error("cannot get pattern path");
      return;
    }
    YAML::Node pattern_path_config = config["PatternPath"];

    if (pattern_path_config.IsSequence()) {
      ctx.pattern_path_set_.reserve(pattern_path_config.size());
      for (int i = 0; i < pattern_path_config.size(); i++) {
        auto [ pattern_path_string, 
               pattern_path_ret ] = util::GetPatternPathInfo(pattern_path_config[i]);
        if (!pattern_path_ret) {
          util::Error("load pattern path error!");
          return;
        }
        if (pattern_path_config[i]["MatchFile"]) {
          if (ctx.frag_num_ > 1) {
            util::Error("only support export match result with single worker");
            return;
          }
          const std::string kMatchFile = pattern_path_config[i]["MatchFile"].as<std::string>();
          ctx.pattern_path_set_.emplace_back(pattern_path_string, kMatchFile);
          continue;
        }
        ctx.pattern_path_set_.emplace_back(pattern_path_string, "");
      }
    }
    else {
      auto [ pattern_path_string, 
             pattern_path_ret ] = util::GetPatternPathInfo(pattern_path_config);
      if (!pattern_path_ret) {
        util::Error("load pattern path error!");
        return;
      }
      if (pattern_path_config["MatchFile"]) {
        if (ctx.frag_num_ > 1) {
          util::Error("only support export match result with single worker");
          return;
        }
        const std::string kMatchFile = pattern_path_config["MatchFile"].as<std::string>();
        ctx.pattern_path_set_.emplace_back(pattern_path_string, kMatchFile);
      }
      else {
        ctx.pattern_path_set_.emplace_back(pattern_path_string, "");
      }
    }

    if (ctx.pattern_path_set_.size() == 0) {
      util::Error("empty pattern set");
      return;
    }
    
    ctx.pattern_set_.reserve(ctx.pattern_path_set_.size());
    assert(ctx.pattern_set_.empty());

    for (const auto& [kPatternName, kMatchFile] : ctx.pattern_path_set_) {

        const auto& [kPatternPathVFile, 
                     kPatternPathEFile, 
                          kPivotId] = kPatternName;

      Pattern pattern;

      auto read_pattern_res = GUNDAM::ReadCSVGraph(pattern, 
                                                  kPatternPathVFile, 
                                                  kPatternPathEFile);

      if (read_pattern_res < 0) {
        util::Error("read pattern error! v file: " + kPatternPathVFile
                                     + " e file: " + kPatternPathEFile);
        return;
      }

      ctx.pattern_set_.emplace_back(std::move(pattern), PatternVertexHandle());

      assert(!ctx.pattern_set_.back().second);

      auto& pattern_ref = ctx.pattern_set_.back().first;

      if (kPivotId != "") {
        const VertexIDType kPatternPivotId = GUNDAM::StringToDataType<VertexIDType>(kPivotId);

        if (!pattern_ref.FindVertex(kPatternPivotId)) {
          // cannot find the pivot vertex in rhs pattern
          std::string pattern_vertex_set;
          for (auto vertex_it = pattern_ref.VertexBegin();
                   !vertex_it.IsDone();
                    vertex_it++) {
            pattern_vertex_set += " " + GUNDAM::ToString(vertex_it->id());
          }
          util::Error("cannot find the pivot vertex with id: " + GUNDAM::ToString(kPatternPivotId)
                      + " in pattern contains of vertex: " + pattern_vertex_set);
          return;
        }
        // need to find it from graph pattern again
        ctx.pattern_set_.back().second 
          = pattern_ref.FindVertex(kPatternPivotId);
      }
      else {
        // randomly select one from rhs_pattern
        std::vector<VertexIDType> pattern_vertex_id_set;
        for (auto vertex_it = pattern_ref.VertexBegin();
                 !vertex_it.IsDone();
                  vertex_it++) {
          pattern_vertex_id_set.emplace_back(vertex_it->id());
        }
        std::sort(pattern_vertex_id_set.begin(),
                  pattern_vertex_id_set.end());
        assert(!pattern_vertex_id_set.empty());

        ctx.pattern_set_.back().second 
          = pattern_ref.FindVertex(*pattern_vertex_id_set.begin());
      }
      assert(ctx.pattern_set_.back().second);
    }

    #ifndef NDEBUG
    for (const auto& [pattern, pivot] : ctx.pattern_set_) {
      assert(pivot);
      assert(pivot == pattern.FindVertex(pivot->id()));
    }
    #endif // NDEBUG

    if (GUNDAM::ReadCSVGraph(ctx.data_graph_, 
                            kDataGraphPathVFile,
                            kDataGraphPathEFile) < 0) {
      util::Error("load data graph failed!");
      return;
    }
     
    ctx.is_incremental_ = false;
    std::vector<VertexIDType> delta_vertex_id_set;
    if (data_graph_path_config["DeltaVFile"]) {
      ctx.is_incremental_ = true;
      const std::string kDataGraphDeltaVFile
        = data_graph_path_config["DeltaVFile"].as<std::string>();

      if (ctx.frag_num_ > 1) {
        util::Error("does not support distributed incremental match");
        return;
      }

      try {
        std::ifstream delta_vertex_file(kDataGraphDeltaVFile);
        
        if (!delta_vertex_file.good()) {
          util::Error("delta vertex file: "
                    + kDataGraphDeltaVFile + " is not good!");
          return;
        }

        bool has_found_empty_line = false;
        while (delta_vertex_file) {
          std::string s;
          if (!std::getline( delta_vertex_file, s )) 
            break;
          if (std::all_of(s.begin(),s.end(), 
                          [](char c){ 
                            return std::isspace(static_cast<unsigned char>(c));
                          })) {
            // allow empty lines at the last
            has_found_empty_line = true;
            continue;
          }
          util::Debug(s);
          if (has_found_empty_line) {
            util::Error("has an empty line(s) but not at the last!");
            return;
          }
          const VertexIDType kVertexId
              = GUNDAM::StringToDataType<VertexIDType>(s);
          delta_vertex_id_set.emplace_back(kVertexId);
        }
        delta_vertex_file.close();
      }
      catch (std::exception& e) {
        util::Error("read delta vertex file: " 
                  + kDataGraphDeltaVFile 
                  + " illegal! : "
                  + e.what());
        return;
      }

      if (delta_vertex_id_set.empty()) {
        util::Error("empty delta vertex set");
        return;
      }
    }

    bool unbalanced_work_load = false;
    std::vector<size_t> worker_set;
    if (config[ "WorkloadUnbalanceRatio" ]) {
      const int kWorkloadUnbalanceRatio
       = config["WorkloadUnbalanceRatio"].as<int>();
      if (kWorkloadUnbalanceRatio <= 0) {
        util::Error("illegal WorkloadUnbalanceRatio: "
           + std::to_string(kWorkloadUnbalanceRatio));
        return;
      }
      if (ctx.frag_num_ == 1) {
        util::Error("needs to be more than one worker!");
        return;
      }
      unbalanced_work_load = true;
      worker_set.reserve(kWorkloadUnbalanceRatio + ctx.frag_num_ - 1);
      for (int worker_id = 0; worker_id < ctx.frag_num_; worker_id++) {
        if (worker_id == kTimmerFragID) {
          for (int i = 0; i < kWorkloadUnbalanceRatio; i++) {
            worker_set.emplace_back(worker_id);
          }
          continue;
        }
        worker_set.emplace_back(worker_id);
      }
      assert(worker_set.size() == kWorkloadUnbalanceRatio + ctx.frag_num_ - 1);
    }

    ctx.work_load_balance_ = false;
    if (config["WorkloadBalance"]) {
      if (unbalanced_work_load) {
        util::Error("specified both WorkloadBalance and WorkloadUnbalanceRatio");
        return;
      }
      if (ctx.is_incremental_) {
        util::Error("does not support workload balance for incremental matching");
        return;
      }
      YAML::Node workload_balance_config = config["WorkloadBalance"]; 

      if (!workload_balance_config["BalancePeriod"]) {
        util::Error("does not specify balance period");
        return;
      }
      ctx.balance_period_ms_ = workload_balance_config["BalancePeriod"].as<int32_t>(); 
      
      if (ctx.balance_period_ms_ <= 0 ) {
        util::Error("Illegal BalancePeriod: " 
                  + std::to_string(ctx.balance_period_ms_)
                  + " ms, needs > 0");
        return;
      }
      // in order not to send all vertex id to each worker and let they
      // find corresponding handle in data graph again, each holds the 
      // set of candidate for pivot

      if (!workload_balance_config["PivotedCandidateBalanceSize"]) {
        util::Error("does not specify pivoted candidate size to balance");
        return;
      }
      ctx.pivoted_candidate_balance_size_ 
      = workload_balance_config["PivotedCandidateBalanceSize"].as<int>();
      if (ctx.pivoted_candidate_balance_size_ <= 0) {
        util::Error("Illegal PivotedCandidateBalanceSize: " 
                  + std::to_string(ctx.pivoted_candidate_balance_size_)
                  + ", needs > 0");
        return;
      }
      ctx.work_load_balance_ = true;
    }

    ctx.supp_limit_ = -1;
    if (config["SuppLimit"]) {
      ctx.supp_limit_ = config["SuppLimit"].as<int64_t>();
      util::Info("SuppLimit: " + std::to_string(ctx.supp_limit_));
    }

    ctx.time_limit_ = -1.0;
    if (config["TimeLimit"]) {
      ctx.time_limit_ = config["TimeLimit"].as<double>();
      util::Info("TimeLimit: " + std::to_string(ctx.time_limit_) + "s");
    }

    ctx.time_limit_per_supp_ = -1.0;
    if (config["TimeLimitPerSupp"]) {
      ctx.time_limit_per_supp_ = config["TimeLimitPerSupp"].as<double>();
      util::Info("TimeLimitPerSupp: " + std::to_string(ctx.time_limit_per_supp_) + "s");
    }

    if (config["Partition"]) {
      YAML::Node partition_config = config["Partition"];
      if (partition_config.size() != ctx.frag_num_) {
        util::Error("partition number miss matches the worker number!");
        return;
      }
      const std::string kVertexIdSetFile = partition_config[(int)ctx.fid_].as<std::string>();
      util::Info("vertex id set file: " + kVertexIdSetFile);
      // :
      //    ToDo: 
      //      allow user to specify graph partition, each worker
      //      loads the vertex id set from kVertexIdSetFile, find the
      //      corresponding vertex handle for each id in the data graph, 
      //      add those corresponding vertex handles into 
      //      ctx.data_graph_pivot_range_
      if (ctx.is_incremental_) {
        // : 
        //  needs to partition the delta vertex set 
        //  according to the partition at the same time
      }
      util::Error("to be completed!");
      return;
    }
    else {
      // does not specify the partition, randomly partition 
      std::vector<DataGraphVertexHandle> vertex_handle_set;
      if (!ctx.is_incremental_) {
        // partition the data graph
        ctx.data_graph_pivot_range_.reserve((ctx.data_graph_.CountVertex() 
                                           / ctx.frag_num_) + 1);
        // based on vertex id, to make sure the partition at
        // each worker are the same
        for (auto vertex_it = ctx.data_graph_.VertexBegin();
                 !vertex_it.IsDone();
                  vertex_it++) {
          vertex_handle_set.emplace_back(vertex_it);
        }
      }
      else {
        // partition the delta vertex set only
        assert(!delta_vertex_id_set.empty());
        for (const auto& vertex_id : delta_vertex_id_set) {
          vertex_handle_set.emplace_back(ctx.data_graph_.FindVertex(vertex_id));
          // should have found
          assert(vertex_handle_set.back());
        }
      }
      std::sort(vertex_handle_set.begin(),
                vertex_handle_set. end (),
                compare_by_vertex_id);

      if (!ctx.work_load_balance_) {
        // without workload balance, divide the pivot candidate here
        for (size_t vertex_handle_idx = 0;
                    vertex_handle_idx < vertex_handle_set.size();
                    vertex_handle_idx++) {
          const size_t kWorkerToAlloc = unbalanced_work_load?
                        worker_set[vertex_handle_idx % worker_set.size()]
                                 : vertex_handle_idx % ctx.frag_num_;
          if (kWorkerToAlloc != ctx.fid_) {
            // would not be considered in this worker
            continue;
          }
          // this vertex is in the fragment of this node
          // should be in the data graph
          assert(vertex_handle_set[vertex_handle_idx]);
          assert(vertex_handle_set[vertex_handle_idx]
              == ctx.data_graph_.FindVertex(vertex_handle_set[vertex_handle_idx]->id()));
          ctx.data_graph_pivot_range_.emplace_back(vertex_handle_set[vertex_handle_idx]);
        }
        #ifndef NDEBUG
        // ctx.data_graph_pivot_range_ should have been 
        // sorted by vertex id
        std::vector<DataGraphVertexHandle> sorted_data_graph_pivot_range
                                            = ctx.data_graph_pivot_range_;
        std::sort(sorted_data_graph_pivot_range.begin(),
                  sorted_data_graph_pivot_range. end (),
                  compare_by_vertex_id);
        assert(sorted_data_graph_pivot_range .size()
              == ctx.data_graph_pivot_range_.size());
        for (size_t vertex_handle_idx = 0;
                    vertex_handle_idx < ctx.data_graph_pivot_range_.size();
                    vertex_handle_idx++) {
          assert(sorted_data_graph_pivot_range [vertex_handle_idx]->id()
                 == ctx.data_graph_pivot_range_[vertex_handle_idx]->id());
          assert(sorted_data_graph_pivot_range [vertex_handle_idx]
                 == ctx.data_graph_pivot_range_[vertex_handle_idx]);
        }
        #endif // NDEBUG
        std::sort(ctx.data_graph_pivot_range_.begin(),
                  ctx.data_graph_pivot_range_.end());

        util::Info("without workload balance, worker: "
                 +  std::to_string(ctx.fid_) 
                 + " pivots vertexes set size: "
                 +  std::to_string(ctx.data_graph_pivot_range_.size()));
      }
      else {
        ctx.sorted_vertex_handle_set_.swap(vertex_handle_set);
        util::Info("with workload balance, worker: "
                 +  std::to_string(ctx.fid_) 
                 + " holds all pivots vertexes with size: "
                 +  std::to_string(ctx.sorted_vertex_handle_set_.size()));
      }

      #ifndef NDEBUG
      if (!ctx.is_incremental_) {
        if (!ctx.work_load_balance_) {
          assert(ctx.data_graph_pivot_range_.size() 
              >= ctx.data_graph_.CountVertex() 
              / ctx.frag_num_);
          assert(ctx.data_graph_pivot_range_.size() 
              <= ctx.data_graph_.CountVertex() 
              / ctx.frag_num_ + 1);
        }
      }
      else {
        assert(ctx.data_graph_pivot_range_.size()   
            >= delta_vertex_id_set.size() 
             / ctx.frag_num_);
        assert(ctx.data_graph_pivot_range_.size() 
            <= delta_vertex_id_set.size() 
            / ctx.frag_num_ + 1);
      }
      #endif // NDEBUG
    }

    std::string msg(kInfoProcessPrefix);
    msg += " " + std::to_string(ctx.fid_)
         + " " + std::to_string(omp_get_num_procs());
    auto& channel_0 = messages.Channels()[0];
    channel_0.SendToFragment(kTimmerFragID, msg);
    
    if (ctx.fid_ == kTimmerFragID
     && ctx.time_log_file_.is_open()) {
      ctx.time_log_file_ << timer_now() << std::endl;
    }

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
   * @brief Incremental evaluation for PatternMatch.
   *
   * @param frag
   * @param ctx
   * @param messageskAllocPivotSetPrefix
   */
  void IncEval(const fragment_t& frag, 
                      context_t& ctx,
              message_manager_t& messages) {

    std::function<bool(const GUNDAM::Match<Pattern, DataGraph>&)>
      prune_callback = [](const GUNDAM::Match<Pattern, DataGraph>& match) -> bool{
      // prune nothing, continue matching
      return false;
    };

    std::function<bool(const GUNDAM::Match<Pattern, DataGraph>&)> 
      match_collect_output_callback 
         = [&ctx](const GUNDAM::Match<Pattern, DataGraph>& match) -> bool {
      ctx.match_set_.AddMatch(match);
      if (ctx.supp_limit_ == -1){
        // x_supp_limit is not specified 
        // continue matching
        return true;
      }
      assert(ctx.match_set_.size() < ctx.supp_limit_);
      if (ctx.match_set_.size() == ctx.supp_limit_){
        // has reached x_supp_limit
        // stop matching
        return false;
      }
      return true;
    };

    bool receive_message      = false,
         receive_info_process = false,
         receive_info_finish  = false,
        // ###########################################
        // ##   protocol without workload balance   ##
        // ###########################################
         receive_begin_match  = false,
        // ########################################
        // ##   protocol with workload balance   ##
        // ########################################
         receive_alloc_pivot_set = false,
         receive_info_status     = false,
        // #################################
        // ##   protocol with multi pattern   ##
        // #################################
         receive_reset_parameter = false;

    size_t received_pivot_vertex_set_idx_set_begin = 0,
           received_pivot_vertex_set_idx_set_end   = 0;

    messages.ParallelProcess<std::string>(
        // thread_num(),
        1, [&ctx, 
            &received_pivot_vertex_set_idx_set_begin, 
            &received_pivot_vertex_set_idx_set_end,
            &receive_message,
            &receive_info_process, 
            &receive_info_finish,
          // ###########################################
          // ##   protocol without workload balance   ##
          // ###########################################
            &receive_begin_match,
          // ########################################
          // ##   protocol with workload balance   ##
          // ########################################
            &receive_alloc_pivot_set,
            &receive_info_status,
          // #################################
          // ##   protocol with multi pattern   ##
          // #################################
            &receive_reset_parameter](int tid, std::string msg) {
          util::Info("fid: "  + std::to_string(ctx.fid_) 
                   + " receive message: " + msg);
          auto res_info_process 
            = std::mismatch(kInfoProcessPrefix.begin(),
                            kInfoProcessPrefix. end (), msg.begin());
          auto res_info_finish 
            = std::mismatch(kInfoFinishPrefix.begin(),
                            kInfoFinishPrefix. end (), msg.begin());
          // ###########################################
          // ##   protocol without workload balance   ##
          // ###########################################
          auto res_begin_match 
            = std::mismatch(kBeginMatchPrefix.begin(),
                            kBeginMatchPrefix. end (), msg.begin());
          // ########################################
          // ##   protocol with workload balance   ##
          // ########################################
          auto res_alloc_pivot_set
            = std::mismatch(kAllocPivotSetPrefix.begin(),
                            kAllocPivotSetPrefix. end (), msg.begin());
          auto res_info_status
            = std::mismatch(kInfoStatusPrefix.begin(),
                            kInfoStatusPrefix. end (), msg.begin());
          // #################################
          // ##   protocol with multi pattern   ##
          // #################################
          auto res_reset_parameter
            = std::mismatch(kResetParameterPrefix.begin(),
                            kResetParameterPrefix. end (), msg.begin());

          receive_message = true;
          if (res_info_finish.first == kInfoFinishPrefix.end()) {
            msg = msg.substr(kInfoFinishPrefix.size());
            // kInfoFinishPrefix is the prefix of msg.
            receive_info_finish = true;

            std::string match_count_str,
                               supp_str;

            std::stringstream ss;
            ss << msg;
            ss >> supp_str;
            
            ctx.final_result_ += std::stoi(supp_str);

            util::Info("## receive_info_finish ##");
          } else if (res_info_process.first == kInfoProcessPrefix.end()) {
            msg = msg.substr(kInfoProcessPrefix.size());
            // kInfoProcessPrefix is the prefix of msg.
            receive_info_process = true;
            util::Info("##  receive_info_process  ##");
          } // ###########################################
            // ##   protocol without workload balance   ##
            // ###########################################
            else if (res_begin_match.first == kBeginMatchPrefix.end()) {
            msg = msg.substr(kBeginMatchPrefix.size());
            // kBeginMatchPrefix is the prefix of msg.
            receive_begin_match = true;
            util::Info("##  receive_begin_match  ##");
          } // ########################################
            // ##   protocol with workload balance   ##
            // ########################################
            else if (res_alloc_pivot_set.first == kAllocPivotSetPrefix.end()) {
            msg = msg.substr(kAllocPivotSetPrefix.size());
            // kAllocPivotSetPrefix is the prefix of msg.
            receive_alloc_pivot_set = true;
            std::stringstream ss;
            std::string begin_str, end_str;
            ss << msg;
            ss >> begin_str;
            ss >>   end_str;
            received_pivot_vertex_set_idx_set_begin = std::stoi(begin_str);
            received_pivot_vertex_set_idx_set_end   = std::stoi(  end_str);
            util::Info("##  receive_alloc_pivot_set  ##");
          } else if (res_info_status.first == kInfoStatusPrefix.end()) {
            msg = msg.substr(kInfoStatusPrefix.size());
            // kInfoStatusPrefix is the prefix of msg.
            receive_info_status = true;

            std::string from_id_str,
                           supp_str,
                       remained_str;

            std::stringstream ss;
            ss << msg;
            ss >> from_id_str;
            ss >> supp_str;
            ss >> remained_str;

            ctx.final_result_ += GUNDAM::StringToDataType<decltype(ctx.final_result_)>(supp_str);

            ctx.remained_pivot_size_[std::stoi(from_id_str)] 
                                   = std::stoi(remained_str);

            util::Info("##  receive_info_status  ##");
          } else if (res_reset_parameter.first == kResetParameterPrefix.end()) {
            msg = msg.substr(kResetParameterPrefix.size());
            // kResetParameterPrefix is the prefix of msg.
            receive_reset_parameter = true;
            util::Info("##  receive_reset_parameter  ##");
          } else {
            // unknown message type
            assert(false);
          }
        });

    if (!receive_message) {
      return;
    }

    if (receive_reset_parameter) {
      if (ctx.CurrentPatternCollectMatchResult()) {
        assert(ctx.CurrentMatchFile() != "");
        std::ofstream rhs_match_file(ctx.CurrentMatchFile());
        ctx.match_set_.Serialize(rhs_match_file);
      }
      ctx.ResetParameter();
      ctx.ToNextPattern();
      if (!ctx.HasPatternToProcess()) {
        if (ctx.fid_ == kTimmerFragID
         && ctx.PatternSetSize() > 1) {
          // multi pattern
          if (ctx.time_log_file_.is_open()) {
            ctx.time_log_file_ << " - " << "total match time" 
                               << ": " << ctx.TotalMatchTime() << " sec" << std::endl;
          }
        }
        return;
      }
      // to begin the next round match
      std::string msg(kInfoProcessPrefix);
      msg += " " + std::to_string(ctx.fid_)
           + " " + std::to_string(omp_get_num_procs());
      auto& channel_0 = messages.Channels()[0];
      channel_0.SendToFragment(kTimmerFragID, msg);
      return;
    }

    if (receive_info_process) {
      // each fragment are loaded, timmer node begin to record
      assert (ctx.fid_ == kTimmerFragID);
      timer_next("distributed pattern match");
      if (!ctx.work_load_balance_) {
        // without work load balance, just send message
        // to each fragment to info them begin matching
        auto& channel_0 = messages.Channels()[0];
        for (int dst_fid = 0; dst_fid < ctx.frag_num_; dst_fid++) {
          std::string msg(kBeginMatchPrefix);
          channel_0.SendToFragment(dst_fid, std::move(msg));
        }
        return;
      }
      // with work load balance, need to alloc pivot set to each
      // worker to each fragment to info them begin matching
    }

    if (receive_info_finish) {
      // match end
      assert(ctx.fid_ == kTimmerFragID);
      // only timmer node record the time log
      ctx.AddMatchTime(timer_now_time());
      if (ctx.time_log_file_.is_open()) {
        ctx.time_log_file_ << ctx.CurrentPatternName() 
                           << "<match>: <" << ctx.final_result_ << ">" 
                           << std::endl;
        ctx.time_log_file_ << timer_now() << std::endl;
      }
      // distributed match end
      util::Output("<match>: <" + std::to_string(ctx.final_result_)
                          + ">");
      // has pattern to process, info all worker to reset the
      // parameter, begin next round
      auto& channel_0 = messages.Channels()[0];
      for (int dst_fid = 0; dst_fid < ctx.frag_num_; dst_fid++) {
        std::string msg(kResetParameterPrefix);
        channel_0.SendToFragment(dst_fid, std::move(msg));
      }
      return;
    }

    if (receive_info_process || receive_info_status) {
      assert(ctx.fid_ == kTimmerFragID);  
      assert(ctx.work_load_balance_);
      // with work load balance, send work load 
      // to each fragment for their processing
      std::vector<std::pair<size_t,  // pivot_vertex_set_idx_set_begin
                            size_t>  // pivot_vertex_set_idx_set_end
                 > pivot_set_for_each_worker(ctx.frag_num_, 
                                             std::pair(0, 0));

      bool all_empty = true;
      for (auto fid = 0; fid < ctx.frag_num_; fid++) {
        util::Debug("fid: " + std::to_string(fid) + " ctx.remained_pivot_size_[fid]: "
                                     + std::to_string(ctx.remained_pivot_size_[fid]));
        
        if (ctx.current_allocated_pivot_set_idx_ 
         >= ctx.sorted_vertex_handle_set_.size()) {
          assert(ctx.current_allocated_pivot_set_idx_ 
              == ctx.sorted_vertex_handle_set_.size());
          // all pivots have been allocated
          if (ctx.remained_pivot_size_[fid] != 0) {
            // still have pivot to process
            all_empty = false;
          }
          continue;
        }

        assert(pivot_set_for_each_worker[fid].first  == 0
            && pivot_set_for_each_worker[fid].second == 0);

        assert(ctx.current_allocated_pivot_set_idx_
             < ctx.sorted_vertex_handle_set_.size());

        pivot_set_for_each_worker[fid].first = ctx.current_allocated_pivot_set_idx_;

        const auto kWorkloadRequires = ctx.pivoted_candidate_balance_size_ 
                                     - ctx.remained_pivot_size_[fid];
        assert(kWorkloadRequires >= 0
            && kWorkloadRequires <= ctx.pivoted_candidate_balance_size_);

        const auto kWorkloadRemained = ctx.sorted_vertex_handle_set_.size()
                                     - ctx.current_allocated_pivot_set_idx_;
        assert(kWorkloadRemained > 0
            && kWorkloadRemained <= ctx.sorted_vertex_handle_set_.size());

        const auto kWorkloadToSend 
                 = kWorkloadRequires < kWorkloadRemained?
                   kWorkloadRequires : kWorkloadRemained;

        pivot_set_for_each_worker[fid].second = pivot_set_for_each_worker[fid].first
                                              + kWorkloadToSend;

        assert(pivot_set_for_each_worker[fid].second > 0
            &&(pivot_set_for_each_worker[fid].second 
             > pivot_set_for_each_worker[fid].first)
            &&(pivot_set_for_each_worker[fid].second <= ctx.sorted_vertex_handle_set_.size()));

        ctx.remained_pivot_size_[fid] += kWorkloadToSend;
        ctx.current_allocated_pivot_set_idx_ += kWorkloadToSend;

        all_empty = false;
      }

      if (all_empty) {
        // all are processed, end the matching
        assert(ctx.fid_ == kTimmerFragID);
        // only timmer node record the time log
        ctx.AddMatchTime(timer_now_time());
        if (ctx.time_log_file_.is_open()) {
          ctx.time_log_file_ << ctx.CurrentPatternName() 
                             << "<match>: <" << ctx.final_result_ << ">"
                             << std::endl;
          ctx.time_log_file_ << timer_now() << std::endl;
        }
        // distributed match end
        util::Output("<match>: <" + std::to_string(ctx.final_result_) + ">");
        // has pattern to process, begin next round
        auto& channel_0 = messages.Channels()[0];
        for (int dst_fid = 0; dst_fid < ctx.frag_num_; dst_fid++) {
          std::string msg(kResetParameterPrefix);
          channel_0.SendToFragment(dst_fid, std::move(msg));
        }
        return;
      }
      auto& channel_0 = messages.Channels()[0];
      for (int dst_fid = 0; dst_fid < ctx.frag_num_; dst_fid++) {
        std::string msg(kAllocPivotSetPrefix);
        msg = std::move(msg) + " " + std::to_string(pivot_set_for_each_worker[dst_fid].first)
                             + " " + std::to_string(pivot_set_for_each_worker[dst_fid].second);
        channel_0.SendToFragment(dst_fid, std::move(msg));
      }
      return;
    }

    assert(( ctx.work_load_balance_ && receive_alloc_pivot_set)
        || (!ctx.work_load_balance_ && receive_begin_match));
    if (!ctx.candidate_set_initialized_) {
      ctx.candidate_set_.clear();

      bool init_success = false;
      if (ctx.is_isomorphism_) {
        init_success = GUNDAM::_dp_iso_using_match::InitCandidateSet<GUNDAM::MatchSemantics::kIsomorphism>(
                              ctx.CurrentPattern(), 
                              ctx.data_graph_, 
                              ctx.candidate_set_);
      }
      else {
        init_success = GUNDAM::_dp_iso_using_match::InitCandidateSet<GUNDAM::MatchSemantics::kHomomorphism>(
                              ctx.CurrentPattern(), 
                              ctx.data_graph_, 
                              ctx.candidate_set_);
      }

      if (!init_success) {
        util::Debug("InitCandidateSet fail!");
        std::string msg(kInfoFinishPrefix);
        auto& channel_0 = messages.Channels()[0];
        channel_0.SendToFragment(kTimmerFragID, msg + " 0"); // supp_x = 0 && supp_y = 0
        return;
      }
      //  if need to workload balance:
      //    refine the entire candidate set at once
      //  if do not need to workload balance:
      //    receive the partition, then refine the candidate set
      if (ctx.work_load_balance_) {
        // with workload balance, refine the entire candidate set at once
        if (!GUNDAM::_dp_iso_using_match::RefineCandidateSet(
                  ctx.CurrentPattern(), 
                  ctx.data_graph_, 
                  ctx.candidate_set_)) {
          util::Debug("RefineCandidateSet fail!");
          std::string msg(kInfoFinishPrefix);
          auto& channel_0 = messages.Channels()[0];
          channel_0.SendToFragment(kTimmerFragID, msg + " 0"); // supp_x = 0 && supp_y = 0
          return;
        }
      }
      ctx.candidate_set_initialized_ = true;
    }

    if (receive_alloc_pivot_set) {
      // #############################
      // ##  receive the pivot set  ##
      // #############################
      assert(ctx.work_load_balance_);
      // add allocated pivot set to this worker

      #ifndef NDEBUG
      const auto kSizeBeforeInsert = ctx.data_graph_pivot_range_.size();
      #endif

      ctx.data_graph_pivot_range_.insert(
      ctx.data_graph_pivot_range_  .end(),
      ctx.sorted_vertex_handle_set_.begin() + received_pivot_vertex_set_idx_set_begin,
      ctx.sorted_vertex_handle_set_.begin() + received_pivot_vertex_set_idx_set_end);

      #ifndef NDEBUG
      if (received_pivot_vertex_set_idx_set_begin 
       == received_pivot_vertex_set_idx_set_end) {
        assert(kSizeBeforeInsert == ctx.data_graph_pivot_range_.size());
      }
      #endif

      // receive all pivot vertexes, needs to sort it 
      std::sort(ctx.data_graph_pivot_range_.begin(),
                ctx.data_graph_pivot_range_.end());
    }

    assert(receive_alloc_pivot_set
        || receive_begin_match);

    util::Debug(" 1 here!");

    auto pivot_candidate_set_it 
         = ctx.candidate_set_.find(ctx.CurrentPivot());
    assert(pivot_candidate_set_it 
          != ctx.candidate_set_.end());
    
    std::vector<DataGraphVertexHandle> selected_pivot_candidate_set;

    util::Debug("pivot_candidate_set_it->second.size(): "
                + std::to_string(pivot_candidate_set_it->second.size()));
    util::Debug("ctx.data_graph_pivot_range_.size(): "
                + std::to_string(ctx.data_graph_pivot_range_.size()));

    std::set_intersection(pivot_candidate_set_it->second.begin(),
                          pivot_candidate_set_it->second. end (),
                          ctx.data_graph_pivot_range_.begin(),
                          ctx.data_graph_pivot_range_. end (),
                          std::back_inserter(
                                selected_pivot_candidate_set));

    util::Debug("selected_pivot_candidate_set.size(): "
                + std::to_string(selected_pivot_candidate_set.size()));

    if (!ctx.work_load_balance_) {
      // add all selected_pivot_candidate_set in it at once
      pivot_candidate_set_it->second.swap(selected_pivot_candidate_set);

      size_t result = 0;

      if (ctx.is_incremental_) {
        if (ctx.is_isomorphism_) {
          result = ctx.CurrentPatternCollectMatchResult()?
                    GUNDAM::IncrementalMatchUsingMatch<
                    GUNDAM::MatchSemantics::kIsomorphism>(ctx.CurrentPattern(),
                                                          ctx.data_graph_,
                                                          ctx.data_graph_pivot_range_,
                                                          match_collect_output_callback,
                                                          ctx.time_limit_)
                  : GUNDAM::IncrementalMatchUsingMatch<
                    GUNDAM::MatchSemantics::kIsomorphism>(ctx.CurrentPattern(),
                                                          ctx.data_graph_,
                                                          ctx.data_graph_pivot_range_,
                                                          ctx.supp_limit_,
                                                          ctx.time_limit_);
        }
        else {
          result = ctx.CurrentPatternCollectMatchResult()?
                    GUNDAM::IncrementalMatchUsingMatch<
                    GUNDAM::MatchSemantics::kHomomorphism>(ctx.CurrentPattern(),
                                                          ctx.data_graph_,
                                                          ctx.data_graph_pivot_range_,
                                                          match_collect_output_callback,
                                                          ctx.time_limit_)
                  : GUNDAM::IncrementalMatchUsingMatch<
                    GUNDAM::MatchSemantics::kHomomorphism>(ctx.CurrentPattern(),
                                                          ctx.data_graph_,
                                                          ctx.data_graph_pivot_range_,
                                                          ctx.supp_limit_,
                                                          ctx.time_limit_);
        }
      }
      else {
        if (GUNDAM::_dp_iso_using_match::RefineCandidateSet(
                ctx.CurrentPattern(), 
                ctx.data_graph_, 
                ctx.candidate_set_)) {
          if (ctx.is_isomorphism_) {
            result = ctx.CurrentPatternCollectMatchResult()?
                    GUNDAM::MatchUsingMatch<
                    GUNDAM::MatchSemantics::kIsomorphism>(ctx.CurrentPattern(), 
                                                          ctx.data_graph_,
                                                          ctx.candidate_set_,
                                                          prune_callback,
                                                          match_collect_output_callback,
                                                          ctx.time_limit_)
                  : GUNDAM::MatchUsingMatch<
                    GUNDAM::MatchSemantics::kIsomorphism>(ctx.CurrentPattern(), 
                                                          ctx.data_graph_,
                                                          ctx.candidate_set_,
                                                          ctx.supp_limit_,
                                                          ctx.time_limit_);
          }
          else {
            result = ctx.CurrentPatternCollectMatchResult()?
                    GUNDAM::MatchUsingMatch<
                    GUNDAM::MatchSemantics::kHomomorphism>(ctx.CurrentPattern(), 
                                                           ctx.data_graph_,
                                                           ctx.candidate_set_,
                                                           prune_callback,
                                                           match_collect_output_callback,
                                                           ctx.time_limit_)
                  : GUNDAM::MatchUsingMatch<
                    GUNDAM::MatchSemantics::kHomomorphism>(ctx.CurrentPattern(), 
                                                           ctx.data_graph_,
                                                           ctx.candidate_set_,
                                                           ctx.supp_limit_,
                                                           ctx.time_limit_);
          }
        }
        else {
          util::Debug("RefineCandidateSet fail!");
        }
      }

      std::string msg(kInfoFinishPrefix);
      auto& channel_0 = messages.Channels()[0];
      channel_0.SendToFragment(kTimmerFragID, msg + " " + std::to_string(result));
      return;
    }

    assert(ctx.work_load_balance_);
    // with workload balance
    // add all selected_pivot_candidate_set one-by-one

    auto kBeginTime = std::chrono::high_resolution_clock::now();

    ctx.data_graph_pivot_range_.swap(selected_pivot_candidate_set);

    std::vector<DataGraphVertexHandle>  temp_pivot_candidate_set;
    pivot_candidate_set_it->second.swap(temp_pivot_candidate_set);
    pivot_candidate_set_it->second.clear();
    pivot_candidate_set_it->second.resize(1);
    assert(pivot_candidate_set_it->second.size() == 1);

    size_t result = 0;

    while (!ctx.data_graph_pivot_range_.empty()) {

      assert(pivot_candidate_set_it->second.size() == 1);

      pivot_candidate_set_it->second[0] = ctx.data_graph_pivot_range_.back();
      
      size_t temp_result = 0;
      if (ctx.is_isomorphism_) {
        temp_result = ctx.CurrentPatternCollectMatchResult()?
                            GUNDAM::MatchUsingMatch<
                            GUNDAM::MatchSemantics::kIsomorphism>(
                              ctx.CurrentPattern(), 
                              ctx.data_graph_,
                              ctx.candidate_set_,
                              prune_callback,
                              match_collect_output_callback,
                              ctx.time_limit_)
                          : GUNDAM::MatchUsingMatch<
                            GUNDAM::MatchSemantics::kIsomorphism>(
                              ctx.CurrentPattern(), 
                              ctx.data_graph_,
                              ctx.candidate_set_,
                              ctx.supp_limit_,
                              ctx.time_limit_);
      }
      else {
        temp_result = ctx.CurrentPatternCollectMatchResult()?
                            GUNDAM::MatchUsingMatch<
                            GUNDAM::MatchSemantics::kHomomorphism>(
                              ctx.CurrentPattern(), 
                              ctx.data_graph_,
                              ctx.candidate_set_,
                              prune_callback,
                              match_collect_output_callback,
                              ctx.time_limit_)
                          : GUNDAM::MatchUsingMatch<
                            GUNDAM::MatchSemantics::kHomomorphism>(
                              ctx.CurrentPattern(), 
                              ctx.data_graph_,
                              ctx.candidate_set_,
                              ctx.supp_limit_,
                              ctx.time_limit_);
      }

      result += temp_result;

      // this pivot is processed
      ctx.data_graph_pivot_range_.pop_back();

      if ((std::chrono::duration_cast<std::chrono::milliseconds>(
           std::chrono::high_resolution_clock::now() - kBeginTime).count()
              > ctx.balance_period_ms_)) {
        break;
      }
    }

    // store back
    pivot_candidate_set_it->second.swap(temp_pivot_candidate_set);
    
    std::string msg(kInfoStatusPrefix);
    auto& channel_0 = messages.Channels()[0];
    channel_0.SendToFragment(kTimmerFragID, msg + " " + std::to_string(ctx.fid_) 
                                                + " " + std::to_string(result) 
                                                + " " + std::to_string(ctx.data_graph_pivot_range_.size()));
  
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

#endif  // EXAMPLES_ANALYTICAL_APPS_PATTERN_MATCH_PATTERN_MATCH_H_