#ifndef EXAMPLES_ANALYTICAL_APPS_PROB_GAR_CHASE_PROB_GAR_CHASE_H_
#define EXAMPLES_ANALYTICAL_APPS_PROB_GAR_CHASE_PROB_GAR_CHASE_H_

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

#include "prob_gar_chase/prob_gar_chase_context.h"

#include "prob_gar/prob_gar.h"
#include "prob_gar/prob_gar_chase.h"
#include "prob_gar/prob_gar_anytime_chase.h"
#include "prob_gar/csv_prob_gar.h"

#include "util/log.h"

#include "../timer.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

namespace grape {

template <typename FRAG_T>
class ProbGarChase : public ParallelAppBase<FRAG_T, ProbGarChaseContext<FRAG_T>>,
                     public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
  inline std::pair<std::tuple<std::string,
                              std::string,
                              std::string,
                              std::string,
                              std::string>,
                   bool> GetProbGarPathInfo(const YAML::Node& prob_gar_path_config) const {
    if (!prob_gar_path_config["VFile"]) {
      util::Error( "cannot get prob gar v file" );
      return std::pair(std::tuple("","","","",""), false);
    }
    const std::string kGarPathVFile 
      = prob_gar_path_config["VFile"].as<std::string>();

    if (!prob_gar_path_config["EFile"]) {
      util::Error( "cannot get prob gar e file" );
      return std::pair(std::tuple("","","","",""), false);
    }
    const std::string kGarPathEFile  
      = prob_gar_path_config["EFile"].as<std::string>();

    if (!prob_gar_path_config["XFile"]) {
      util::Error( "cannot get prob gar x file" );
      return std::pair(std::tuple("","","","",""), false);
    }
    const std::string kGarPathXFile 
      = prob_gar_path_config["XFile"].as<std::string>();

    if (!prob_gar_path_config["YFile"]) {
      util::Error( "cannot get prob gar y file" );
      return std::pair(std::tuple("","","","",""), false);
    }
    const std::string kGarPathYFile  
      = prob_gar_path_config["YFile"].as<std::string>();

    if (!prob_gar_path_config["PFile"]) {
      util::Error( "cannot get prob gar p file" );
      return std::pair(std::tuple("","","","",""), false);
    }
    const std::string kGarPathPFile  
      = prob_gar_path_config["PFile"].as<std::string>();

    return std::pair(std::tuple(kGarPathVFile,
                                kGarPathEFile, 
                                kGarPathXFile,
                                kGarPathYFile,
                                kGarPathPFile), true);
  }

  inline std::vector<
         std::tuple<std::string,
                    std::string,
                    std::string,
                    std::string,
                    std::string>> 
        GetProbGarPathSetInfo(const YAML::Node& prob_gar_path_set_config) const {
    using ProbGarPathSetType = std::vector<
                               std::tuple<std::string,
                                          std::string,
                                          std::string,
                                          std::string,
                                          std::string>>;
    if (prob_gar_path_set_config["ProbGarSet"]) {
      // is a set of prob gar
      ProbGarPathSetType prob_gar_path_set;
      for (int i = 0; i < prob_gar_path_set_config["ProbGarSet"].size(); i++) {
        auto [ prob_gar_path,
               prob_gar_path_ret ] = GetProbGarPathInfo(prob_gar_path_set_config["ProbGarSet"][i]);
        if (!prob_gar_path_ret) {
          // illegal path!
          return ProbGarPathSetType();
        }
        prob_gar_path_set.emplace_back(prob_gar_path);
      }
      return prob_gar_path_set;
    }
    auto [ prob_gar_path,
           prob_gar_path_ret ] = GetProbGarPathInfo(prob_gar_path_set_config);
    if (!prob_gar_path_ret) {
      // illegal path!
      return ProbGarPathSetType();
    }
    return ProbGarPathSetType{prob_gar_path};
  }

  static constexpr std::string_view kWorkerProbGarChasePosfix = "@worker";

  static constexpr std::string_view kRulePosfix    = ProbGarChaseContext<FRAG_T>::kRulePosfix;
  static constexpr std::string_view kRuleSetPosfix = ProbGarChaseContext<FRAG_T>::kRuleSetPosfix;

  static constexpr std::string_view kSuppPosfix    = ProbGarChaseContext<FRAG_T>::kSuppPosfix;
  static constexpr std::string_view kSuppSetPosfix = ProbGarChaseContext<FRAG_T>::kSuppSetPosfix;
                  
  using    VertexIDType = typename ProbGarChaseContext<FRAG_T>::   VertexIDType;
  using VertexLabelType = typename ProbGarChaseContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType = typename ProbGarChaseContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType = typename ProbGarChaseContext<FRAG_T>::  EdgeLabelType;
  using    DataGraph    = typename ProbGarChaseContext<FRAG_T>::DataGraph;
  using      Pattern    = typename ProbGarChaseContext<FRAG_T>::  Pattern;
  using     ProbGarType = typename ProbGarChaseContext<FRAG_T>::  ProbGarType;

  using   PatternVertexHandle = typename GUNDAM::VertexHandle<  Pattern>::type;
  using DataGraphVertexHandle = typename GUNDAM::VertexHandle<DataGraph>::type;

  using CandidateSet = typename ProbGarChaseContext<FRAG_T>::CandidateSet;

  static constexpr auto compare_by_vertex_id 
  =[](const DataGraphVertexHandle& vertex_handle_0,
      const DataGraphVertexHandle& vertex_handle_1) {
    return vertex_handle_0->id() < vertex_handle_1->id();
  };

  // the fragment calculate confidence and deliever the file name of the gars
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

  // ######################################
  // ##   protocol with multi prob gar   ##
  // ######################################
  // from kTimmerFragID to ordinary workers
  //    info them to reset parameter
  static constexpr std::string_view kResetParameterPrefix = "#reset_parameter";

  static constexpr size_t kInitialPivotSetNum = 5;

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(ProbGarChase<FRAG_T>, 
                          ProbGarChaseContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for ProbGarChase.
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

    if (config["TimeLogFile"]) {
      assert(!ctx.time_log_file_.is_open());
      const std::string kTimeLogFileName = config["TimeLogFile"].as<std::string>();
      if (ctx.fid_ == kTimmerFragID) {
        ctx.time_log_file_.open(kTimeLogFileName, std::ios::app);
        if (!ctx.time_log_file_.is_open()) {
          util::Error("open time log file failed: " + kTimeLogFileName);
          return;
        }
        ctx.time_log_file_ << "time log of: " << ctx.yaml_file_ << std::endl;
      }
    }

    bool use_anytime_chase = false;
    float anytime_chase_time_limit = -1.0;
    float anytime_chase_conf_bound =  1.0;
    if (config["AnytimeChase"]) {
      use_anytime_chase = true;
      YAML::Node anytime_chase_config = config["AnytimeChase"];
      if (!anytime_chase_config["TimeLimit"]) {
        util::Error("does not specify time limit in AnytimeChase");
        return;
      }
      anytime_chase_time_limit = anytime_chase_config["TimeLimit"].as<float>();
      if ( anytime_chase_config["ConfBound"]) {
        anytime_chase_conf_bound = anytime_chase_config["ConfBound"].as<float>();
        if (anytime_chase_conf_bound <= 0 
         || anytime_chase_conf_bound >  1) {
          util::Error("Illegal confidence bound: " + std::to_string(anytime_chase_conf_bound));
          return;
        }
      }
    }

    if (!config["ProbGarPath"]) {
      util::Error("cannot get prob gar path");
      return;
    }
    YAML::Node prob_gar_path_config = config["ProbGarPath"];

    if (prob_gar_path_config.IsSequence()) {
      ctx.all_prob_gar_path_set_.reserve(prob_gar_path_config.size());
      for (int i = 0; i < prob_gar_path_config.size(); i++) {
        auto prob_gar_path_set = GetProbGarPathSetInfo(prob_gar_path_config[i]);
        if ( prob_gar_path_set.empty()) {
          util::Error("load prob gar path error!");
          return;
        }
        ctx.all_prob_gar_path_set_.emplace_back(prob_gar_path_set);
      }
    }
    else {
      auto prob_gar_path_set = GetProbGarPathSetInfo(prob_gar_path_config);
      if ( prob_gar_path_set.empty() ) {
        util::Error("load gar path error!");
        return;
      }
      ctx.all_prob_gar_path_set_.emplace_back(prob_gar_path_set);
    }

    if (ctx.all_prob_gar_path_set_.size() == 0) {
      util::Error("empty gar set");
      return;
    }
    
    ctx.all_prob_gar_set_.reserve(ctx.all_prob_gar_path_set_.size());
    assert(ctx.all_prob_gar_set_.empty());

    for (const auto& prob_gar_path_set : ctx.all_prob_gar_path_set_) {
      std::vector<ProbGarType> prob_gar_set;
      for (const auto& [kGarPathVFile, 
                        kGarPathEFile, 
                        kGarPathXFile, 
                        kGarPathYFile, 
                        kGarPathPFile] : prob_gar_path_set) {

        ProbGarType prob_gar;

        auto read_gar_res = prob_gar::ReadProbGAR(prob_gar, kGarPathVFile, 
                                                            kGarPathEFile, 
                                                            kGarPathXFile, 
                                                            kGarPathYFile, 
                                                            kGarPathPFile);

        if (read_gar_res < 0) {
          util::Error("read prob gar error! v file: " + kGarPathVFile
                                        + " e file: " + kGarPathEFile
                                        + " x file: " + kGarPathXFile
                                        + " y file: " + kGarPathYFile
                                        + " p file: " + kGarPathPFile);
          return;
        }
        prob_gar_set.emplace_back(std::move(prob_gar));
      }
      ctx.all_prob_gar_set_.emplace_back(std::move(prob_gar_set));
    }

    if (GUNDAM::ReadCSVGraph(ctx.data_graph_, 
                            kDataGraphPathVFile,
                            kDataGraphPathEFile) < 0) {
      util::Error("load data graph failed!");
      return;
    }

    for (auto& prob_gar_set : ctx.all_prob_gar_set_) {
      std::map<typename GUNDAM::EdgeID<DataGraph>::type, float> diff_edge_set;
      DataGraph data_graph(ctx.data_graph_);
      util::Info("process prob gar set: " + ctx.CurrentProbGarSetName());
      if (use_anytime_chase) {
        prob_gar::ProbGARAnytimeChase(prob_gar_set, 
                                        data_graph, 
                                     diff_edge_set, 
                          anytime_chase_conf_bound, 
                          anytime_chase_time_limit);
      }
      else {
        prob_gar::ProbGARChase(prob_gar_set, 
                                 data_graph, 
                              diff_edge_set);
      }
      // for (const auto& [edge_id, conf] : diff_edge_set) {
      //   util::Output("edge id: " + std::to_string(edge_id)
      //                + " conf: " + std::to_string(   conf));
      // }
      ctx.time_log_file_ << ctx.CurrentProbGarSetName() << std::endl;
      ctx.time_log_file_ << timer_now() << std::endl;
      ctx.ToNextProbGarSet();
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
   * @brief Incremental evaluation for ProbGarChase.
   *
   * @param frag
   * @param ctx
   * @param messageskAllocPivotSetPrefix
   */
  void IncEval(const fragment_t& frag, 
                      context_t& ctx,
              message_manager_t& messages) {
  
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

#endif  // EXAMPLES_ANALYTICAL_APPS_PROB_GAR_CHASE_PROB_GAR_CHASE_H_