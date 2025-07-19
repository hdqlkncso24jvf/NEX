#ifndef EXAMPLES_ANALYTICAL_APPS_GAR_SUPP_GAR_SUPP_H_
#define EXAMPLES_ANALYTICAL_APPS_GAR_SUPP_GAR_SUPP_H_

#include <grape/grape.h>
#include <omp.h>

#include <functional>

#include "gundam/graph_type/graph.h"
#include "gundam/match/match.h"
#include "gundam/io/csvgraph.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/large_graph2.h"
#include "gundam/match/matchresult.h"
#include "gundam/graph_type/small_graph.h"
#include "gundam/io/csvgraph.h"
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../timer.h"

#include "gar_supp/gar_supp_context.h"

#include "gar/csv_gar.h"
#include "gar/gar.h"
#include "gar/gar_supp.h"
#include "gar/gar_match.h"
#include "gar/gar_config.h"

#include "util/log.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

namespace grape {

template <typename FRAG_T>
class GarSupp : public ParallelAppBase<FRAG_T, GarSuppContext<FRAG_T>>,
                public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
  static constexpr std::string_view kWorkerGarSuppPosfix = "@worker";

  static constexpr std::string_view kRulePosfix    = GarSuppContext<FRAG_T>::kRulePosfix;
  static constexpr std::string_view kRuleSetPosfix = GarSuppContext<FRAG_T>::kRuleSetPosfix;

  static constexpr std::string_view kSuppPosfix    = GarSuppContext<FRAG_T>::kSuppPosfix;
  static constexpr std::string_view kSuppSetPosfix = GarSuppContext<FRAG_T>::kSuppSetPosfix;
                  
  using    VertexIDType = typename GarSuppContext<FRAG_T>::   VertexIDType;
  using VertexLabelType = typename GarSuppContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType = typename GarSuppContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType = typename GarSuppContext<FRAG_T>::  EdgeLabelType;
  using    DataGraph    = typename GarSuppContext<FRAG_T>::DataGraph;
  using      Pattern    = typename GarSuppContext<FRAG_T>::  Pattern;
  using      GarType    = typename GarSuppContext<FRAG_T>::GARPatternToDataGraph;

  // the fragment calculate confidence and deliever the file name of the gars
  // to other workers
  static constexpr int kCalcConfFragID = 0;

  // from ordinary workers to kCalcConfFragID
  //    info the kCalcConfFragID how many process each worker have
  static constexpr std::string_view kInfoProcessPrefix = "#info_process";
  // from kCalcConfFragID to ordinary worker
  //    deliever the file name of the gars that they needs to process
  static constexpr std::string_view kDeliverGarFileNamePrefix = "#deliver_gar_file_name";
  // from ordinary worker to kCalcConfFragID
  //    inform that all support has been calcluated, it would be ok to calcluate the
  //    confidence
  static constexpr std::string_view kInfoFinishPrefix = "#info_finish";

  // legal gar supp file for workers:
  //        xxx + kSuppPosfix/kSuppSetPosfix + kWorkerGarSuppPosfix + [0-9]+
  std::pair<std::string, bool> GarSuppFileWithOutWorkerPrefix(const std::string& kFile){
    std::string supp_file_without_worker_posfix = kFile;
    if (supp_file_without_worker_posfix.back() < '0'
     || supp_file_without_worker_posfix.back() > '9'){
      // is not worker posfix
      std::cout << "the last charactor is not number" << std::endl;
      return std::pair(std::string(), false);
    }
    // remove the suffixal numbers
    size_t posfix_idx = kFile.size() - 1;
    for (;posfix_idx >= 0; posfix_idx--) {
      if (supp_file_without_worker_posfix[posfix_idx] >= '0'
       && supp_file_without_worker_posfix[posfix_idx] <= '9'){
        continue;
      }
      break;
    }
    if (posfix_idx == 0){
      // this string containes only numbers, is illegal
      std::cout << "this string containes only numbers" << std::endl;
      return std::pair(std::string(), false);
    }
    assert(posfix_idx > 0);
    assert (supp_file_without_worker_posfix[posfix_idx] < '0'
         || supp_file_without_worker_posfix[posfix_idx] > '9');

    supp_file_without_worker_posfix = supp_file_without_worker_posfix.substr(0, posfix_idx + 1);

    // remove kWorkerGarSuppPosfix
    const bool isWorkerPosfix = kWorkerGarSuppPosfix.size()
                                         <= supp_file_without_worker_posfix.size()
               && std::mismatch(kWorkerGarSuppPosfix.begin(),
                                kWorkerGarSuppPosfix. end (),
                                supp_file_without_worker_posfix.end()         
                              - kWorkerGarSuppPosfix.size(), 
                                supp_file_without_worker_posfix.end()).first 
                             == kWorkerGarSuppPosfix.end();

    if (!isWorkerPosfix){
      // is not worker posfix 
      std::cout << "is not worker posfix:" << supp_file_without_worker_posfix << std::endl;
      return std::pair(std::string(), false);
    }

    supp_file_without_worker_posfix = supp_file_without_worker_posfix.substr(0, 
                                      supp_file_without_worker_posfix.length()
                                    - kWorkerGarSuppPosfix.size());

    // to check whether the current posfix is kSuppPosfix or kSuppSetPosfix
    const bool isSuppPosfix = kSuppPosfix.size() <= supp_file_without_worker_posfix.size() 
             && std::mismatch(kSuppPosfix.begin(), 
                              kSuppPosfix. end (),
                               supp_file_without_worker_posfix.end()         
                            - kSuppPosfix.size(), 
                               supp_file_without_worker_posfix.end()).first 
                           == kSuppPosfix.end();
    if (isSuppPosfix){
      return std::pair(supp_file_without_worker_posfix, true);
    }

    const bool isSuppSetPosfix = kSuppSetPosfix.size() <= supp_file_without_worker_posfix.size() 
                && std::mismatch(kSuppSetPosfix.begin(), 
                                 kSuppSetPosfix. end (),
                                 supp_file_without_worker_posfix.end()         - kSuppSetPosfix.size(), 
                                 supp_file_without_worker_posfix.end()).first == kSuppSetPosfix.end();
    if (isSuppSetPosfix){
      return std::pair(supp_file_without_worker_posfix, true);
    }

    std::cout << "is not supp/supp_set posfix" << std::endl;
    return std::pair(std::string(), false);
  }

  void CollectGarSuppFromWorkers(const std::string& kGarDir){
    std::set<std::string> dir_files;
    util::GetFiles(kGarDir, dir_files);

    std::cout << "CollectGarSuppFromWorkers: " << kGarDir << std::endl;

    std::map<std::string, // supp file removed worker posfix
             std::vector<std::string> // supp files does not remove worker posfix
            > file_name_remove_worker_posfix_set;
    for (const auto& file : dir_files){
      std::cout << "file: " << file << std::endl;
      auto [file_name_remove_worker_posfix,
            file_name_remove_worker_posfix_ret] = this->GarSuppFileWithOutWorkerPrefix(file);
      if (!file_name_remove_worker_posfix_ret){
        assert(file_name_remove_worker_posfix == "");
        // does not have legal posfix
        std::cout << "does not have legal posfix" << std::endl;
        continue;
      }
      std::cout << "has legal posfix" << std::endl;
      assert(file_name_remove_worker_posfix != "");
      // add file_name_remove_worker_posfix to file_name_remove_worker_posfix_set
      // if it does not exist before 
      file_name_remove_worker_posfix_set[file_name_remove_worker_posfix].emplace_back(file);
    }

    std::cout << "##############################" << std::endl
              << "#  " << file_name_remove_worker_posfix_set.size() << std::endl
              << "##############################" << std::endl;

    for (const auto& file_name_remove_worker_posfix 
                   : file_name_remove_worker_posfix_set) {
      std::set<std::string> gar_supp_set;
      std::ifstream  file_remove_worker_posfix
               (kGarDir + "/" + file_name_remove_worker_posfix.first); 
      std::cout << "file_name_remove_worker_posfix.first: "
                <<  file_name_remove_worker_posfix.first
                << std::endl;
      while (file_remove_worker_posfix) {
        std::string s;
        if (!std::getline( file_remove_worker_posfix, s ))
          break;
        gar_supp_set.emplace(s);
      }
      std::cout << "gar_supp_set.size(): "
                <<  gar_supp_set.size()
                << std::endl;

      file_remove_worker_posfix.close();

      for (const auto& file_name_with_worker_posfix
                     : file_name_remove_worker_posfix.second){
        std::ifstream  file_with_worker_posfix
                 (kGarDir + "/" + file_name_with_worker_posfix); 
        while (file_with_worker_posfix) {
          std::string s;
          if (!std::getline( file_with_worker_posfix, s ))
            break;
          gar_supp_set.emplace(s);
        }
      }

      std::ofstream  file_remove_worker_posfix_out
               (kGarDir + "/" + file_name_remove_worker_posfix.first); 
      for (const auto& gar_supp : gar_supp_set){
        file_remove_worker_posfix_out << gar_supp << std::endl;
      }
    }
    return;
  }

  void ExportSupp(int worker_id,
                  std::map<std::string, std::ofstream>& gar_supp_file_set,
                     const std::string& gar_name,
                     const std::string& gar_file_name, 
                          uint64_t  x_supp,
                          uint64_t xy_supp) const {

    const std::string kThisWorkerGarFileName = gar_file_name
                                             + std::string(kWorkerGarSuppPosfix)
                                             + std::to_string(worker_id);

    std::cout << "# gar_name: " << gar_name << std::endl 
              << "# gar_file_name: " << kThisWorkerGarFileName << std::endl 
              << "# x_supp: " << x_supp << std::endl
              << "# xy_supp: " << xy_supp << std::endl;

    auto [gar_supp_file_set_it,
          gar_supp_file_set_ret]
        = gar_supp_file_set.emplace(kThisWorkerGarFileName, std::ofstream());

    if (gar_supp_file_set_ret){
      // successfully added, this support file does not exist before
      gar_supp_file_set_it->second.open(kThisWorkerGarFileName, std::ios::out 
                                                              | std::ios::trunc);
    }
    gar_supp_file_set_it->second << gar_name 
                         << "\t" <<  xy_supp 
                         << "\t" <<   x_supp << std::endl;
    return;
  }

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(GarSupp<FRAG_T>, 
                          GarSuppContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for GarSupp.
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

    ctx.rule_prefix_set_ = {""};
    if (config["RulePrefix"]){
      ctx.rule_prefix_set_.clear();
      ctx.rule_prefix_set_.emplace(config["RulePrefix"].as<std::string>());
    }
    
    if (!config["Action"]){
      std::cout << "does not specify what to do!" 
                << std::endl;
      return;
    }
    ctx.action_
      = config["Action"].as<std::string>();

    if (ctx.action_ != "calculate_confidence" 
     && ctx.action_ != "calculate_support"){
      std::cout<<"unknown action type: "
               <<ctx.action_<<std::endl;
      return;
    }
    
    if (!config["DataGraph"]){
      std::cout << "cannot get data graph!" 
                << std::endl;
      return;
    }
    YAML::Node data_graphs_config = config["DataGraph"];

    ctx.time_limit_ = -1.0;
    if (config["TimeLimit"]) {
      ctx.time_limit_ = config["TimeLimit"].as<double>();
    }
    
    ctx.time_limit_per_supp_ = -1.0;
    if (config["TimeLimitPerSupp"]) {
      ctx.time_limit_per_supp_ = config["TimeLimitPerSupp"].as<double>();
    }
    
    ctx.pivot_all_vertexes_ = false;
    if (config["Pivot"]) {
      const std::string pivot_setting = config["Pivot"].as<std::string>();
      if (pivot_setting == "rhs_only") {
        ctx.pivot_all_vertexes_ = false;
      }
      else if (pivot_setting == "all_vertexes") { 
        ctx.pivot_all_vertexes_ = true;
      }
      else {
        util::Error("unknown pivot setting: " + pivot_setting);
        return;
      }
    }

    for(int i=0; i < data_graphs_config.size(); ++i){
      YAML::Node data_graph_config = data_graphs_config[i];
      if (!data_graph_config["GraphDir"]){
        std::cout << "the dir of the " << i+1
                  << "'th data graph has not been been configured"
                  << std::endl;
        continue;
      }
      const std::string kDataGraphDir
        = data_graph_config["GraphDir"].as<std::string>();

      if (!data_graph_config["GraphName"]){
        std::cout << "the name of the " << i+1
                  << "'th data graph has not been been configured"
                  << std::endl;
        return;
      }
      const std::string kDataGraphName
        = data_graph_config["GraphName"].as<std::string>();

      if (!data_graph_config["MlLiteralEdgesFile"]){
        ctx.AddDataGraphFile(kDataGraphDir,
                             kDataGraphName);
        continue;
      }

      std::string ml_literal_edges_file = data_graph_config["MlLiteralEdgesFile"].as<std::string>();

      ctx.AddDataGraphFile(kDataGraphDir,
                           kDataGraphName,
                           ml_literal_edges_file);
    }
        
    if (!config["GarDirs"]) {
      std::cout << "cannot get gar dirs!" << std::endl;
      return;
    }
    YAML::Node gar_dirs_config = config["GarDirs"];

    for(int i=0; i < gar_dirs_config.size(); ++i){
      ctx.gar_dirs_.emplace_back(gar_dirs_config[i].as<std::string>());
    }

    if (ctx.fid_ == kCalcConfFragID) {
      for (const auto& gar_dir : ctx.gar_dirs_) {
        this->CollectGarSuppFromWorkers(gar_dir);
      }
    }

    ctx.supp_candidate_posfix_ = "";
    
    if (config["PreserveYLiteralType"]) {
      // should not specify both the literal that need 
      // to be removed and those need to be preserved
      assert(!config["RemoveYLiteralType"]);
      
      YAML::Node preserve_y_literal_config = config["PreserveYLiteralType"];
      if (preserve_y_literal_config.size() != 1){
        std::cout<<"can only preserve one kind of literal!"<<std::endl;
        return;
      }
      for(int i=0; i < preserve_y_literal_config.size(); ++i){
        YAML::Node y_literal_type_config 
             = preserve_y_literal_config[i];
        if (!y_literal_type_config["XVertexLabel"]){
          std::cout<<"does not specify x vertex label"<<std::endl;
          continue;
        }
        if (!y_literal_type_config["YVertexLabel"]){
          std::cout<<"does not specify y vertex label"<<std::endl;
          continue;
        }
        if (!y_literal_type_config["EdgeLabel"]){
          std::cout<<"does not specify edge label"<<std::endl;
          continue;
        }
        ctx.preserve_edge_literal_types_.emplace(
                        y_literal_type_config["XVertexLabel"].as<VertexLabelType>(),
                        y_literal_type_config[   "EdgeLabel"].as<  EdgeLabelType>(),
                        y_literal_type_config["YVertexLabel"].as<VertexLabelType>());

        if (!y_literal_type_config["SuppCandidate"]){
          // does not specify the support candidate
          continue;
        }

        // support candidate has been specified
        std::string supp_candidate_file_path 
          = y_literal_type_config["SuppCandidate"].as<std::string>();

        ctx.supp_candidate_posfix_ = "_candidate";

        std::ifstream supp_candidate_file(
                      supp_candidate_file_path);
        std::string str; 

        if (supp_candidate_file.fail()){
          std::cout<<"open candidate file: "
                   <<supp_candidate_file_path<<" error!"<<std::endl;
          return;
        }
        
        while (std::getline(supp_candidate_file, str)) {
          std::string buf;                 // Have a buffer string

          std::stringstream ss(str);       // Insert the string into a stream

          std::vector<std::string> tokens; // Create vector to hold our words

          while (ss >> buf)
            tokens.emplace_back(buf);

          std::vector<VertexIDType> supp_candidate;
          supp_candidate.reserve(tokens.size());
          for (const auto& candidate_id_str : tokens){
            supp_candidate.emplace_back(std::stoi(candidate_id_str));
          }
          ctx.supp_candidates_.emplace(std::move(supp_candidate));
        }
        if (ctx.supp_candidates_.empty()){
          std::cout<<"specify an empty support candidates set"<<std::endl;
          return;
        }
        std::cout<<"support candidate set: "<<ctx.supp_candidates_.size()<<std::endl;
      }
    }
    else{
      if (config["RemoveYLiteralType"]){
        YAML::Node remove_y_literal_config = config["RemoveYLiteralType"];
        for(int i=0; i < remove_y_literal_config.size(); ++i){
          YAML::Node y_literal_type_config 
            = remove_y_literal_config[i];
          if (!y_literal_type_config["XVertexLabel"]){
            std::cout<<"does not specify x vertex label"<<std::endl;
            continue;
          }
          if (!y_literal_type_config["YVertexLabel"]){
            std::cout<<"does not specify y vertex label"<<std::endl;
            continue;
          }
          if (!y_literal_type_config["EdgeLabel"]){
            std::cout<<"does not specify edge label"<<std::endl;
            continue;
          }
          ctx.remove_edge_literal_types_.emplace(
                        y_literal_type_config["XVertexLabel"].as<VertexLabelType>(),
                        y_literal_type_config[   "EdgeLabel"].as<  EdgeLabelType>(),
                        y_literal_type_config["YVertexLabel"].as<VertexLabelType>());
        }
      }
    }

    timer_next("run gar supp");

    ctx.ResetProcessingDataGraph();

    std::string msg(kInfoProcessPrefix);
    msg += " " + std::to_string(ctx.fid_)
         + " " + std::to_string(omp_get_num_procs());
    auto& channel_0 = messages.Channels()[0];
    channel_0.SendToFragment(kCalcConfFragID, msg);

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
   * @brief Incremental evaluation for GarSupp.
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
         receive_deliver      = false, 
         receive_info_finish  = false, 
         receive_message      = false;

    messages.ParallelProcess<std::string>(
        // thread_num(),
        1, [&ctx, 
            &gar_file_set,
            &receive_info_process,
            &receive_deliver, 
            &receive_info_finish, 
            &receive_message](int tid, std::string msg) {
          std::cout << "fid: " << ctx.fid_ << " receive message: " << msg
                    << std::endl;
          auto res_info_process 
            = std::mismatch(kInfoProcessPrefix.begin(),
                            kInfoProcessPrefix. end (), msg.begin());
          auto res_deliver 
            = std::mismatch(kDeliverGarFileNamePrefix.begin(),
                            kDeliverGarFileNamePrefix. end (), msg.begin());
          auto res_info_finish 
            = std::mismatch(kInfoFinishPrefix.begin(),
                            kInfoFinishPrefix. end (), msg.begin());
          receive_message = true;
          if (res_info_finish.first == kInfoFinishPrefix.end()) {
            msg = msg.substr(kInfoFinishPrefix.size());
            // kInfoFinishPrefix is the prefix of msg.
            receive_info_finish = true;
            // a data graph has been processed
            // ctx.current_process_data_graph_idx_ has
            // already moved to the next data graph 
            // that were not processes except the kCalcConfFragID
            std::cout<<"## receive_info_finish ##"<<std::endl;
          } else if (res_deliver.first == kDeliverGarFileNamePrefix.end()) {
            msg = msg.substr(kDeliverGarFileNamePrefix.size());
            // kDeliverGarFileNamePrefix is the prefix of msg.
            receive_deliver = true;
            // received the gar files from the kCalcConfFragID
            std::stringstream ss;
            std::string str;
            ss << msg;
            int counter = 0;
            while (ss >> str) {
              const int kStrNum = std::stoi(str);
              assert(kStrNum > 0);
              std::vector<std::string> str_set;
              str_set.reserve(kStrNum);
              for (int i = 0; i < kStrNum; i++) {
                ss >> str;
                str_set.emplace_back(str);
              }
              if (str_set.size() == 1){
                counter++;
              }
              else{
                counter += str_set.size() - 1;
              }
              gar_file_set.emplace_back(std::move(str_set));
            }
            std::cout << "##  receive_deliver  ##" <<std::endl;
            std::cout << "gar number: " << counter <<std::endl;
          } else if (res_info_process.first == kInfoProcessPrefix.end()){
            msg = msg.substr(kInfoProcessPrefix.size());
            // kInfoProcessPrefix is the prefix of msg.
            receive_info_process = true;
            std::string fid_str, proc_num_str;
            std::stringstream ss;
            ss << msg;
            ss >> fid_str;
            ss >> proc_num_str;
            ctx.process_num_.emplace(std::stoi(fid_str), 
                                     std::stoi(proc_num_str));
            std::cout<<"## receive_info_process ##"<<std::endl;
          } else {
            // unknown message type
            assert(false);
          }
        });

    if (ctx.action_ == "calculate_confidence"
     && receive_deliver){
      // does not calculate support, collect existed confidence
      std::string msg(kInfoFinishPrefix);
      auto& channel_0 = messages.Channels()[0];
      channel_0.SendToFragment(kCalcConfFragID, msg);   
      return;
    }

    if (ctx.fid_ == kCalcConfFragID) {
      // only the kCalcConfFragID does not move 
      // current_process_data_graph_idx_ to the next
      // unprocessed data graph after processing

      for (const auto& gar_dir : ctx.gar_dirs_) {
        this->CollectGarSuppFromWorkers(gar_dir);
      }
      
      assert(!ctx.AllDataGraphProcessed());
      if (receive_info_finish) {
        std::cout<<"## receive_info_finish is true ##"<<std::endl;
        // a data graph has been processed, 
        // calculate the confidence of gar on it

        using GarType = gar::GraphAssociationRule<Pattern, DataGraph>;

        for (const auto& gar_dir : ctx.gar_dirs_) {

          std::cout << "processing folder: " << gar_dir << std::endl;

          std::vector<std::vector<std::string>> qualified_gar_set
                                          = ctx.QualifiedGarSet(gar_dir, true);

          std::cout << "qualified gar file number in folder: "
                    <<  gar_dir << std::endl;

          std::string confidence_file_path
                = ctx.ConfFilePathOnCurrentDataGraph(gar_dir);

          std::cout << "confidence file: "
                    <<  confidence_file_path << std::endl;

          std::ofstream confidence_file(
                        confidence_file_path, std::ios::out 
                                            | std::ios::trunc);

          for (const auto& qualified_gar_path
                         : qualified_gar_set) {

            std::map<std::string, 
                     std::pair<int, int>> qualified_gar_supp_set;

            ctx.GatherSuppOnCurrentDataGraph(qualified_gar_path, 
                                             qualified_gar_supp_set);

            for (const auto& qualified_gar_supp
                           : qualified_gar_supp_set) {
              auto& supp_xy = qualified_gar_supp.second.first;
              auto& supp_x  = qualified_gar_supp.second.second;
              auto& qualified_gar_path = qualified_gar_supp.first;
              std::cout << "supp_xy: " << supp_xy << std::endl
                        << "supp_x : " << supp_x  << std::endl;
              assert(supp_xy <= supp_x);
              if (supp_x == 0) {
                // supp = 0
                assert(supp_xy == 0);
                confidence_file << qualified_gar_path
                                << "\t" << 0
                                << "\t" << supp_xy
                                << "\t" << supp_x
                                << std::endl;
                continue;
              }
              confidence_file << qualified_gar_path
                              << "\t" << ((double)supp_xy) / ((double)supp_x)
                              << "\t" << supp_xy
                              << "\t" << supp_x
                              << std::endl;
            }
          }
        }
        //mantain the processing id of it 
        ctx.MoveToNextDataGraph();
      }

      if (receive_info_finish
       || receive_info_process) {
        // if received info_finish, it means that the current 
        // processing has been done
        // received info_process, it means that it is ready
        // to process the first data graph
        assert(!receive_deliver);

        if (ctx.AllDataGraphProcessed()){
          // all data graph has been processed
          return;
        }
        // begin to process the next (or first) data graph
        const auto& data_graph_file 
              = ctx.CurrentProcessDataGraphFile();

        std::vector<std::string> gar_file_set;

        // build up the workload mapping table based on the
        // proc_num each worker has
        std::vector<int> worker_id;
        for (const auto& process_num : ctx.process_num_){
          for (int i = 0; i < process_num.second; i++)
            worker_id.emplace_back(process_num.first);
        }
        std::random_shuffle ( worker_id.begin(), 
                              worker_id. end () );
        std::vector<std::string> msg_to_workers;
        msg_to_workers.resize(ctx.frag_num_);

        // count the current gar index
        int counter = 0;
        for(const auto& gar_dir : ctx.gar_dirs_){
          std::cout<<"processing folder: "<<gar_dir<<std::endl;

          std::vector<std::vector<std::string>> 
            qualified_gar_set = ctx.QualifiedGarSet(gar_dir, false);

          for (const auto& qualified_gar : qualified_gar_set) {
            assert(qualified_gar.size() >= 1);
            std::cout << "gar set file: " << qualified_gar[0] << std::endl;
            if (qualified_gar.size() == 1) {
              continue;
            }
            for (int i = 1; i < qualified_gar.size(); i++){
              std::cout << "\t" << qualified_gar[i];
            }
            std::cout << std::endl;
          }

          counter  = 0;
          std::vector<std::vector<std::vector<std::string>>>
            qualified_gar_set_for_workers;
          qualified_gar_set_for_workers.resize(ctx.frag_num_);
          // deliver the gathered gars to different workers
          for (const auto& qualified_gar : qualified_gar_set) {
            assert(qualified_gar.size() >= 1);
            if (qualified_gar.size() == 1) {
              const int to_worker = worker_id.at(counter % worker_id.size());
              counter++;
              assert(to_worker >= 0 && to_worker < ctx.frag_num_);
              qualified_gar_set_for_workers[to_worker].emplace_back(
                  std::vector<std::string>{qualified_gar[0]});
              continue;
            }
            std::vector<bool> has_set_to_worker(ctx.frag_num_, false);
            for (int i = 1; i < qualified_gar.size(); i++) {
              const int to_worker = worker_id.at(counter % worker_id.size());
              counter++;
              assert(to_worker >= 0 && to_worker < ctx.frag_num_);
              if (!has_set_to_worker[to_worker]){
                has_set_to_worker[to_worker] = true;
                qualified_gar_set_for_workers[to_worker].emplace_back(
                  std::vector<std::string>{qualified_gar[0]});
              }
              assert(*(qualified_gar_set_for_workers[to_worker].back().begin())
                    == qualified_gar[0]);
              qualified_gar_set_for_workers[to_worker].back()
                                                      .emplace_back(std::move(qualified_gar[i]));
            }
          }

          for (int worker_id = 0; worker_id < ctx.frag_num_; worker_id++){
            assert(worker_id  < qualified_gar_set_for_workers.size());
            qualified_gar_set = qualified_gar_set_for_workers[worker_id];

            std::string qualified_gar_set_str;

            for (const auto& qualified_gar : qualified_gar_set){
              qualified_gar_set_str = std::move(qualified_gar_set_str) 
                                    + "\t"
                                    + std::to_string(qualified_gar.size());
              for (auto& str : qualified_gar) {
                qualified_gar_set_str = std::move(qualified_gar_set_str) 
                                      + "\t"
                                      + std::move(str);
              }
            }
            msg_to_workers[worker_id] = std::move(qualified_gar_set_str);
          }
        }
        
        auto& channel_0 = messages.Channels()[0];
        // send message to ordinary worker
        for (int dst_fid = 0; dst_fid < ctx.frag_num_; dst_fid++) {
          std::string msg(kDeliverGarFileNamePrefix);
          channel_0.SendToFragment(dst_fid, std::move(msg) 
                                          + std::move(msg_to_workers[dst_fid]));
        }
        // all message are sended
        return;
      }
    }

    if (!receive_message) {
      return;
    }

    // process the received gars
    assert(receive_deliver);

    // read the data graph
    DataGraph data_graph;
    int res = GUNDAM::ReadCSVGraph(data_graph, 
                  ctx.CurrentProcessDataGraphFile().DataGraphPrefixToString() + "_v.csv",
                  ctx.CurrentProcessDataGraphFile().DataGraphPrefixToString() + "_e.csv");
    if (res < 0){
      std::cout << "load data graph failed: " 
                <<  ctx.CurrentProcessDataGraphFile().DataGraphPrefixToString()
                << std::endl;
      return;
    }

    std::map<EdgeLabelType,
             EdgeLabelType> ml_to_normal_edge_label, 
                            normal_to_ml_edge_label;

    bool kHasMlLiteralEdgesFile = ctx.CurrentProcessDataGraphFile().has_ml_literal_edges_file();

    if (kHasMlLiteralEdgesFile) {
      std::cout << "has ml literal edges file" << std::endl;
      EdgeLabelType max_edge_label = EdgeLabelType(),
        minimal_negtive_edge_label = EdgeLabelType();

      EdgeIDType edge_id_allocator = EdgeIDType();
      for (auto vertex_it = data_graph.VertexBegin();
               !vertex_it.IsDone();
                vertex_it++) {
        for (auto out_edge_it = vertex_it->OutEdgeBegin();
                 !out_edge_it.IsDone();
                  out_edge_it++) {
          edge_id_allocator = edge_id_allocator < out_edge_it->id()?
                              edge_id_allocator : out_edge_it->id();
          if (out_edge_it->label() < 0) {
            minimal_negtive_edge_label = minimal_negtive_edge_label < out_edge_it->label()?
                                         minimal_negtive_edge_label : out_edge_it->label();
            continue;
          }
          max_edge_label = max_edge_label > out_edge_it->label()?
                           max_edge_label : out_edge_it->label();
        }
      }
      const EdgeLabelType kMlEdgeLabelOffset = max_edge_label 
                                 - minimal_negtive_edge_label + 1;
      edge_id_allocator++;

      const std::string kMlLiteralEdgesFile
          = ctx.CurrentProcessDataGraphFile()
                     .ml_literal_edges_file();
      std::ifstream ml_literal_edge_file(kMlLiteralEdgesFile);
      
      if (!ml_literal_edge_file.good()){
        std::cout << " ml literal edge file is not good! " << std::endl;
        return;
      }

      while (ml_literal_edge_file) {
        std::string s;
        if (!std::getline( ml_literal_edge_file, s )) 
          break;

        std::istringstream ss( s );
        std::vector <std::string> record;
        record.reserve(3);
        while (ss) {
          std::string s;
          if (!std::getline( ss, s, ',' )) {
            break;
          }
          record.emplace_back( s );
        }
         VertexIDType  src_id    = GUNDAM::StringToDataType< VertexIDType>(record[0]);
        EdgeLabelType edge_label = GUNDAM::StringToDataType<EdgeLabelType>(record[1]);
         VertexIDType  dst_id    = GUNDAM::StringToDataType< VertexIDType>(record[2]);
 
        EdgeLabelType ml_edge_label = edge_label + kMlEdgeLabelOffset;
        ml_to_normal_edge_label[ml_edge_label] = edge_label;
        normal_to_ml_edge_label[edge_label] = ml_edge_label;
        auto [ edge_handle,
               edge_ret ] = data_graph.AddEdge(src_id, dst_id, 
                                               ml_edge_label,
                                                  edge_id_allocator++);
        assert(edge_ret);
      }
    }

    std::cout << "number of gars to be verified: "
              << gar_file_set.size()
              << std::endl;

    using   PatternVertexHandleType = typename GUNDAM::VertexHandle<  Pattern>::type;
    using DataGraphVertexHandleType = typename GUNDAM::VertexHandle<DataGraph>::type;

    using   PatternVertexIDType = typename   Pattern::VertexType::IDType;
    using DataGraphVertexIDType = typename DataGraph::VertexType::IDType;
    
    using SuppSizeType = int32_t;

    std::vector<std::tuple<std::string, // gar file
                           std::string, // gar name
                               GarType>> gar_set;
    ctx.LoadGars(gar_file_set, gar_set);

    std::cout << "loaded gar_set size: "
              << gar_set.size() 
              << std::endl;

    std::map<std::string, 
             std::ofstream> gar_supp_file_set;

    omp_lock_t write_support_file_lock;
    omp_init_lock(&write_support_file_lock);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < gar_set.size(); i++) {
      const std::string& gar_file_name = std::get<0>(gar_set[i]),
                              gar_name = std::get<1>(gar_set[i]);

      GarType& gar = std::get<2>(gar_set[i]);

      if (ctx.supp_candidates_.empty()){
        // does not specified support candidate
        bool lhs_has_ml_literal = false;
        for (const auto& x_literal_ptr 
                   : gar.x_literal_set()) {
          if (x_literal_ptr->type()
                     == gar::LiteralType::kMlLiteral){
            lhs_has_ml_literal = true;
            break;
          }
        }
        if (!lhs_has_ml_literal){
          // does not have ml literal in lhs
          std::vector<PatternVertexHandleType> gar_pivot_set;
          if (ctx.pivot_all_vertexes_) {
            for (auto vertex_it = gar.pattern().VertexBegin();
                     !vertex_it.IsDone();
                      vertex_it++) {
              gar_pivot_set.emplace_back(vertex_it);
            }
            assert(gar_pivot_set.size() == gar.pattern().CountVertex());
          }
          auto [x_supp, xy_supp] 
            = gar::GarSupp(gar, data_graph, -1, -1, 
                           ctx.time_limit_, 
                           ctx.time_limit_per_supp_,
                           gar_pivot_set);
                                                
          assert(x_supp >= xy_supp);
          // found that vertex
          omp_set_lock(&write_support_file_lock);
          this->ExportSupp(ctx.fid_,
                          gar_supp_file_set,
                          gar_name, 
                          gar_file_name, x_supp,
                                        xy_supp);
          omp_unset_lock(&write_support_file_lock);
          continue;
        }
        // has ml literal in lhs
        std::cout << "evaluating gar with ml literal" << std::endl;
        Pattern pattern_with_ml_edge(gar.pattern());
        EdgeIDType max_edge_id = EdgeIDType();
        for (auto vertex_it = pattern_with_ml_edge.VertexBegin();
                 !vertex_it.IsDone();
                  vertex_it++) {
          for (auto out_edge_it = vertex_it->OutEdgeBegin();
                   !out_edge_it.IsDone();
                    out_edge_it++) {
            max_edge_id = max_edge_id > out_edge_it->id()?
                          max_edge_id : out_edge_it->id();
          }
        }
        max_edge_id++;
        using LiteralInfoType = gar::LiteralInfo<Pattern, DataGraph>;
        for (const auto& x_literal_ptr 
                   : gar.x_literal_set()) {
          if (x_literal_ptr->type()
                     == gar::LiteralType::kMlLiteral){
            LiteralInfoType literal_info = x_literal_ptr->info();
            auto ml_edge_label_it = normal_to_ml_edge_label.find(literal_info.edge_label());
            assert(ml_edge_label_it != normal_to_ml_edge_label.end());
            auto [edge_handle,
                  edge_ret ] = pattern_with_ml_edge.AddEdge(literal_info.x_id(),
                                                            literal_info.y_id(),
                                                             ml_edge_label_it->second,
                                                            max_edge_id++);
            assert(edge_ret);
          }
        }
        GarType gar_with_ml_edge(pattern_with_ml_edge);
        for (const auto& x_literal_ptr 
                   : gar.x_literal_set()) {
          if (x_literal_ptr->type()
                     == gar::LiteralType::kMlLiteral){
            // ml literal has already beed added in the pattern
            continue;
          }
          gar_with_ml_edge.AddX(x_literal_ptr->info());
        }
        for (const auto& y_literal_ptr 
                   : gar.y_literal_set()) {
          gar_with_ml_edge.AddY(y_literal_ptr->info());
        }

        std::vector<PatternVertexHandleType> gar_with_ml_edge_pivot_set;
        if (ctx.pivot_all_vertexes_) {
          for (auto vertex_it = gar_with_ml_edge.pattern().VertexBegin();
                   !vertex_it.IsDone();
                    vertex_it++) {
            gar_with_ml_edge_pivot_set.emplace_back(vertex_it);
          }
          assert(gar_with_ml_edge_pivot_set.size() 
              == gar_with_ml_edge.pattern().CountVertex());
        }

        auto [x_supp, xy_supp] 
          = gar::GarSupp(gar_with_ml_edge, data_graph, -1, -1, 
                         ctx.time_limit_, 
                         ctx.time_limit_per_supp_,
                         gar_with_ml_edge_pivot_set);    
        std::cout << "x_supp: " << x_supp << "\txy_supp: " << xy_supp << std::endl;    
        assert(x_supp >= xy_supp);
        // found that vertex
        omp_set_lock(&write_support_file_lock);
        this->ExportSupp(ctx.fid_,
                        gar_supp_file_set,
                        gar_name, 
                        gar_file_name, x_supp,
                                      xy_supp);
        omp_unset_lock(&write_support_file_lock);
        continue;
      }

      // the support set is specified, to verify each
      // support in the candidate
      std::vector<PatternVertexHandleType> supp_list_ordered;
      for (const auto& y_literal : gar.y_literal_set()){
        y_literal->CalPivot(supp_list_ordered);
      }

      using MatchMap = std::map<PatternVertexHandleType, 
                              DataGraphVertexHandleType>;

      auto& kPattern   = gar.pattern();
      auto& kDataGraph = data_graph;

      std::map<PatternVertexHandleType,
               std::vector<DataGraphVertexHandleType>> pattern_candidate_set;

      if (!GUNDAM::_dp_iso::InitCandidateSet<GUNDAM::MatchSemantics::kIsomorphism>(
                    kPattern,
                    kDataGraph,
                    pattern_candidate_set)) {
        // does not have match
        omp_set_lock(&write_support_file_lock);
        this->ExportSupp(ctx.fid_,
                         gar_supp_file_set,
                         gar_name,
                         gar_file_name, 0, 0);
        omp_unset_lock(&write_support_file_lock);
        continue;
      }
      if (!GUNDAM::_dp_iso::RefineCandidateSet(
                    kPattern, 
                    kDataGraph, 
                    pattern_candidate_set)) {
        // does not have match
        omp_set_lock(&write_support_file_lock);
        this->ExportSupp(ctx.fid_,
                         gar_supp_file_set,
                         gar_name,
                         gar_file_name, 0, 0);
        omp_unset_lock(&write_support_file_lock);
        continue;
      }

      uint64_t x_supp = 0, 
              xy_supp = 0;
      std::vector<PatternVertexHandleType> gar_pivot_set;
      if (ctx.pivot_all_vertexes_) {
        for (auto vertex_it = gar.pattern().VertexBegin();
                  !vertex_it.IsDone();
                  vertex_it++) {
          gar_pivot_set.emplace_back(vertex_it);
        }
        assert(gar_pivot_set.size() == gar.pattern().CountVertex());
      }
      for (const auto& candidate : ctx.supp_candidates_){
        // : need to add time limit here
        assert(supp_list_ordered.size() 
                    == candidate.size());

        GUNDAM::Match<Pattern, DataGraph> pattern_to_data_graph_partial_match;

        for (int i = 0; i < supp_list_ordered.size(); i++){
          pattern_to_data_graph_partial_match.AddMap(
                           gar.pattern().FindVertex(supp_list_ordered[i]->id()),
                              data_graph.FindVertex(candidate[i]));
        }
        auto [partial_x_supp, partial_xy_supp] 
          = gar::GarSupp(gar, data_graph, 
                   pattern_to_data_graph_partial_match, 
                   pattern_candidate_set,
                    -1, -1, ctx.time_limit_per_supp_, 
                            ctx.time_limit_per_supp_,
                         gar_pivot_set);

        assert(partial_x_supp >= partial_xy_supp);
        assert(partial_x_supp <= 1);
        if (partial_x_supp == 1){
          x_supp++;
        }
        if (partial_xy_supp == 1){
          xy_supp++;
        }
      }

      omp_set_lock(&write_support_file_lock);
      this->ExportSupp(ctx.fid_,
                       gar_supp_file_set,
                       gar_name,
                       gar_file_name, x_supp, 
                                     xy_supp);
      omp_unset_lock(&write_support_file_lock);
    }

    for (auto& gar_supp_file 
             : gar_supp_file_set){
      gar_supp_file.second.close();
    }

    if (ctx.fid_ != kCalcConfFragID){
      ctx.MoveToNextDataGraph();
    }
    
    std::string msg(kInfoFinishPrefix);
    auto& channel_0 = messages.Channels()[0];
    channel_0.SendToFragment(kCalcConfFragID, msg);        

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

#endif  // EXAMPLES_ANALYTICAL_APPS_GAR_SUPP_GAR_SUPP_H_