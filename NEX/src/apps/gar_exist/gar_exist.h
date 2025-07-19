#ifndef EXAMPLES_ANALYTICAL_APPS_GAR_EXIST_GAR_EXIST_H_
#define EXAMPLES_ANALYTICAL_APPS_GAR_EXIST_GAR_EXIST_H_

#include <grape/grape.h>
#include <omp.h>

#include <functional>

#include "gundam/algorithm/match_using_match.h"

#include "gundam/tool/same_pattern.h"

#include "gundam/match/match.h"
#include "gundam/match/matchresult.h"

#include "gundam/graph_type/graph.h"
#include "gundam/graph_type/small_graph.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/large_graph2.h"
#include "gundam/io/csvgraph.h"
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "gar_exist/gar_exist_context.h"

#include "util/file.h"

#include "gar/csv_gar.h"
#include "gar/gar.h"
#include "gar/gar_match.h"
#include "gar/gar_config.h"
#include "gar/same_gar.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../timer.h"

namespace grape {

template <typename FRAG_T>
class GarExist : public ParallelAppBase<FRAG_T, GarExistContext<FRAG_T>>,
                 public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
                  
  using    VertexIDType = typename GarExistContext<FRAG_T>::   VertexIDType;
  using VertexLabelType = typename GarExistContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType = typename GarExistContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType = typename GarExistContext<FRAG_T>::  EdgeLabelType;
  using    DataGraph = typename GarExistContext<FRAG_T>::DataGraph;
  using      Pattern = typename GarExistContext<FRAG_T>::  Pattern;

  inline std::vector<std::string> GarPathSet(const std::string& gar_dir,
                                             const std::set<std::string>& rule_prefix_set) const {
    const std::string kRulePosfix = "_v.csv";

    std::vector<std::string> qualified_gar_path_set_in_one_dir;

    std::set<std::string> dir_files;

    util::GetFiles(gar_dir, dir_files);
    for (const auto& rule_prefix : rule_prefix_set){
      for (auto gar_file : dir_files){
        const bool isPrefix = rule_prefix.size() 
                              <= gar_file.size() 
            && std::mismatch(rule_prefix.begin(), 
                             rule_prefix. end (),
                                gar_file.begin(), 
                                gar_file. end ()).first == rule_prefix.end();
        const bool isPosfix = kRulePosfix.size() 
                              <= gar_file.size() 
             && std::mismatch(kRulePosfix.begin(), 
                              kRulePosfix. end (),
                                 gar_file.end() - kRulePosfix.size(), 
                                 gar_file.end()).first == kRulePosfix.end();
        if (!isPrefix || !isPosfix){
          continue;
        }

        gar_file = gar_file.substr(0, gar_file.length()
                                 - kRulePosfix.size());
        qualified_gar_path_set_in_one_dir.emplace_back(gar_dir + "/" + gar_file);
      }
    }
    return std::move(qualified_gar_path_set_in_one_dir);
  }

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(GarExist<FRAG_T>, 
                          GarExistContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for GarExist.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void PEval(const fragment_t& frag, 
                    context_t& ctx,
            message_manager_t& messages) {

    using GarType = gar::GraphAssociationRule<Pattern, DataGraph>;

    messages.InitChannels(thread_num());
    auto begin = timer();
    timer_next("build index");

    std::cout<<"yaml file: "<<ctx.yaml_file_<<std::endl;

    YAML::Node config = YAML::LoadFile(ctx.yaml_file_);

    std::vector<std::string> src_gars_path, 
                             dst_gars_path;
                             
    std::set<std::string> src_rule_prefix_set = {"gar_level_"};
    if (config["SrcRulePrefix"]){
      src_rule_prefix_set.clear();
      src_rule_prefix_set.emplace(config["SrcRulePrefix"].as<std::string>());
    }
    if (config["SrcGarDir"]) {
      const std::string kSrcGarDir = config["SrcGarDir"].as<std::string>();
      src_gars_path = GarPathSet(kSrcGarDir, src_rule_prefix_set);
    }
    else {
      if (!config["SrcGarNames"]){
        std::cout << "cannot get src gars!" << std::endl;
        return;
      }
      YAML::Node src_gar_names_config = config["SrcGarNames"]; 
      src_gars_path.reserve(src_gar_names_config.size());
      for(int i=0; i < src_gar_names_config.size(); ++i){
        const std::string kGarName = src_gar_names_config[i].as<std::string>();
        src_gars_path.emplace_back(kGarName);
      }
    }
    
    std::set<std::string> dst_rule_prefix_set = {"gar_level_"};
    if (config["DstRulePrefix"]){
      dst_rule_prefix_set.clear();
      dst_rule_prefix_set.emplace(config["DstRulePrefix"].as<std::string>());
    }
    if (config["DstGarDir"]) {
      const std::string kDstGarDir = config["DstGarDir"].as<std::string>();
      dst_gars_path = GarPathSet(kDstGarDir, dst_rule_prefix_set);
    }
    else {
      if (!config["DstGarNames"]){
        std::cout << "cannot get dst gars!" << std::endl;
        return;
      }
      YAML::Node dst_gar_names_config = config["DstGarNames"]; 
      dst_gars_path.reserve(dst_gar_names_config.size());
      for(int i=0; i < dst_gar_names_config.size(); ++i){
        const std::string kGarName = dst_gar_names_config[i].as<std::string>();
        dst_gars_path.emplace_back(kGarName);
      }
    }

    bool export_same = false;
    // true for export same gars in two sets
    // false to export the gars in src dir that cannot find the same
    // gar in dst dir
    if (config["ExportLog"]){
      const std::string kExportLog = config["ExportLog"].as<std::string>();
      if (kExportLog != std::string("same")
       && kExportLog != std::string("cannot_find")){
        std::cout<<"illegal configuration of ExportLog: "<<kExportLog<<std::endl;
        return;
      }
      if (kExportLog == std::string("same")){
        export_same = true;
      }
    }

    std::vector<std::pair<GarType, std::string>> src_gar_set;
    src_gar_set.reserve(src_gars_path.size());
    for (const auto& src_gar_path : src_gars_path){
      const std::string kGarVFile = src_gar_path + "_v.csv",
                        kGarEFile = src_gar_path + "_e.csv",
                        kGarXFile = src_gar_path + "_x.csv",
                        kGarYFile = src_gar_path + "_y.csv";
      GarType gar;
      auto res = gar::ReadGAR(gar, kGarVFile, kGarEFile,
                                   kGarXFile, kGarYFile);
      if (res < 0){
        std::cout<<"loading gar: "<<src_gar_path<<" failed! :"<<res<<std::endl;
        return;
      }
      src_gar_set.emplace_back(std::move(gar), src_gar_path);
    }

    std::vector<std::pair<GarType, std::string>> dst_gar_set;
    dst_gar_set.reserve(dst_gars_path.size());
    for (const auto& dst_gar_path : dst_gars_path){
      const std::string kGarVFile = dst_gar_path + "_v.csv",
                        kGarEFile = dst_gar_path + "_e.csv",
                        kGarXFile = dst_gar_path + "_x.csv",
                        kGarYFile = dst_gar_path + "_y.csv";
      GarType gar;
      auto res = gar::ReadGAR(gar, kGarVFile, kGarEFile,
                                   kGarXFile, kGarYFile);
      if (res < 0){
        std::cout<<"loading gar: "<<dst_gar_path<<" failed! :"<<res<<std::endl;
        return;
      }
      dst_gar_set.emplace_back(std::move(gar), dst_gar_path);
    }

    for (int i = 0; i < src_gar_set.size(); i++) {
      const auto& kSrcGar     = src_gar_set[i].first;
      const auto& kSrcGarName = src_gar_set[i].second;

      bool duplicated = false;
      for (const auto& dst_gar : dst_gar_set){
        const auto& kDstGar     = dst_gar.first;
        const auto& kDstGarName = dst_gar.second;

        if (!gar::SameGar(kSrcGar, kDstGar)){
          // this two gar is not the same 
          continue;
        }

        // these two gars are the same
        if (export_same){
          std::cout<<"gar: "<<kSrcGarName
                   <<" is the same as gar: "<<kDstGarName<<std::endl;
        }
        duplicated = true;
        break;
      }
      if (!duplicated && !export_same){
        std::cout<<"gar: "<<kSrcGarName
                 <<" cannot find the same gar in dst gars! "<<std::endl;
      }
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
   * @brief Incremental evaluation for GarExist.
   *
   * @param frag
   * @param ctx
   * @param messages
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

#endif  // EXAMPLES_ANALYTICAL_APPS_GAR_EXIST_GAR_EXIST_H_