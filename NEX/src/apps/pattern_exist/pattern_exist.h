#ifndef EXAMPLES_ANALYTICAL_APPS_PATTERN_EXIST_PATTERN_EXIST_H_
#define EXAMPLES_ANALYTICAL_APPS_PATTERN_EXIST_PATTERN_EXIST_H_

#include <grape/grape.h>
#include <omp.h>

#include <functional>

#include "gundam/tool/same_pattern.h"

#include "gundam/match/match.h"
#include "gundam/match/matchresult.h"

#include "gundam/io/csvgraph.h"

#include "gundam/graph_type/graph.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/large_graph2.h"
#include "gundam/graph_type/small_graph.h"
#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "pattern_exist/pattern_exist_context.h"
#include "util/file.h"

#include "gar/csv_gar.h"
#include "gar/gar.h"
#include "gar/gar_match.h"
#include "gar/gar_config.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../timer.h"

namespace grape {

template <typename FRAG_T>
class PatternExist : public ParallelAppBase<FRAG_T, PatternExistContext<FRAG_T>>,
                     public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
                  
  using    VertexIDType = typename PatternExistContext<FRAG_T>::   VertexIDType;
  using VertexLabelType = typename PatternExistContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType = typename PatternExistContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType = typename PatternExistContext<FRAG_T>::  EdgeLabelType;
  using    DataGraph = typename PatternExistContext<FRAG_T>::DataGraph;
  using      Pattern = typename PatternExistContext<FRAG_T>::  Pattern;

  inline std::vector<std::string> PatternPathSet(const std::string& pattern_dir,
                                                 const std::set<std::string>& rule_prefix_set) const {
    const std::string kPatternPosfix = "_v.csv";

    std::vector<std::string> qualified_pattern_path_set_in_one_dir;

    std::set<std::string> dir_files;

    util::GetFiles(pattern_dir, dir_files);
    for (const auto& rule_prefix : rule_prefix_set){
      for (auto pattern_file : dir_files){
        const bool isPrefix = rule_prefix.size() 
                          <= pattern_file.size() 
            && std::mismatch(rule_prefix.begin(), 
                             rule_prefix. end (),
                            pattern_file.begin(), 
                            pattern_file. end ()).first == rule_prefix.end();
        const bool isPosfix = kPatternPosfix.size() 
                             <= pattern_file.size() 
             && std::mismatch(kPatternPosfix.begin(), 
                              kPatternPosfix. end (),
                                 pattern_file.end() - kPatternPosfix.size(), 
                                 pattern_file.end()).first == kPatternPosfix.end();
        if (!isPrefix || !isPosfix){
          continue;
        }

        pattern_file = pattern_file.substr(0, pattern_file.length()
                                 - kPatternPosfix.size());
        qualified_pattern_path_set_in_one_dir.emplace_back(pattern_dir + "/" + pattern_file);
      }
    }
    return std::move(qualified_pattern_path_set_in_one_dir);
  }

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(PatternExist<FRAG_T>, 
                          PatternExistContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for PatternExist.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void PEval(const fragment_t& frag, 
                    context_t& ctx,
            message_manager_t& messages) {

    using GARPatternToDataGraph 
        = gar::GraphAssociationRule<Pattern, DataGraph>;

    messages.InitChannels(thread_num());
    auto begin = timer();
    timer_next("build index");

    std::cout<<"yaml file: "<<ctx.yaml_file_<<std::endl;

    YAML::Node config = YAML::LoadFile(ctx.yaml_file_);

    std::vector<std::string> src_patterns_path, 
                             dst_patterns_path;
                             
    std::set<std::string> src_pattern_prefix_set = {"pattern_level_"};
    if (config["SrcPatternPrefix"]){
      src_pattern_prefix_set.clear();
      src_pattern_prefix_set.emplace(config["SrcPatternPrefix"].as<std::string>());
    }
    if (config["SrcPatternDir"]) {
      const std::string kSrcPatternDir = config["SrcPatternDir"].as<std::string>();
      src_patterns_path = PatternPathSet(kSrcPatternDir, src_pattern_prefix_set);
    }
    else {
      if (!config["SrcPatternNames"]){
        std::cout << "cannot get src patterns!" << std::endl;
        return;
      }
      else {
        YAML::Node src_pattern_names_config = config["SrcPatternNames"]; 
        src_patterns_path.reserve(src_pattern_names_config.size());
        for(int i=0; i < src_pattern_names_config.size(); ++i){
          const std::string kGarName = src_pattern_names_config[i].as<std::string>();
          src_patterns_path.emplace_back(kGarName);
        }
      }
    }
    
    std::set<std::string> dst_pattern_prefix_set = {"pattern_level_"};
    if (config["DstPatternPrefix"]){
      dst_pattern_prefix_set.clear();
      dst_pattern_prefix_set.emplace(config["DstPatternPrefix"].as<std::string>());
    }
    if (config["DstPatternDir"]) {
      const std::string kDstPatternDir = config["DstPatternDir"].as<std::string>();
      dst_patterns_path = PatternPathSet(kDstPatternDir, dst_pattern_prefix_set);
    }
    else {
      if (!config["DstPatternNames"]){
        std::cout << "cannot get dst patterns!" << std::endl;
        return;
      }
      else {
        YAML::Node dst_pattern_names_config = config["DstPatternNames"]; 
        dst_patterns_path.reserve(dst_pattern_names_config.size());
        for(int i=0; i < dst_pattern_names_config.size(); ++i){
          const std::string kGarName = dst_pattern_names_config[i].as<std::string>();
          dst_patterns_path.emplace_back(kGarName);
        }
      }
    }

    bool export_same = false;
    // true for export same patterns in two sets
    // false to export the patterns in src dir that cannot find the same
    // pattern in dst dir
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

    std::vector<std::pair<Pattern, std::string>> src_pattern_set;
    src_pattern_set.reserve(src_patterns_path.size());
    for (const auto& src_pattern_path : src_patterns_path){
      const std::string kPatternVFile = src_pattern_path + "_v.csv",
                        kPatternEFile = src_pattern_path + "_e.csv";
      Pattern pattern;
      auto res = GUNDAM::ReadCSVGraph(pattern, kPatternVFile, 
                                               kPatternEFile);
      if (res < 0){
        std::cout<<"loading gar: "<<src_pattern_path<<" failed! :"<<res<<std::endl;
        return;
      }
      src_pattern_set.emplace_back(std::move(pattern), src_pattern_path);
    }

    std::vector<std::pair<Pattern, std::string>> dst_pattern_set;
    dst_pattern_set.reserve(dst_patterns_path.size());
    for (const auto& dst_pattern_path : dst_patterns_path){
      const std::string kPatternVFile = dst_pattern_path + "_v.csv",
                        kPatternEFile = dst_pattern_path + "_e.csv";
      Pattern pattern;
      auto res = GUNDAM::ReadCSVGraph(pattern, kPatternVFile, 
                                               kPatternEFile);
      if (res < 0){
        std::cout<<"loading gar: "<<dst_pattern_path<<" failed! :"<<res<<std::endl;
        return;
      }
      dst_pattern_set.emplace_back(std::move(pattern), dst_pattern_path);
    }

    for (int i = 0; i < src_pattern_set.size(); i++) {
      const auto& kSrcPattern     = src_pattern_set[i].first;
      const auto& kSrcPatternName = src_pattern_set[i].second;

      bool duplicated = false;
      for (const auto& dst_pattern : dst_pattern_set){
        const auto& kDstPattern    = dst_pattern.first;
        const auto& kDstPatternName= dst_pattern.second;

        if (!GUNDAM::SamePattern(kSrcPattern, 
                                 kDstPattern)){
          continue;
        }
        // these two gars are the same
        if (export_same){
          std::cout<<" pattern: "<<kSrcPatternName
                   <<" is the same as pattern: "<<kDstPatternName<<std::endl;
        }
        duplicated = true;
        break;
      }
      if (!duplicated && !export_same){
        std::cout<<" gar: "<<kSrcPatternName
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
   * @brief Incremental evaluation for PatternExist.
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

#endif  // EXAMPLES_ANALYTICAL_APPS_PATTERN_EXIST_PATTERN_EXIST_H_