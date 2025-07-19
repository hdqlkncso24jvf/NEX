#ifndef EXAMPLES_ANALYTICAL_APPS_CONSTANT_RULE_ANALYSIS_CONSTANT_RULE_ANALYSIS_H_
#define EXAMPLES_ANALYTICAL_APPS_CONSTANT_RULE_ANALYSIS_CONSTANT_RULE_ANALYSIS_H_

#include <grape/grape.h>
#include <omp.h>

#include <functional>

#include "gundam/graph_type/graph.h"
#include "gundam/graph_type/small_graph.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/large_graph2.h"

#include "gundam/match/match.h"
#include "gundam/match/matchresult.h"

#include "gundam/io/csvgraph.h"
#include "gundam/io/csvgraph.h"

#include "gar/gar_supp.h"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "constant_rule_analysis/constant_rule_analysis_context.h"

#include "gar/csv_gar.h"
#include "gar/gar.h"
#include "gar/literal_stand_alone_info.h"
#include "gar/gar_config.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../timer.h"

namespace grape {

namespace _constant_rule_analysis{

template <typename   Pattern,
          typename DataGraph>
Pattern RhsPattern(const gar::GraphAssociationRule<Pattern, DataGraph>& gar){
  
  Pattern rhs_pattern(gar.pattern());
  std::set<typename GUNDAM::VertexHandle<Pattern>::type> gar_y_vertex_ptr_set;

  for (const auto& y_literal : gar.y_literal_set()){
    y_literal->CalPivot(gar_y_vertex_ptr_set);
  }
  std::set<typename Pattern::VertexType::IDType> erase_id_set;
  // erase all vertex that is not contained in y literal
  for (auto vertex_it = rhs_pattern.VertexBegin();
           !vertex_it.IsDone(); 
            vertex_it++){
    bool in_y_literals = false;
    for (const auto& ptr_it : gar_y_vertex_ptr_set){
      if (ptr_it->id() == vertex_it->id()){
        in_y_literals = true;
        break;
      }
    }
    if (in_y_literals){
      continue;
    }
    erase_id_set.emplace(vertex_it->id());
  }
  for (const auto& erase_id : erase_id_set){
    rhs_pattern.EraseVertex(erase_id);
  }
  return rhs_pattern;
}

}; // namespace _constant_rule_analysis

template <typename FRAG_T>
class ConstantRuleAnalysis : public ParallelAppBase<FRAG_T, ConstantRuleAnalysisContext<FRAG_T>>,
                             public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
  using    VertexIDType = typename ConstantRuleAnalysisContext<FRAG_T>::   VertexIDType;
  using VertexLabelType = typename ConstantRuleAnalysisContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType = typename ConstantRuleAnalysisContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType = typename ConstantRuleAnalysisContext<FRAG_T>::  EdgeLabelType;
  using    DataGraph    = typename ConstantRuleAnalysisContext<FRAG_T>::DataGraph;
  using      Pattern    = typename ConstantRuleAnalysisContext<FRAG_T>::  Pattern;

  // seperated by the last @
  inline std::pair<std::string, 
                   std::string> SplitGarFileNameGarName(
             const std::string& gar_file_name_gar_name) const {
    size_t found = gar_file_name_gar_name.find_last_of("@");
    return std::pair(gar_file_name_gar_name.substr(0,found),
                     gar_file_name_gar_name.substr(found+1));
  }

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(ConstantRuleAnalysis<FRAG_T>, 
                          ConstantRuleAnalysisContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for ConstantRuleAnalysis.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void PEval(const fragment_t& frag, 
                    context_t& ctx,
            message_manager_t& messages) {

    using GarType = gar::GraphAssociationRule<Pattern, DataGraph>;

    using LiteralStandAloneInfoType = gar::LiteralStandAloneInfo<Pattern, DataGraph>;

    using   PatternVertexHandle = typename GUNDAM::VertexHandle<  Pattern>::type;
    using DataGraphVertexHandle = typename GUNDAM::VertexHandle<DataGraph>::type;

    messages.InitChannels(thread_num());
    auto begin = timer();
    timer_next("run constant_rule_analysis");

    std::cout<<"yaml file: "<<ctx.yaml_file_<<std::endl;

    YAML::Node config = YAML::LoadFile(ctx.yaml_file_);

    if (!config["GarFile"]){
      std::cout<<"cannot get gar file"<<std::endl;
      return;
    }
    const std::string kGarFile = config["GarFile"].as<std::string>();

    std::ifstream gars_file(kGarFile);

    std::map<std::string,
             std::pair<int, int>> gar_supp;

    std::map<std::string,
             std::map<std::string, std::pair<int, int>>> gar_supp_set;

    std::string str;

    while (std::getline(gars_file, str)) {
      std::string buf;                 // Have a buffer string
      std::stringstream ss(str);       // Insert the string into a stream

      std::vector<std::string> tokens; // Create vector to hold our words

      while (ss >> buf)
        tokens.emplace_back(buf);

      const std::string kGarName = tokens[0];

      int supp_xy = std::stoi(tokens[2]),
          supp_x  = std::stoi(tokens[3]);
      
      auto [ gar_file_name,
             gar_name ] = SplitGarFileNameGarName(kGarName);
      if (gar_name == "") {
        // is seperated gar
        auto [gar_supp_it,
              gar_supp_ret]
            = gar_supp.emplace(gar_file_name, std::pair(supp_xy, supp_x));
        if (!gar_supp_ret) {
          std::cout << "duplicated seperated gar name: " << gar_file_name
                    << std::endl;
          return;
        }
        continue;
      }
      // is gar set, if this file does not exist before add it 
      // into the map
      auto [ gar_name_it,
             gar_name_ret ] = gar_supp_set[gar_file_name].emplace(gar_name, std::pair(supp_xy, supp_x));
      if (!gar_name_ret){
        std::cout << "duplicated gar name: " << gar_name << std::endl
                  << " in gar file: "   << gar_file_name << std::endl;
        return;
      }
    }

    std::vector<std::tuple<std::string, GarType, int, int>> selected_gar_set;

    for (const auto& gar_supp : gar_supp_set){
      assert(!gar_supp.second.empty());

      const std::string& kGarFileName = gar_supp.first;

      const auto& kGarSuppSet = gar_supp.second;

      const std::string kGarVSetFile = kGarFileName + "_v_set.csv",
                        kGarESetFile = kGarFileName + "_e_set.csv",
                        kGarXSetFile = kGarFileName + "_x_set.csv",
                        kGarYSetFile = kGarFileName + "_y_set.csv";

      std::vector<GarType> gar_set;
      std::vector<std::string> gar_name_set;
                            
      auto read_gar_res = gar::ReadGARSet(gar_set, gar_name_set,
                                         kGarVSetFile, 
                                         kGarESetFile, 
                                         kGarXSetFile, 
                                         kGarYSetFile);

      assert(gar_set.size() == gar_name_set.size());

      for (size_t gar_idx = 0; gar_idx < gar_set.size(); gar_idx++) {
        const std::string& kGarName = gar_name_set[gar_idx];
        auto gar_supp_set_it = kGarSuppSet.find(kGarName);
        if (gar_supp_set_it == kGarSuppSet.end()){
          // this gar is not contained in the kGarSuppSet
          continue;
        }

        std::cout << kGarFileName + "@" + kGarName << std::endl;
        assert( gar_set[gar_idx].y_literal_set().Count() == 1);

        // this gar is contained in kGarSuppSet
        selected_gar_set.emplace_back(kGarFileName + "@" + kGarName, 
                                       gar_set[gar_idx], 
                                       gar_supp_set_it->second.first, 
                                       gar_supp_set_it->second.second);
      }
    }

    std::cout << "selected_gar_set.size(): "
              <<  selected_gar_set.size()
              << std::endl;

    std::vector<std::tuple<std::string, GarType, int, int>> rhs_constant_gar_set;

    std::cout << "##########################################" << std::endl
              << "# begin to collect gar with constant rhs #" << std::endl
              << "##########################################" << std::endl;
    for (auto& selected_gar : selected_gar_set) {
      if ((*(std::get<1>(selected_gar).y_literal_set().begin()))->type() 
         == gar::LiteralType::kConstantLiteral){
        rhs_constant_gar_set.emplace_back(selected_gar);
      }
    }

    std::vector<std::tuple<std::string, GarType, int, int, int, int>> lhs_constant_gar_set;

    std::cout << "##########################################" << std::endl
              << "# begin to collect gar with constant lhs #" << std::endl
              << "##########################################" << std::endl;
    for (const auto& selected_gar : selected_gar_set) {
      GarType temp_selected_gar = std::get<1>(selected_gar);

      bool lhs_has_constant_literal = false;

      for (auto x_literal_ptr_it  = temp_selected_gar.x_literal_set().begin();
                x_literal_ptr_it != temp_selected_gar.x_literal_set(). end ();) {
        if ((*x_literal_ptr_it)->type() == gar::LiteralType::kConstantLiteral) {
          lhs_has_constant_literal = true;
          x_literal_ptr_it = temp_selected_gar.x_literal_set()
                                              .Erase(x_literal_ptr_it);
          continue;
        }
        x_literal_ptr_it++;
      }

      if (!lhs_has_constant_literal){
        continue;
      }

      bool found_lhs_without_constant_literal_gar = false;
      for (const auto& lhs_without_constant_literal_gar : selected_gar_set) {
        if (gar::SameGar(temp_selected_gar, 
                  std::get<1>(lhs_without_constant_literal_gar))){
          found_lhs_without_constant_literal_gar = true;
          lhs_constant_gar_set.emplace_back(std::get<0>(selected_gar),
                                            std::get<1>(selected_gar),
                                            std::get<2>(selected_gar),
                                            std::get<3>(selected_gar),
                                            std::get<2>(lhs_without_constant_literal_gar),
                                            std::get<3>(lhs_without_constant_literal_gar));
          break;
        }
      }
      if (found_lhs_without_constant_literal_gar){
        continue;
      }
      lhs_constant_gar_set.emplace_back(std::get<0>(selected_gar),
                                        std::get<1>(selected_gar),
                                        std::get<2>(selected_gar),
                                        std::get<3>(selected_gar), 0, 0);
    }

    std::cout << "number of gar with constant rhs: " 
              << rhs_constant_gar_set.size() << std::endl;

    for (auto& rhs_constant_gar : rhs_constant_gar_set) {
      std::cout << std::get<0>(rhs_constant_gar) 
        << "\t" << std::get<2>(rhs_constant_gar) 
        << "\t" << std::get<3>(rhs_constant_gar) << std::endl;
    }

    std::cout << "number of gar with constant lhs: " 
              << lhs_constant_gar_set.size() << std::endl;

    for (auto& lhs_constant_gar : lhs_constant_gar_set) {
      std::cout << std::get<0>(lhs_constant_gar) 
        << "\t" << std::get<2>(lhs_constant_gar) 
        << "\t" << std::get<3>(lhs_constant_gar) 
        << "\t" << std::get<4>(lhs_constant_gar) 
        << "\t" << std::get<5>(lhs_constant_gar) << std::endl;
    }

    // std::vector<std::tuple<GarType, std::string, int, int>> rhs_constant_gar_set;

    // for (size_t gar_idx = 0; gar_idx < gar_set.size(); gar_idx++) {
    //   GarType&    gar      = gar_set     [gar_idx];
    //   std::string gar_name = gar_name_set[gar_idx];

    //   size_t supp_xy = ,
    //          supp_x  = ;
      
    //   auto gar_rhs_type = (*(all_gar_set[gar_idx].y_literal_set().begin()))->type();

    //   if (gar_rhs_type == gar::LiteralType::kConstantLiteral) {
    //     rhs_constant_gar_set.emplace_back(gar, gar_name, supp_xy, supp_x);
    //   }
    // }

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
   * @brief Incremental evaluation for ConstantRuleAnalysis.
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

#endif  // EXAMPLES_ANALYTICAL_APPS_CONSTANT_RULE_ANALYSIS_CONSTANT_RULE_ANALYSIS_H_