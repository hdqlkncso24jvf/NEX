#ifndef EXAMPLES_ANALYTICAL_APPS_GAR_OVERLAP_GAR_OVERLAP_H_
#define EXAMPLES_ANALYTICAL_APPS_GAR_OVERLAP_GAR_OVERLAP_H_

#include <grape/grape.h>

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

#include "gar_overlap/gar_overlap_context.h"

#include "gar/csv_gar.h"
#include "gar/gar.h"
#include "gar/gar_match.h"
#include "gar/gar_imply.h"
#include "gar/gar_config.h"
#include "gar/same_gar.h"
#include "gar/literal_info.h"

#include "util/file.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../timer.h"

namespace grape {

namespace gar_overlap{

// seperated by the last @
inline std::pair<std::string, 
                 std::string> SplitGarFileNameGarName(
           const std::string& gar_file_name_gar_name) {
  size_t found = gar_file_name_gar_name.find_last_of("@");
  return std::pair(gar_file_name_gar_name.substr(0,found),
                   gar_file_name_gar_name.substr(found+1));
}

template<typename GAR>
int LoadGARs(
      const std::set<std::string>& gar_file_set,
               const std::string&  gar_dir,
           std::list<std::pair<std::string, GAR>>& gar_list,
           std::set<std::string>&     gar_with_constant_literal_list,
           std::set<std::string>& gar_rhs_with_constant_literal_list) {

  std::map<std::string, 
           std::set<std::string>> gar_name_set;

  for (const auto& gar_file : gar_file_set) {
    auto [ gar_file_name,
           gar_name ] = SplitGarFileNameGarName(gar_file);
    if (gar_name == ""){
      // is seperated gar
      gar_name_set.emplace(gar_file_name, std::set<std::string>());
      continue;
    }
    // add a new file in it if it is not contained 
    auto [ gar_name_it,
           gar_name_ret ]
         = gar_name_set[gar_file_name].emplace(gar_name);
    if (!gar_name_ret){
      std::cout << "duplicate gar name: " << gar_name << std::endl
                << "\tin file: "     << gar_file_name << std::endl;
      return -1;
    }
  }
  for (const auto& gar_name : gar_name_set){
    const std::string& kGarFileName = gar_name.first;
    if (gar_name.second.empty()) {
      // is seperated gar 
      const std::string kGarName = gar_dir + "/" + kGarFileName;
      const std::string 
            kGarVFile = kGarName + "_v.csv",
            kGarEFile = kGarName + "_e.csv",
            kGarXFile = kGarName + "_x.csv",
            kGarYFile = kGarName + "_y.csv";

      GAR gar;
      auto res = gar::ReadGAR(gar, kGarVFile, 
                                   kGarEFile,
                                   kGarXFile, 
                                   kGarYFile);

      if (res < 0) {
        std::cout << "Read GAR error: " << res << std::endl;
        return res;
      }
      gar_list.emplace_back(kGarFileName, std::move(gar));
      continue;
    }
    // is gar set
    std::vector<GAR> gar_set;
    std::vector<std::string> gar_name_set;
    const std::string kGarName = gar_dir + "/" + kGarFileName;
    const std::string 
      kGarVSetFile = kGarName + "_v_set.csv",
      kGarESetFile = kGarName + "_e_set.csv",
      kGarXSetFile = kGarName + "_x_set.csv",
      kGarYSetFile = kGarName + "_y_set.csv";

    auto res = gar::ReadGARSet(gar_set, 
                               gar_name_set, kGarVSetFile, 
                                             kGarESetFile,
                                             kGarXSetFile, 
                                             kGarYSetFile);
                    
    if (res < 0) {
      std::cout << "Read GAR error: " << res << std::endl;
      return res;
    }
    const auto& kGarNameSet = gar_name.second;

    for (size_t gar_idx = 0; 
                gar_idx < gar_set.size(); 
                gar_idx++) {
      if (kGarNameSet.find(gar_name_set[gar_idx])
       == kGarNameSet.end()){
        // this gar is not contained in the specified rule
        continue;
      }
      // this gar is contained in the specified rule
      gar_list.emplace_back(kGarName + "@" + gar_name_set[gar_idx], 
                                        std::move(gar_set[gar_idx]));
    }
  }
  return 0;
}

template<typename GAR>
void MinimizeGARSet(std::list<std::pair<std::string, GAR>>& gar_list) {
  std::list<std::pair<std::string, GAR>> temp_gar_list;
  std::list<GAR> gar_without_name_list;
  temp_gar_list.swap( gar_list);
  assert(gar_list.size() == 0);

  for (auto& temp_gar_cit : temp_gar_list) {
    /// to check whether dst_gar can be implied from
    /// other rules in gar_list
    assert(gar_without_name_list.size() == gar_list.size());
    const bool can_be_implied = gar::GarImply(gar_without_name_list,
                                         temp_gar_cit.second);
    if (!can_be_implied){
      std::cout << temp_gar_cit.first << std::endl;
      std::cout << "cannot be implied by: " << std::endl;
      for (const auto& gar : gar_list) {
        std::cout << "\t" << gar.first << std::endl;
      }
      gar_list             .emplace_back(temp_gar_cit);
      gar_without_name_list.emplace_back(temp_gar_cit.second);
      continue;
    }
    std::cout << temp_gar_cit.first << std::endl;
    std::cout << "can be implied by: " << std::endl;
    for (const auto& gar : gar_list) {
      std::cout << "\t" << gar.first << std::endl;
    }
  }
  return;
}
  
} // namespace gar_overlap

template <typename FRAG_T>
class GarOverlap : public ParallelAppBase<FRAG_T, GarOverlapContext<FRAG_T>>,
                   public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
                  
  using    VertexIDType = typename GarOverlapContext<FRAG_T>::   VertexIDType;
  using VertexLabelType = typename GarOverlapContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType = typename GarOverlapContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType = typename GarOverlapContext<FRAG_T>::  EdgeLabelType;
  using    DataGraph = typename GarOverlapContext<FRAG_T>::DataGraph;
  using      Pattern = typename GarOverlapContext<FRAG_T>::  Pattern;

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(GarOverlap<FRAG_T>, 
                          GarOverlapContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for GarOverlap.
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

    if (!config["SrcGarDir"]){
      std::cout << "does not specify src gar dir!" 
                << std::endl;
      return;
    }
    const std::string kSrcGarDir
             = config["SrcGarDir"].as<std::string>();
            
    if (!config["DstGarDir"]){
      std::cout << "does not specify dst gar dir!" 
                << std::endl;
      return;
    }
    const std::string kDstGarDir
             = config["DstGarDir"].as<std::string>();

    if (!config["SrcGarFile"]){
      std::cout << "does not specify src gar file!" 
                << std::endl;
      return;
    }
    const std::string kSrcGarFile
             = config["SrcGarFile"].as<std::string>();
    std::ifstream src_gars_file(kSrcGarFile);
            
    if (!config["DstGarFile"]){
      std::cout << "does not specify dst gar dir!" 
                << std::endl;
      return;
    }
    const std::string kDstGarFile
             = config["DstGarFile"].as<std::string>();
    std::ifstream dst_gars_file(kDstGarFile);

    std::cout<<"src gar file: "<<kSrcGarFile<<std::endl
             <<"dst gar file: "<<kDstGarFile<<std::endl;

    std::string minimized_src_gar_dir = "./";
    if (config["MinimizedSrcGarDir"]){
      minimized_src_gar_dir = config["MinimizedSrcGarDir"].as<std::string>();
    }

    std::string minimized_dst_gar_dir = "./";
    if (config["MinimizedDstGarDir"]){
      minimized_dst_gar_dir = config["MinimizedDstGarDir"].as<std::string>();
    }

    // std::cout<<"data graph dir: "<<kDataGraphDir<<std::endl;

    if (!config["ConfBound"]){
      std::cout << "does not specify confidence bound!" 
                << std::endl;
      return;
    }
    const double kConfBound
        = config["ConfBound"].as<double>();

    std::cout<<"confidence bound: "<<kConfBound<<std::endl;

    int supp_bound = 0;
    if (config["SuppBound"]){
      supp_bound = config["SuppBound"].as<int>();
      assert(supp_bound >= 0);
    }
    std::cout<<"support bound: "<<supp_bound<<std::endl;
    
    std::set<std::string> qualified_src_gar_file_set,
                          qualified_dst_gar_file_set;

    std::string str; 

    std::map<std::string, std::tuple<double, int, int>> src_gar_conf_supp;

    std::cout<<"###############"<<std::endl
             <<"## src files ##"<<std::endl
             <<"###############"<<std::endl;
    // src_gars_file
    while (std::getline(src_gars_file, str)) {
      std::string buf;                 // Have a buffer string
      std::stringstream ss(str);       // Insert the string into a stream

      std::vector<std::string> tokens; // Create vector to hold our words

      while (ss >> buf)
        tokens.push_back(buf);

      std::string gar_name = tokens[0];
      double gar_conf = std::stod(tokens[1]);
      
      int gar_xy_supp = std::stoi(tokens[2]);
      int gar_x_supp  = std::stoi(tokens[3]);

      assert(gar_xy_supp <= gar_x_supp);

      src_gar_conf_supp.emplace(gar_name, std::make_tuple(gar_conf, gar_xy_supp, gar_x_supp));

      if (gar_conf < kConfBound){
        continue;
      }
      if (gar_x_supp < supp_bound){
        continue;
      }
      qualified_src_gar_file_set.emplace(gar_name);
    }

    std::cout<<"qualified src gar number: "
             << qualified_src_gar_file_set.size()
             <<std::endl;
             
    std::map<std::string, std::tuple<double, int, int>> dst_gar_conf_supp;
    
    std::cout<<"###############"<<std::endl
             <<"## dst files ##"<<std::endl
             <<"###############"<<std::endl;
    // dst_gars_file
    while (std::getline(dst_gars_file, str)) {
      std::string buf;                 // Have a buffer string
      std::stringstream ss(str);       // Insert the string into a stream

      std::vector<std::string> tokens; // Create vector to hold our words

      while (ss >> buf)
        tokens.push_back(buf);

      std::string gar_name = tokens[0];
      double gar_conf = std::stod(tokens[1]);
      
      int gar_xy_supp = std::stoi(tokens[2]);
      int gar_x_supp  = std::stoi(tokens[3]);

      assert(gar_xy_supp <= gar_x_supp);
      
      dst_gar_conf_supp.emplace(gar_name, std::make_tuple(gar_conf, gar_xy_supp, gar_x_supp));

      if (gar_conf < kConfBound){
        continue;
      }
      if (gar_x_supp < supp_bound){
        continue;
      }
      qualified_dst_gar_file_set.emplace(gar_name);
    }
    std::cout << "qualified dst gar number: "
              <<  qualified_dst_gar_file_set.size()
              << std::endl;

    // using Pattern = GUNDAM::SmallGraph<uint32_t, uint32_t, uint32_t, uint32_t>;
    using Pattern   = GUNDAM::LargeGraph<uint64_t, uint32_t, std::string,
                                         uint64_t, uint32_t, std::string>;
    // using DataGraph = GUNDAM::LargeGraph2<uint64_t, uint32_t, std::string,
    //                                       uint64_t, uint32_t, std::string>;
    // using DataGraph = Pattern;
    using DataGraph = GUNDAM::Graph<
                      GUNDAM::SetVertexIDType<uint64_t>,
                      GUNDAM::SetVertexAttributeStoreType<GUNDAM::AttributeType::kGrouped>,
                      GUNDAM::SetVertexLabelType<uint32_t>,
                      GUNDAM::SetVertexIDContainerType<GUNDAM::ContainerType::Map>,
                      GUNDAM::SetVertexAttributeKeyType<std::string>,
                      GUNDAM::SetEdgeIDType<uint64_t>,
                      GUNDAM::SetEdgeAttributeStoreType<GUNDAM::AttributeType::kGrouped>,
                      GUNDAM::SetEdgeLabelType<uint32_t>,
                      GUNDAM::SetEdgeAttributeKeyType<std::string>>;

    using   PatternVertexID = typename   Pattern::VertexType::IDType;
    using DataGraphVertexID = typename DataGraph::VertexType::IDType;
    using   PatternVertexConstPtr = typename GUNDAM::VertexHandle<  Pattern>::type;
    using DataGraphVertexConstPtr = typename GUNDAM::VertexHandle<DataGraph>::type;
    using GAR = gar::GraphAssociationRule<Pattern, Pattern>;
    using MatchMap = std::map<PatternVertexConstPtr, DataGraphVertexConstPtr>;
    using MatchResultList = std::vector<MatchMap>;

    std::list<std::pair<std::string, GAR>> src_gar_list,
                                           dst_gar_list;

    std::set<std::string>     gar_with_constant_literal_list;
    std::set<std::string> gar_rhs_with_constant_literal_list;
                  
    gar_overlap::LoadGARs(qualified_src_gar_file_set,
                        kSrcGarDir,
                        src_gar_list,
                        gar_with_constant_literal_list,
                    gar_rhs_with_constant_literal_list);
    std::cout<<"qualified src gars loaded"<<std::endl;
    std::ofstream src_gar_with_constant_literal_list_file 
                ("src_gar_with_constant_literal_list_file.txt");
    for (const auto& gar_name : gar_with_constant_literal_list){
      auto src_gar_conf_supp_it = src_gar_conf_supp.find(gar_name);
      src_gar_with_constant_literal_list_file << gar_name 
                                      << "\t" << std::get<0>(src_gar_conf_supp_it->second)
                                      << "\t" << std::get<1>(src_gar_conf_supp_it->second)
                                      << "\t" << std::get<2>(src_gar_conf_supp_it->second)
                                              << std::endl;
    }
    gar_with_constant_literal_list.clear();
    std::ofstream src_gar_rhs_with_constant_literal_list_file 
                ("src_gar_rhs_with_constant_literal_list_file.txt");
    for (const auto& gar_name : gar_rhs_with_constant_literal_list){
      src_gar_rhs_with_constant_literal_list_file << gar_name << std::endl;
    }
    gar_rhs_with_constant_literal_list.clear();
    gar_overlap::LoadGARs(qualified_dst_gar_file_set,
                        kDstGarDir,
                        dst_gar_list,
                        gar_with_constant_literal_list,
                    gar_rhs_with_constant_literal_list);

    std::ofstream dst_gar_with_constant_literal_list_file 
                ("dst_gar_with_constant_literal_list_file.txt");
    for (const auto& gar_name : gar_with_constant_literal_list){
      auto dst_gar_conf_supp_it = dst_gar_conf_supp.find(gar_name);
      dst_gar_with_constant_literal_list_file << gar_name 
                                      << "\t" << std::get<0>(dst_gar_conf_supp_it->second)
                                      << "\t" << std::get<1>(dst_gar_conf_supp_it->second)
                                      << "\t" << std::get<2>(dst_gar_conf_supp_it->second)
                                              << std::endl;
    }
    gar_with_constant_literal_list.clear();
    std::ofstream dst_gar_rhs_with_constant_literal_list_file 
                ("src_gar_rhs_with_constant_literal_list_file.txt");
    for (const auto& gar_name : gar_rhs_with_constant_literal_list){
      dst_gar_rhs_with_constant_literal_list_file << gar_name << std::endl;
    }
    gar_rhs_with_constant_literal_list.clear();

    std::cout<<"qualified dst gars loaded"<<std::endl;
    
    util::Mkdir(minimized_src_gar_dir);
    util::Mkdir(minimized_dst_gar_dir);

    int cover_set_counter = 0;
    gar_overlap::MinimizeGARSet(src_gar_list);
    std::cout<<"src cover set size: "
             << src_gar_list.size()<<std::endl;
    gar_overlap::MinimizeGARSet(dst_gar_list);
    cover_set_counter = 0;
    for (auto& gar : dst_gar_list){
      const std::string rule_name = minimized_dst_gar_dir 
                                + "/dst_rule_"
                                + std::to_string(cover_set_counter);
      const std::string v_file = rule_name + "_v.csv",
                        e_file = rule_name + "_e.csv",
                        x_file = rule_name + "_x.csv",
                        y_file = rule_name + "_y.csv";
      gar::WriteGAR(gar.second, v_file, e_file, x_file, y_file);
      cover_set_counter++;
    }
    std::cout<<"dst cover set size: "
             << dst_gar_list.size()<<std::endl;

    std::list<GAR> src_gar_list_without_name;
    for (const auto& src_gar : src_gar_list) {
      src_gar_list_without_name.emplace_back(src_gar.second);
    }

    // reduce the redundancy in f
    for (auto dst_gar_cit  = dst_gar_list.begin();
              dst_gar_cit != dst_gar_list.end();){
      const bool can_be_implied = gar::GarImply(src_gar_list_without_name,
                                                dst_gar_cit->second);
      if (can_be_implied){
        dst_gar_cit = dst_gar_list.erase(dst_gar_cit);
        continue;
      }
      dst_gar_cit++;
    }

    cover_set_counter = 0;
    for (auto& gar : src_gar_list){
      const std::string rule_name = minimized_src_gar_dir 
                                + "/src_cover_rule_"
                                + std::to_string(cover_set_counter);
      const std::string v_file = rule_name + "_v.csv",
                        e_file = rule_name + "_e.csv",
                        x_file = rule_name + "_x.csv",
                        y_file = rule_name + "_y.csv";
      gar::WriteGAR(gar.second, v_file, e_file, x_file, y_file);
      cover_set_counter++;
    }

    cover_set_counter = 0;
    for (auto& gar : dst_gar_list){
      const std::string rule_name = minimized_dst_gar_dir 
                                + "/dst_cover_rule_"
                                + std::to_string(cover_set_counter);
      const std::string v_file = rule_name + "_v.csv",
                        e_file = rule_name + "_e.csv",
                        x_file = rule_name + "_x.csv",
                        y_file = rule_name + "_y.csv";
      gar::WriteGAR(gar.second, v_file, e_file, x_file, y_file);
      cover_set_counter++;
    }

    std::cout<<"src set size: "<<src_gar_list.size()<<std::endl;
    std::cout<<"dst set size: "<<dst_gar_list.size()<<std::endl;

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
   * @brief Incremental evaluation for GarOverlap.
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

#endif  // EXAMPLES_ANALYTICAL_APPS_GAR_OVERLAP_GAR_OVERLAP_H_