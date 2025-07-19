#ifndef EXAMPLES_ANALYTICAL_APPS_GAR_ANALYSIS_GAR_ANALYSIS_H_
#define EXAMPLES_ANALYTICAL_APPS_GAR_ANALYSIS_GAR_ANALYSIS_H_

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

#include "gar/gar_supp.h"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "gar_analysis/gar_analysis_context.h"

#include "gar/csv_gar.h"
#include "gar/gar.h"
#include "gar/literal_stand_alone_info.h"
#include "gar/gar_config.h"

#include "gundam/tool/max_id.h"

#include "util/log.h"

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "../timer.h"

namespace grape {

namespace _gar_analysis{

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


template <typename   Pattern,
          typename DataGraph>
inline gar::GraphAssociationRule<Pattern, DataGraph> 
  AddRhsToLhs(const gar::GraphAssociationRule<Pattern, DataGraph>& gar){
  using GarType = gar::GraphAssociationRule<Pattern, DataGraph>;
  using LiteralInfoType = gar::LiteralInfo<Pattern, DataGraph>;
  assert(gar.y_literal_set().Count() == 1);

  if ((*gar.y_literal_set().begin())->type() 
     == gar::LiteralType::kEdgeLiteral) {
    Pattern pattern_with_rhs_edge_literal(gar.pattern());
    LiteralInfoType literal_info = (*gar.y_literal_set().begin())->info();

    using EdgeIDType = typename GUNDAM::EdgeID<Pattern>::type;

    EdgeIDType max_edge_id = GUNDAM::MaxEdgeId(pattern_with_rhs_edge_literal);
    max_edge_id++;

    auto [ edge_handle,
           edge_ret ] = pattern_with_rhs_edge_literal.AddEdge(literal_info.x_id(),
                                                              literal_info.y_id(),
                                                              literal_info.edge_label(),
                                                              max_edge_id);
    assert(edge_ret); // should added successfully

    GarType gar_with_rhs_literal(pattern_with_rhs_edge_literal);
    
    for (const auto& x_literal_ptr : gar.x_literal_set()) {
      gar_with_rhs_literal.AddX(x_literal_ptr->info());
    }
    for (const auto& y_literal_ptr : gar.y_literal_set()) {
      gar_with_rhs_literal.AddY(y_literal_ptr->info());
    }
    return gar_with_rhs_literal;
  }
  GarType gar_with_rhs_literal(gar);
  for (const auto& y_literal_ptr : gar.y_literal_set()) {
    gar_with_rhs_literal.AddX(y_literal_ptr->info());
  }
  assert(gar_with_rhs_literal.x_literal_set().Count() 
                       == gar.x_literal_set().Count() 
                        + gar.y_literal_set().Count());
  return gar_with_rhs_literal;
}

}; // namespace _gar_analysis

template <typename FRAG_T>
class GarAnalysis : public ParallelAppBase<FRAG_T, GarAnalysisContext<FRAG_T>>,
                    public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
  using    VertexIDType = typename GarAnalysisContext<FRAG_T>::   VertexIDType;
  using VertexLabelType = typename GarAnalysisContext<FRAG_T>::VertexLabelType;
  using      EdgeIDType = typename GarAnalysisContext<FRAG_T>::     EdgeIDType;
  using   EdgeLabelType = typename GarAnalysisContext<FRAG_T>::  EdgeLabelType;
  using    DataGraph = typename GarAnalysisContext<FRAG_T>::DataGraph;
  using      Pattern = typename GarAnalysisContext<FRAG_T>::  Pattern;

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(GarAnalysis<FRAG_T>, 
                          GarAnalysisContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for GarAnalysis.
   *
   * @param frag
   * @param ctx
   * @param messages
   */
  void PEval(const fragment_t& frag, 
                    context_t& ctx,
            message_manager_t& messages) {

    using GarType = gar::GraphAssociationRule<Pattern, DataGraph>;

    using   PatternVertexHandle = typename GUNDAM::VertexHandle<  Pattern>::type;
    using DataGraphVertexHandle = typename GUNDAM::VertexHandle<DataGraph>::type;

    messages.InitChannels(thread_num());
    auto begin = timer();
    timer_next("run gar_analysis");

    std::cout<<"yaml file: "<<ctx.yaml_file_<<std::endl;

    YAML::Node config = YAML::LoadFile(ctx.yaml_file_);
    
    if (!config["DataGraphPath"]){
      std::cout<<"cannot get data graph path"<<std::endl;
      return;
    }
    YAML::Node data_graph_path_config 
    = config["DataGraphPath"];

    if (!data_graph_path_config["VFile"]){
      std::cout<<"cannot get data graph v file"<<std::endl;
      return;
    }
    const std::string kDataGraphPathVFile 
          = data_graph_path_config["VFile"].as<std::string>();

    if (!data_graph_path_config["EFile"]){
      std::cout<<"cannot get data graph e file"<<std::endl;
      return;
    }
    const std::string kDataGraphPathEFile 
          = data_graph_path_config["EFile"].as<std::string>();

    double time_limit = -1.0,
           time_limit_per_supp = -1.0;
    if (config["TimeLimit"]){
      time_limit = config["TimeLimit"].as<double>();
    }
    
    if (config["TimeLimitPerSupp"]){
      time_limit_per_supp = config["TimeLimitPerSupp"].as<double>();
    }

    bool count_matched_rhs_edge = false;
    if (config["CountMatchRhsEdge"]){
      count_matched_rhs_edge = config["CountMatchRhsEdge"].as<bool>();
    }

    int support_bound = 1000;

    bool collect_supp_xy_only = false;
    if (config["CollectSuppXYOnly"]) {
      collect_supp_xy_only = config["CollectSuppXYOnly"].as<bool>();
    }

    DataGraph data_graph;
    
    GUNDAM::ReadCSVGraph(data_graph, 
                        kDataGraphPathVFile,
                        kDataGraphPathEFile);

    if (!config["GARPath"]){
      std::cout<<"cannot get gar path"<<std::endl;
      return;
    }
    YAML::Node gar_path_set_config 
     = config["GARPath"];

    using SupportIDSetType = std::set<std::vector<VertexIDType>>;

    const std::string kVFilePosfix = "_v.csv";
    const std::string kVSetFilePosfix = "_v_set.csv";

    using LiteralStandAloneInfoType = gar::LiteralStandAloneInfo<Pattern, DataGraph>;

    // first gather all gars, classify them into classes 
    // according to their rhs literal
    std::map<LiteralStandAloneInfoType, std::set<std::string>> gar_stand_alone_info_set;
    for (int i = 0; i < gar_path_set_config.size(); i++){
      YAML::Node gar_path_config = gar_path_set_config[i];

      if (gar_path_config["VSetFile"]
       || gar_path_config["ESetFile"]
       || gar_path_config["XSetFile"]
       || gar_path_config["YSetFile"]
       || gar_path_config["GarIdSet"]){
        if (!gar_path_config["VSetFile"]){
          std::cout<<"cannot get gar v set file"<<std::endl;
          return;
        }
        const std::string kGarPathVSetFile
               = gar_path_config["VSetFile"].as<std::string>();

        if (!gar_path_config["ESetFile"]){
          std::cout<<"cannot get gar e set file"<<std::endl;
          return;
        }
        const std::string kGarPathESetFile
               = gar_path_config["ESetFile"].as<std::string>();

        if (!gar_path_config["XSetFile"]){
          std::cout<<"cannot get gar x set file"<<std::endl;
          return;
        }
        const std::string kGarPathXSetFile 
               = gar_path_config["XSetFile"].as<std::string>();

        if (!gar_path_config["YSetFile"]){
          std::cout<<"cannot get gar y set file"<<std::endl;
          return;
        }
        const std::string kGarPathYSetFile
               = gar_path_config["YSetFile"].as<std::string>();

        std::set<std::string> selected_gar_id_set;

        if (gar_path_config["GarIdSet"]){
          YAML::Node gar_id_config
                  = gar_path_config["GarIdSet"];

          for (int i = 0; i < gar_id_config.size(); i++) {
            const std::string kGarId = gar_id_config[i].as<std::string>();
            auto [selected_gar_id_set_it,
                  selected_gar_id_set_ret] 
                = selected_gar_id_set.emplace(kGarId);
            if (!selected_gar_id_set_ret) {
              std::cout << "duplicated gar id: " << kGarId << " in : " << std::endl
                        << "\tgar_v_set: " << kGarPathVSetFile << std::endl
                        << "\tgar_e_set: " << kGarPathESetFile << std::endl
                        << "\tgar_x_set: " << kGarPathXSetFile << std::endl
                        << "\tgar_y_set: " << kGarPathYSetFile << std::endl;
              return;
            }
          }
          if (selected_gar_id_set.empty()){
            std::cout << "specified empty gar id set" << std::endl;
            return;
          }
        }

        const bool kSpecifiedGarIdSet = !selected_gar_id_set.empty();

        const std::string kGarSetPathName
                        = kGarPathVSetFile.substr(0,
                          kGarPathVSetFile.length()
                               - kVSetFilePosfix.size());

        std::vector<GarType> all_gar_set;
        std::vector<std::string> all_gar_name_set;
        gar::ReadGARSet(all_gar_set,
                        all_gar_name_set, kGarPathVSetFile,
                                          kGarPathESetFile,
                                          kGarPathXSetFile,
                                          kGarPathYSetFile);

        assert(all_gar_set.size() == all_gar_name_set.size());

        for (size_t gar_idx = 0; gar_idx < all_gar_set.size(); gar_idx++) {
          if (kSpecifiedGarIdSet) {
            // to find whether this id is contained in the
            // set of specified gar_id set
            auto selected_gar_id_set_it 
               = selected_gar_id_set.find(all_gar_name_set[gar_idx]);
            if (selected_gar_id_set_it == selected_gar_id_set.end()){
              // is not contained in the specified gar set
              continue;
            }
            // this gar needs to be considered
            selected_gar_id_set.erase(selected_gar_id_set_it);
          }
          
          LiteralStandAloneInfoType gar_rhs_type = (*(all_gar_set[gar_idx].y_literal_set().begin()))->stand_alone_info();
                            
          auto [gar_stand_alone_info_set_it,
                gar_stand_alone_info_set_ret] 
              = gar_stand_alone_info_set.emplace(gar_rhs_type, std::set<std::string>());

          assert(gar_stand_alone_info_set_it != gar_stand_alone_info_set.end() );
          
          gar_stand_alone_info_set_it->second.emplace(kGarSetPathName + "@" + all_gar_name_set[gar_idx]);
        }
      }
      else {
        if (!gar_path_config["VFile"]){
          std::cout<<"cannot get gar v file"<<std::endl;
          return;
        }
        const std::string kGarPathVFile 
               = gar_path_config["VFile"].as<std::string>();

        if (!gar_path_config["EFile"]){
          std::cout<<"cannot get gar e file"<<std::endl;
          return;
        }
        const std::string kGarPathEFile 
               = gar_path_config["EFile"].as<std::string>();

        if (!gar_path_config["XFile"]){
          std::cout<<"cannot get gar x file"<<std::endl;
          return;
        }
        const std::string kGarPathXFile 
              = gar_path_config["XFile"].as<std::string>();

        if (!gar_path_config["YFile"]){
          std::cout<<"cannot get gar y file"<<std::endl;
          return;
        }
        const std::string kGarPathYFile 
              = gar_path_config["YFile"].as<std::string>();

        GarType gar;
                            
        auto read_gar_res = gar::ReadGAR(gar, kGarPathVFile, 
                                              kGarPathEFile, 
                                              kGarPathXFile, 
                                              kGarPathYFile);
              
        if (read_gar_res < 0){
          std::cout<<"read gar error!"<<std::endl;
          return;
        }

        LiteralStandAloneInfoType gar_rhs_type = (*(gar.y_literal_set().begin()))->stand_alone_info();
        
        const std::string kGarName
                        = kGarPathVFile.substr(0,
                          kGarPathVFile.length()
                              - kVFilePosfix.size());
                          
        auto [gar_stand_alone_info_set_it,
              gar_stand_alone_info_set_ret] 
            = gar_stand_alone_info_set.emplace(gar_rhs_type, std::set<std::string>());

        assert(gar_stand_alone_info_set_it != gar_stand_alone_info_set.end() );
        
        gar_stand_alone_info_set_it->second.emplace(kGarName);
      }
    }

    int gar_total_number = 0;
    for (const auto& gar_stand_alone_info : gar_stand_alone_info_set){
      std::string gar_stand_alone_info_str;
      gar_stand_alone_info_str << gar_stand_alone_info.first;
      std::cout<< "gar class: " << gar_stand_alone_info_str << std::endl;
      std::cout<< "gar number: " << gar_stand_alone_info.second.size() << std::endl;
      for (const auto& gar_name : gar_stand_alone_info.second) {
        std::cout << "\t" << gar_name << std::endl;
      }
      gar_total_number += gar_stand_alone_info.second.size();
      std::cout << std::endl;
    }

    std::cout << "gar_stand_alone_info_set.size(): "
              <<  gar_stand_alone_info_set.size()
              << std::endl;

    std::cout << "gar_total_number: "
              <<  gar_total_number
              << std::endl;

    std::map<std::string, 
             std::pair<DataGraph, SupportIDSetType>> sampled_graph_set;

    constexpr uint8_t kMatchedRuleNameIdx = 0,
              kContainedInSuppRuleNameIdx = 1,
       kMatchedContainedInSuppRuleNameIdx = 2,
    kNotMatchedContainedInSuppRuleNameIdx = 3;

    std::map <std::string, 
              std::tuple<std::map<std::string,
                                  std::pair<int, int>>, //          matched                              rules number
                         std::map<std::string, int>,    //                      contained in support set rules number
                         std::map<std::string, int>,    //          matched and contained in support set rules number
                         std::map<std::string, int>>    // does not matched and contained in support set rules number
             > sampled_graph_statistic;

    std::map<std::string, std::set<std::string>> graph_package_set;

    if (config["SampledGraphPath"]){
      YAML::Node sampled_graph_path_set_config 
       = config["SampledGraphPath"];
      for (int i = 0; i < sampled_graph_path_set_config.size(); i++) {
        YAML::Node sampled_graph_path_config 
                 = sampled_graph_path_set_config[i];

        if (sampled_graph_path_config["GraphPackage"]){
          YAML::Node graph_package_config = sampled_graph_path_config["GraphPackage"];

          if (!graph_package_config["PackageName"]){
            std::cout << "The name of the graph package is not specified" 
                      << std::endl;
            return;
          }
          const std::string kGraphPackageName 
          = graph_package_config["PackageName"].as<std::string>();

          std::cout << "kGraphPackageName: "
                    <<  kGraphPackageName << std::endl;

          auto [graph_package_set_it,
                graph_package_set_ret]
              = graph_package_set.emplace(kGraphPackageName, std::set<std::string>());

          assert(graph_package_set_ret);

          auto& graph_package_sampled_graph_set 
              = graph_package_set_it->second;

          if (!graph_package_config["SampledGraphSet"]) {
            std::cout << "The set of sampled graph for the graph package is not specified" 
                      << std::endl;
            return;
          }
          YAML::Node sampled_graph_package_config 
                           = graph_package_config["SampledGraphSet"];

          for (int i = 0; i < sampled_graph_package_config.size(); i++) {
            YAML::Node sampled_graph_path_config 
                     = sampled_graph_package_config[i];
            
            if (!sampled_graph_path_config["VFile"]){
              std::cout<<"cannot get data graph v file"<<std::endl;
              return;
            }
            const std::string kSampledGraphPathVFile 
                  = sampled_graph_path_config["VFile"].as<std::string>();

            if (!sampled_graph_path_config["EFile"]){
              std::cout<<"cannot get data graph e file"<<std::endl;
              return;
            }
            const std::string kSampledGraphPathEFile 
                  = sampled_graph_path_config["EFile"].as<std::string>();

            const std::string kSampledGraphName
                            = kSampledGraphPathVFile.substr(0,
                              kSampledGraphPathVFile.length()
                                            - kVFilePosfix.size());

            auto [graph_package_sampled_graph_set_it,
                  graph_package_sampled_graph_set_ret]
                = graph_package_sampled_graph_set.emplace(kSampledGraphName);
            if (!graph_package_sampled_graph_set_ret) {
              std::cout << "duplicate sampled graph: " << kSampledGraphName << std::endl
                        << " in graph package: " << kGraphPackageName
                        << std::endl;
              return;
            }

            auto [sampled_graph_set_it,
                  sampled_graph_set_ret] 
                = sampled_graph_set.emplace(kSampledGraphName, 
                                      std::pair(DataGraph(), 
                                        SupportIDSetType()));

            if (!sampled_graph_set_ret) {
              std::cout << "duplicate sampled graph: " 
                        << kSampledGraphName << std::endl;
              continue;
            }

            auto [sampled_graph_statistic_it,
                  sampled_graph_statistic_ret]
                = sampled_graph_statistic.emplace(kSampledGraphName,
                                                  std::tuple(std::map<std::string, 
                                                                      std::pair<int, int>>(),
                                                             std::map<std::string, int>(),
                                                             std::map<std::string, int>(),
                                                             std::map<std::string, int>()));
            // should added successfully
            assert(sampled_graph_statistic_ret);

            auto& sampled_graph 
                = sampled_graph_set_it->second.first;

            GUNDAM::ReadCSVGraph(sampled_graph, 
                                kSampledGraphPathVFile,
                                kSampledGraphPathEFile);

            auto& sampled_graph_support_id_set
                = sampled_graph_set_it->second.second;

            if (sampled_graph_path_config["SuppFile"]){

              const std::string kSuppFile = sampled_graph_path_config["SuppFile"].as<std::string>();

              std::ifstream supp_file(kSuppFile);

              while (supp_file) {
                std::string s;
                if (!std::getline( supp_file, s ))
                  break;

                std::istringstream ss( s );
                std::vector <std::string> record;
                std::string buf; 
                while (ss >> buf)
                  record.emplace_back(buf);

                std::vector <VertexIDType> support_id;
                for (const auto& id_str : record){
                  support_id.emplace_back(std::stoi(id_str));
                }
                assert(record.size() ==  support_id.size());
                auto [ sampled_graph_support_id_set_it,
                      sampled_graph_support_id_set_ret ] 
                    = sampled_graph_support_id_set.emplace(support_id);
                assert(sampled_graph_support_id_set_ret);
              }
              assert(!sampled_graph_support_id_set.empty());
            }
          }
          continue;
        }

        if (!sampled_graph_path_config["VFile"]){
          std::cout<<"cannot get data graph v file"<<std::endl;
          return;
        }
        const std::string kSampledGraphPathVFile 
              = sampled_graph_path_config["VFile"].as<std::string>();

        if (!sampled_graph_path_config["EFile"]){
          std::cout<<"cannot get data graph e file"<<std::endl;
          return;
        }
        const std::string kSampledGraphPathEFile 
              = sampled_graph_path_config["EFile"].as<std::string>();

        const std::string kSampledGraphName
                        = kSampledGraphPathVFile.substr(0,
                          kSampledGraphPathVFile.length()
                                        - kVFilePosfix.size());

        auto [sampled_graph_set_it,
              sampled_graph_set_ret] 
            = sampled_graph_set.emplace(kSampledGraphName, 
                                  std::pair(DataGraph(), 
                                     SupportIDSetType()));
                                     
        if (!sampled_graph_set_ret) {
          std::cout << "duplicate sampled graph: " << kSampledGraphName << std::endl;
          continue;
        }

        auto [sampled_graph_statistic_it,
              sampled_graph_statistic_ret]
            = sampled_graph_statistic.emplace(kSampledGraphName,
                                              std::tuple(std::map<std::string, 
                                                                  std::pair<int, int>>(),
                                                         std::map<std::string, int>(),
                                                         std::map<std::string, int>(),
                                                         std::map<std::string, int>()));
        // should added successfully
        assert(sampled_graph_statistic_ret);

        auto& sampled_graph 
            = sampled_graph_set_it->second.first;

        GUNDAM::ReadCSVGraph(sampled_graph, 
                            kSampledGraphPathVFile,
                            kSampledGraphPathEFile);

        auto& sampled_graph_support_id_set 
            = sampled_graph_set_it->second.second;

        if (sampled_graph_path_config["SuppFile"]){

          const std::string kSuppFile = sampled_graph_path_config["SuppFile"].as<std::string>();

          std::ifstream supp_file(kSuppFile);

          while (supp_file) {
            std::string s;
            if (!std::getline( supp_file, s ))
              break;

            std::istringstream ss( s );
            std::vector <std::string> record;
            std::string buf; 
            while (ss >> buf)
              record.emplace_back(buf);

            std::vector <VertexIDType> support_id;
            for (const auto& id_str : record){
              support_id.emplace_back(std::stoi(id_str));
            }
            assert(record.size() ==  support_id.size());
            auto [ sampled_graph_support_id_set_it,
                   sampled_graph_support_id_set_ret ] 
                 = sampled_graph_support_id_set.emplace(support_id);
            assert(sampled_graph_support_id_set_ret);
          }
          assert(!sampled_graph_support_id_set.empty());
        }
      }
    }

    if (!config["LogFile"]) {
      std::cout << "does not specified output log file" << std::endl;
      return;
    }
    const std::string kLogFile = config["LogFile"].as<std::string>();

    std::set<std::string> gar_name_set;

    std::map<std::string, std::pair<int, int>> gar_supp_on_original_graph;

    std::vector<std::pair<GarType, std::string>> gar_set;

    for (int i = 0; i < gar_path_set_config.size(); i++) {
      YAML::Node gar_path_config = gar_path_set_config[i];

      if (gar_path_config["VSetFile"]
       || gar_path_config["ESetFile"]
       || gar_path_config["XSetFile"]
       || gar_path_config["YSetFile"]
       || gar_path_config["GarIdSet"]){
        if (!gar_path_config["VSetFile"]){
          std::cout<<"cannot get gar v set file"<<std::endl;
          return;
        }
        const std::string kGarPathVSetFile
               = gar_path_config["VSetFile"].as<std::string>();

        if (!gar_path_config["ESetFile"]){
          std::cout<<"cannot get gar e set file"<<std::endl;
          return;
        }
        const std::string kGarPathESetFile
               = gar_path_config["ESetFile"].as<std::string>();

        if (!gar_path_config["XSetFile"]){
          std::cout<<"cannot get gar x set file"<<std::endl;
          return;
        }
        const std::string kGarPathXSetFile 
               = gar_path_config["XSetFile"].as<std::string>();

        if (!gar_path_config["YSetFile"]){
          std::cout<<"cannot get gar y set file"<<std::endl;
          return;
        }
        const std::string kGarPathYSetFile
               = gar_path_config["YSetFile"].as<std::string>();


        if (!gar_path_config["GarIdSet"]){
          std::cout<<"cannot get gar id set"<<std::endl;
          return;
        }
        YAML::Node gar_id_config
                 = gar_path_config["GarIdSet"];

        std::set<std::string> selected_gar_id_set;

        for (int i = 0; i < gar_id_config.size(); i++) {
          const std::string kGarId = gar_id_config[i].as<std::string>();
          auto [selected_gar_id_set_it,
                selected_gar_id_set_ret] 
              = selected_gar_id_set.emplace(kGarId);
          if (!selected_gar_id_set_ret) {
            std::cout << "duplicated gar id: " << kGarId << std::endl
                      << "\tgar_v_set: " << kGarPathVSetFile << std::endl
                      << "\tgar_e_set: " << kGarPathESetFile << std::endl
                      << "\tgar_x_set: " << kGarPathXSetFile << std::endl
                      << "\tgar_y_set: " << kGarPathYSetFile << std::endl;
            return;
          }
        }

        if (selected_gar_id_set.empty()){
          std::cout << "specified empty gar id set" << std::endl;
          return;
        }

        const std::string kGarSetPathName
                        = kGarPathVSetFile.substr(0,
                          kGarPathVSetFile.length()
                               - kVSetFilePosfix.size());

        std::vector<GarType> all_gar_set;
        std::vector<std::string> all_gar_name_set;
        gar::ReadGARSet(all_gar_set,
                        all_gar_name_set, kGarPathVSetFile,
                                          kGarPathESetFile,
                                          kGarPathXSetFile,
                                          kGarPathYSetFile);

        assert(all_gar_set.size() == all_gar_name_set.size());

        for (size_t gar_idx = 0; gar_idx < all_gar_set.size(); gar_idx++) {
          auto selected_gar_id_set_it 
             = selected_gar_id_set.find(all_gar_name_set[gar_idx]);
          if (selected_gar_id_set_it == selected_gar_id_set.end()){
            continue;
          }
          // this gar needs to be considered
          selected_gar_id_set.erase(selected_gar_id_set_it);
          gar_set.emplace_back(all_gar_set     [gar_idx],
                               all_gar_name_set[gar_idx]);
          
          auto [ gar_name_set_it,
                 gar_name_set_ret ]
               = gar_name_set.emplace(kGarSetPathName + "@" + all_gar_name_set[gar_idx]);
          if (!gar_name_set_ret){
            // has already have this name
            std::cout << "duplicated gar name: " << *(gar_name_set_it)
                      << std::endl;
            return;
          }
        }
      }
      else {
        if (!gar_path_config["VFile"]){
          std::cout<<"cannot get gar v file"<<std::endl;
          return;
        }
        const std::string kGarPathVFile 
              = gar_path_config["VFile"].as<std::string>();

        if (!gar_path_config["EFile"]){
          std::cout<<"cannot get gar e file"<<std::endl;
          return;
        }
        const std::string kGarPathEFile
              = gar_path_config["EFile"].as<std::string>();

        if (!gar_path_config["XFile"]){
          std::cout<<"cannot get gar x file"<<std::endl;
          return;
        }
        const std::string kGarPathXFile 
              = gar_path_config["XFile"].as<std::string>();

        if (!gar_path_config["YFile"]){
          std::cout<<"cannot get gar y file"<<std::endl;
          return;
        }
        const std::string kGarPathYFile 
              = gar_path_config["YFile"].as<std::string>();

        const std::string kGarPathName
                        = kGarPathVFile.substr(0,
                          kGarPathVFile.length()
                               - kVFilePosfix.size());

        auto [gar_name_it,
              gar_name_ret] = gar_name_set.emplace(kGarPathName);
        if (!gar_name_ret) {
          std::cout << "duplicate gar name: " << kGarPathName << std::endl;
          return;
        }

        GarType gar;
                            
        auto read_gar_res = gar::ReadGAR(gar, kGarPathVFile, 
                                              kGarPathEFile, 
                                              kGarPathXFile, 
                                              kGarPathYFile);
              
        if (read_gar_res < 0){
          std::cout<<"read gar error!"<<std::endl;
          return;
        }

        gar_set.emplace_back(gar, kGarPathName);
      }
    }

    omp_lock_t statistic_lock;
    omp_init_lock(&statistic_lock);

    std::cout <<"omp_get_num_procs: "
              << omp_get_num_procs()<<std::endl;

    #pragma omp parallel for schedule(dynamic)
    for (size_t gar_idx = 0; gar_idx < gar_set.size(); gar_idx++) {

      GarType gar = collect_supp_xy_only? 
                             _gar_analysis::AddRhsToLhs(
                              gar_set[gar_idx].first)
                            : gar_set[gar_idx].first;
      std::string& gar_name = gar_set[gar_idx].second;

      Pattern rhs_pattern = _gar_analysis::RhsPattern(gar);

      GUNDAM::Match rhs_to_pattern_match(rhs_pattern, 
                                          gar.pattern(),"same_id_map");

      std::vector<PatternVertexHandle> rhs_vertex_handle_set;
      rhs_vertex_handle_set.reserve(rhs_pattern.CountVertex());
      for (auto vertex_it = rhs_pattern.VertexBegin();
               !vertex_it.IsDone();
                vertex_it++) {
        rhs_vertex_handle_set.emplace_back(vertex_it);
      }
      assert(rhs_vertex_handle_set.size() 
          == rhs_pattern.CountVertex());

      using MatchType = GUNDAM::Match<Pattern, DataGraph>;

      auto satisfy_x_callback_with_supp_set
        = [&rhs_vertex_handle_set,
          &rhs_to_pattern_match](const MatchType& match,
            std::set<
            std::vector<DataGraphVertexHandle>>& satisfy_x_support_set){
        assert(rhs_vertex_handle_set.size() == match.size());
        std::vector<DataGraphVertexHandle> support;
        for (auto& rhs_vertex_handle 
                 : rhs_vertex_handle_set) {
          auto pattern_vertex_handle 
                    = rhs_to_pattern_match.MapTo(
                                  rhs_vertex_handle);
          assert(pattern_vertex_handle);
          auto dst_handle = match.MapTo(pattern_vertex_handle);
          assert(dst_handle);
          support.emplace_back(dst_handle);
        }
        assert(support.size() == rhs_vertex_handle_set.size());
        auto [satisfy_x_support_it,
              satisfy_x_support_ret]
            = satisfy_x_support_set.emplace(support);
        // should added successfully
        assert(satisfy_x_support_ret);
        return true;
      };

      auto satisfy_xy_callback_with_supp_set
        = [&rhs_vertex_handle_set,
           &rhs_to_pattern_match](const MatchType& match,
            std::set<
            std::vector<DataGraphVertexHandle>>& satisfy_xy_support_set){
        assert(rhs_vertex_handle_set.size() == match.size());
        std::vector<DataGraphVertexHandle> support;
        for (auto& rhs_vertex_handle : rhs_vertex_handle_set) {
          PatternVertexHandle pattern_vertex_handle 
                    = rhs_to_pattern_match.MapTo(
                                  rhs_vertex_handle);
          auto dst_handle = match.MapTo(pattern_vertex_handle);
          assert(dst_handle);
          support.emplace_back(dst_handle);
        }
        assert(support.size() == rhs_vertex_handle_set.size());
        auto [satisfy_xy_support_it,
              satisfy_xy_support_ret]
            = satisfy_xy_support_set.emplace(support);
        // should added successfully
        assert(satisfy_xy_support_ret);
        return true;
      };

      std::set<std::vector<DataGraphVertexHandle>> 
        data_graph_satisfy_x_support_set,
        data_graph_satisfy_xy_support_set;

      std::function<bool(const MatchType&)>  
            satisfy_x_callback_on_data_graph
        = [&satisfy_x_callback_with_supp_set,
          &data_graph_satisfy_x_support_set](const MatchType& match){
        return satisfy_x_callback_with_supp_set(match, data_graph_satisfy_x_support_set);
      };

      std::function<bool(const MatchType&)>  
            satisfy_xy_callback_on_data_graph
        = [&satisfy_xy_callback_with_supp_set,
          &data_graph_satisfy_xy_support_set](const MatchType& match){
        return satisfy_xy_callback_with_supp_set(match, data_graph_satisfy_xy_support_set);
      };

      auto [x_supp, xy_supp] = gar::GarSupp(gar, data_graph, 
                                            satisfy_x_callback_on_data_graph,
                                            satisfy_xy_callback_on_data_graph,
                                            time_limit,
                                            time_limit_per_supp);

      assert( data_graph_satisfy_x_support_set.size() ==  x_supp);
      assert(data_graph_satisfy_xy_support_set.size() == xy_supp);
      assert( data_graph_satisfy_x_support_set.size()
          >= data_graph_satisfy_xy_support_set.size());

      omp_set_lock(&statistic_lock);

      if (count_matched_rhs_edge) {
        if ((*gar.y_literal_set().begin())->type() == gar::LiteralType::kEdgeLiteral) {
          for (auto& xy_support_set : data_graph_satisfy_xy_support_set) {
            assert(xy_support_set.size() == 2);
            bool meet_the_same_edge = false;
            auto rhs_literal_info = (*gar.y_literal_set().begin())->info();
            for (auto out_edge_it = xy_support_set[0]->OutEdgeBegin();
                     !out_edge_it.IsDone();
                      out_edge_it++) {
              if (out_edge_it->label() != rhs_literal_info.edge_label()) {
                continue;
              }
              if (out_edge_it->dst_handle() != xy_support_set[1]) {
                continue;
              }
              meet_the_same_edge = true;
              auto [attr_handle, 
                    attr_ret] = out_edge_it->AddAttribute("satisfy_gar_set", std::string(""));
              assert(attr_handle);
              attr_handle->template value<std::string>() += "\t" + gar_name;
              break;
            }
            if (!meet_the_same_edge) {
              for (auto in_edge_it = xy_support_set[0]->InEdgeBegin();
                       !in_edge_it.IsDone();
                        in_edge_it++) {
                if (in_edge_it->label() != rhs_literal_info.edge_label()) {
                  continue;
                }
                if (in_edge_it->src_handle() != xy_support_set[1]) {
                  continue;
                }
                meet_the_same_edge = true;
                auto [attr_handle, 
                      attr_ret] = in_edge_it->AddAttribute("satisfy_gar_set", std::string(""));
                assert(attr_handle);
                attr_handle->template value<std::string>() += "\t" + gar_name;
                break;
              }
            }
            assert(meet_the_same_edge);
          }
        }
        else {
          std::cout << "gar_name: " << gar_name << " rhs is not edge literal"
                    << std::endl;
        }
      }

      gar_supp_on_original_graph.emplace(gar_name, std::pair(x_supp, xy_supp));

      std::cout<<"#########################################"<<std::endl;
      std::cout<<"## satisfy_x_support in original graph ##"<<std::endl;
      std::cout<<"#########################################"<<std::endl;
      // for (const auto& satisfy_x_support 
      //                : satisfy_x_support_set){
      //   std::cout<<"## ----------------- ##"<<std::endl;
      //   auto rhs_vertex_handle_it = rhs_vertex_handle_set.begin();
      //   for (const auto& data_graph_vertex_handle_it : satisfy_x_support){
      //     assert(rhs_vertex_handle_it 
      //         != rhs_vertex_handle_set.end());
      //     std::cout <<  "## src: " <<      (*rhs_vertex_handle_it)->id()
      //               <<"\t## dst: " << data_graph_vertex_handle_it ->id()<<std::endl;
      //     rhs_vertex_handle_it++;
      //   }
      //   auto satisfy_xy_support_it 
      //      = satisfy_xy_support_set.find(satisfy_x_support);
      //   if ( satisfy_xy_support_it 
      //     != satisfy_xy_support_set.end()){
      //     std::cout << "satisfy y literals" << std::endl;
      //     continue;
      //   }
      //   std::cout << "does not satisfy y literals" << std::endl;
      // }

      std::cout << "staisfy x : " <<  x_supp << std::endl;
      std::cout << "staisfy xy: " << xy_supp << std::endl;
      omp_unset_lock(&statistic_lock);

      if (!sampled_graph_set.empty()){

        for (auto& sampled_graph_it : sampled_graph_set) {
          const std::string& kSampledGraphName = sampled_graph_it.first;
          auto& sampled_graph                = sampled_graph_it.second.first;
          auto& sampled_graph_support_id_set = sampled_graph_it.second.second;
          auto sampled_graph_statistic_it
              = sampled_graph_statistic.find(kSampledGraphName);
          assert(sampled_graph_statistic_it
              != sampled_graph_statistic.end());
          auto& statistic = sampled_graph_statistic_it->second;

          std::set<std::vector<DataGraphVertexHandle>> 
            sampled_graph_satisfy_x_support_set,
            sampled_graph_satisfy_xy_support_set;

          std::function<bool(const MatchType&)> 
                satisfy_x_callback_on_sample_graph
            = [&satisfy_x_callback_with_supp_set,
              &sampled_graph_satisfy_x_support_set](const MatchType& match){
            return satisfy_x_callback_with_supp_set(match, sampled_graph_satisfy_x_support_set);
          };

          std::function<bool(const MatchType&)> 
                satisfy_xy_callback_on_sample_graph
            = [&satisfy_xy_callback_with_supp_set,
              &sampled_graph_satisfy_xy_support_set](const MatchType& match){
            return satisfy_xy_callback_with_supp_set(match, sampled_graph_satisfy_xy_support_set);
          };

          auto [x_supp, xy_supp] = gar::GarSupp(gar, sampled_graph,
                                                satisfy_x_callback_on_sample_graph,
                                                satisfy_xy_callback_on_sample_graph,
                                                time_limit,
                                                time_limit_per_supp);

          assert( sampled_graph_satisfy_x_support_set.size() ==  x_supp);
          assert(sampled_graph_satisfy_xy_support_set.size() == xy_supp);
          assert( sampled_graph_satisfy_x_support_set.size()
              >= sampled_graph_satisfy_xy_support_set.size());

          const bool kMatched = (xy_supp > 0);

          int matched_in_original_graph_contained_in_support_set_number = 0;

          if (!sampled_graph_support_id_set.empty()) {
            for (const auto& data_graph_satisfy_xy_support
                            : data_graph_satisfy_xy_support_set) {
              if (data_graph_satisfy_xy_support.size() > 2){
                // to large, cannot be handled
                std::cout << "data_graph_satisfy_xy_support.size() is too larget"
                          << std::endl;
                continue;
              }

              // verify in support set 
              assert(data_graph_satisfy_xy_support.size() == 1
                  || data_graph_satisfy_xy_support.size() == 2);

              std::vector<VertexIDType> support_id;
              support_id.reserve(data_graph_satisfy_xy_support.size());
              for (const auto& data_graph_vertex_handle 
                            : data_graph_satisfy_xy_support){
                support_id.emplace_back(data_graph_vertex_handle->id());
              }

              auto support_id_it = sampled_graph_support_id_set.find(support_id);
              if (support_id_it != sampled_graph_support_id_set.end()){
                // can be found in sampled graph support set
                matched_in_original_graph_contained_in_support_set_number++;
                continue;
              }

              if (data_graph_satisfy_xy_support.size() == 2){
                std::vector<VertexIDType> reverse_support_id;
                reverse_support_id.resize(2);
                reverse_support_id[0] = support_id[1];
                reverse_support_id[1] = support_id[0];
                
                auto support_id_it = sampled_graph_support_id_set.find(reverse_support_id);
                if (support_id_it != sampled_graph_support_id_set.end()){
                  // can be found in sampled graph support set
                  matched_in_original_graph_contained_in_support_set_number++;
                }
              }
            }
          }

          int matched_in_sampled_graph_contained_in_support_set_number = 0;

          if (!sampled_graph_support_id_set.empty()) {
            for (const auto& satisfy_xy_support
                            : sampled_graph_satisfy_xy_support_set) {
              if (satisfy_xy_support.size() > 2){
                // to large, cannot be handled
                std::cout << "satisfy_xy_support.size() is too larget"
                          << std::endl;
                continue;
              }

              // verify in support set 
              assert(satisfy_xy_support.size() == 1
                  || satisfy_xy_support.size() == 2);

              std::vector<VertexIDType> support_id;
              support_id.reserve(satisfy_xy_support.size());
              for (const auto& sample_graph_vertex_handle 
                            : satisfy_xy_support){
                support_id.emplace_back(sample_graph_vertex_handle->id());
              }

              auto support_id_it = sampled_graph_support_id_set.find(support_id);
              if (support_id_it != sampled_graph_support_id_set.end()){
                // can be found in sampled graph support set
                matched_in_sampled_graph_contained_in_support_set_number++;
                continue;
              }

              if (satisfy_xy_support.size() == 2){
                std::vector<VertexIDType> reverse_support_id;
                reverse_support_id.resize(2);
                reverse_support_id[0] = support_id[1];
                reverse_support_id[1] = support_id[0];
                
                auto support_id_it = sampled_graph_support_id_set.find(reverse_support_id);
                if (support_id_it != sampled_graph_support_id_set.end()){
                  // can be found in sampled graph support set
                  matched_in_sampled_graph_contained_in_support_set_number++;
                }
              }
            }
          }

          assert(matched_in_sampled_graph_contained_in_support_set_number 
            <= matched_in_original_graph_contained_in_support_set_number);

          omp_set_lock(&statistic_lock);
          std::cout << "matched_in_original_graph_contained_in_support_set_number: "
                    <<  matched_in_original_graph_contained_in_support_set_number << std::endl;

          std::cout << "matched_in_sampled_graph_contained_in_support_set_number: "
                    <<  matched_in_sampled_graph_contained_in_support_set_number << std::endl;

          if (kMatched){
            auto [gar_name_it, gar_name_ret]
              = std::get<kMatchedRuleNameIdx>(statistic).emplace(gar_name, std::pair(x_supp, xy_supp));
            // should added successfully
            assert(gar_name_ret);
          }

          if (matched_in_original_graph_contained_in_support_set_number > 0){
            auto [gar_name_it, gar_name_ret]
              = std::get<kContainedInSuppRuleNameIdx>(statistic).emplace(gar_name, matched_in_original_graph_contained_in_support_set_number);
            // should added successfully
            assert(gar_name_ret);
          }

          if (kMatched && (matched_in_sampled_graph_contained_in_support_set_number > 0)){
            auto [gar_name_it, gar_name_ret]
              = std::get<kMatchedContainedInSuppRuleNameIdx>(statistic).emplace(gar_name, matched_in_sampled_graph_contained_in_support_set_number);
            // should added successfully
            assert(gar_name_ret);
          }

          if (!kMatched && (matched_in_original_graph_contained_in_support_set_number > 0)){
            auto [gar_name_it, gar_name_ret]
              = std::get<kNotMatchedContainedInSuppRuleNameIdx>(statistic).emplace(gar_name, matched_in_original_graph_contained_in_support_set_number);
            // should added successfully
            assert(gar_name_ret);
          }

          sampled_graph_satisfy_x_support_set .clear();
          sampled_graph_satisfy_xy_support_set.clear();

          std::cout<<"########################################"<<std::endl;
          std::cout<<"## satisfy_x_support in sampled graph: "<< kSampledGraphName << " ##" << std::endl;
          std::cout<<"########################################"<<std::endl;
          // for (const auto& satisfy_x_support : sampled_graph_satisfy_x_support_set){
          //   std::cout<<"## ----------------- ##"<<std::endl;
          //   auto rhs_vertex_handle_it = rhs_vertex_handle_set.begin();
          //   for (const auto& data_graph_vertex_handle_it : satisfy_x_support){
          //     assert(rhs_vertex_handle_it 
          //         != rhs_vertex_handle_set.end());
          //     std::cout <<  "## src: " <<      (*rhs_vertex_handle_it)->id()
          //               <<"\t## dst: " << data_graph_vertex_handle_it ->id()<<std::endl;
          //     rhs_vertex_handle_it++;
          //   }
          //   auto satisfy_xy_support_it 
          //     = satisfy_xy_support_set.find(satisfy_x_support);
          //   if (satisfy_xy_support_it 
          //   != satisfy_xy_support_set.end()){
          //     std::cout << "satisfy y literals" << std::endl;
          //     continue;
          //   }
          //   std::cout << "does not satisfy y literals" << std::endl;
          // }
          std::cout << "staisfy x : " <<  x_supp << std::endl;
          std::cout << "staisfy xy: " << xy_supp << std::endl;
          omp_unset_lock(&statistic_lock);
        }
      }
    }

    for (const auto& gar_supp : gar_supp_on_original_graph){
      std::cout << gar_supp.first << "\t" << gar_supp.second.first
                                  << "\t" << gar_supp.second.second
                                  << std::endl;
    }

    if (count_matched_rhs_edge) {
      std::set<std::pair<size_t, std::set<std::string>>> covered_gars_set;
      std::vector<std::pair<EdgeIDType, std::set<std::string>>> edge_covered_gars;
      for (auto vertex_it = data_graph.VertexBegin();
               !vertex_it.IsDone();
                vertex_it++) {
        for (auto out_edge_it = vertex_it->OutEdgeBegin();
                 !out_edge_it.IsDone();
                  out_edge_it++) {
          auto attribute_handle = out_edge_it->FindAttribute("satisfy_gar_set");
          if (!attribute_handle) {
            // is not covered by any gar
            continue;
          }
          std::istringstream ss( attribute_handle->template value<std::string>() );
          std::string covered_gar; 
          edge_covered_gars.emplace_back(out_edge_it->id(), std::set<std::string>());
          while (ss >> covered_gar)
            edge_covered_gars.back().second.emplace(covered_gar);
          covered_gars_set.emplace(edge_covered_gars.back().second.size(),
                                   edge_covered_gars.back().second);
        }
      }
      
      GUNDAM::WriteCSVGraph(data_graph, 
                           kDataGraphPathVFile + "_marked_covered_gar",
                           kDataGraphPathEFile + "_marked_covered_gar");
      std::ofstream log_file(kDataGraphPathEFile + "_edge_covered_gar_set");

      std::sort(edge_covered_gars.begin(), 
                edge_covered_gars.end(), 
             [](const std::pair<EdgeIDType, std::set<std::string>>& i,
                const std::pair<EdgeIDType, std::set<std::string>>& j){
                return i.second.size() < j.second.size();
             });
      
      for (const auto& edge_covered_gar 
                     : edge_covered_gars) {
        log_file << edge_covered_gar.first;

        for (const auto& covered_gar : edge_covered_gar.second){
          log_file << "\t" << covered_gar;
        }
        log_file << std::endl;
      }

      log_file.close();
      log_file.open(kDataGraphPathEFile + "_cover_rule");
      size_t edge_counter = 0;
      std::map<std::string, size_t> covered_gar_set;
      for (auto edge_covered_gars_it  = edge_covered_gars.begin();
                edge_covered_gars_it != edge_covered_gars.end();
                edge_covered_gars_it++) {
        for (auto gar_name_it  = edge_covered_gars_it->second.begin();
                  gar_name_it != edge_covered_gars_it->second.end();
                  gar_name_it++) {
          covered_gar_set[*gar_name_it]++;
        }
        size_t satisfy_support_gar_counter = 0;
        for (const auto& covered_gar : covered_gar_set) {
          if (covered_gar.second >= support_bound) {
            satisfy_support_gar_counter++;
          }
        }
        edge_counter++;
        log_file << edge_counter << "\t" << satisfy_support_gar_counter << std::endl;
      }

      log_file.close();
      log_file.open(kDataGraphPathEFile + "_cover_rule_reverse");
      edge_counter = 0;
      covered_gar_set.clear();
      for (auto edge_covered_gars_it  = edge_covered_gars.rbegin();
                edge_covered_gars_it != edge_covered_gars.rend();
                edge_covered_gars_it++) {
        for (auto gar_name_it  = edge_covered_gars_it->second.begin();
                  gar_name_it != edge_covered_gars_it->second.end();
                  gar_name_it++) {
          covered_gar_set[*gar_name_it]++;
        }
        size_t satisfy_support_gar_counter = 0;
        for (const auto& covered_gar : covered_gar_set) {
          if (covered_gar.second >= support_bound) {
            satisfy_support_gar_counter++;
          }
        }
        edge_counter++;
        log_file << edge_counter << "\t" << satisfy_support_gar_counter << std::endl;
      }

      for (const auto& gars_set : covered_gars_set) {
        std::cout << "\t" << gars_set.first;
        for (const auto& gar_name : gars_set.second){
          std::cout << "\t" << gar_name;
        }
        std::cout << std::endl;
      }
      // getchar();
    }

    // statistic by sampled graph
    for (const auto& statistic : sampled_graph_statistic) {
      std::cout << "\t" << statistic.first
               << "\t" << std::get<   kMatchedRuleNameIdx               >(statistic.second).size()
               << "\t" << std::get<          kContainedInSuppRuleNameIdx>(statistic.second).size()
               << "\t" << std::get<   kMatchedContainedInSuppRuleNameIdx>(statistic.second).size()
               << "\t" << std::get<kNotMatchedContainedInSuppRuleNameIdx>(statistic.second).size()
               << std::endl;
    }

    std::cout << "supp x on sampled graph";
    for (const auto& statistic : sampled_graph_statistic) {
      std::cout << "\t" << statistic.first;
    }
    std::cout << std::endl;
    std::cout << "support size";
    for (const auto& sampled_graph : sampled_graph_set) {
      std::cout << "\t" << sampled_graph.second.second.size();
    }
    std::cout << std::endl;
    // statistic by gar
    for (const auto& gar_name : gar_name_set) {
      // export support x
      std::cout << gar_name;
      for (const auto& statistic : sampled_graph_statistic) {
        auto statistic_it =  std::get<kMatchedRuleNameIdx>(statistic.second).find(gar_name);
        if ( statistic_it == std::get<kMatchedRuleNameIdx>(statistic.second).end()) {
          std::cout << "\t" << 0;
          continue;
        }
        std::cout << "\t" << statistic_it->second.first;
      }
      std::cout << std::endl;
    }

    std::cout << "supp xy on sampled graph";
    for (const auto& statistic : sampled_graph_statistic) {
      std::cout << "\t" << statistic.first;
    }
    std::cout << std::endl;
    std::cout << "support size";
    for (const auto& sampled_graph : sampled_graph_set) {
      std::cout << "\t" << sampled_graph.second.second.size();
    }
    std::cout << std::endl;
    // statistic by gar
    for (const auto& gar_name : gar_name_set) {
      // export support xy
      std::cout << gar_name;
      for (const auto& statistic : sampled_graph_statistic) {
        auto statistic_it =  std::get<kMatchedRuleNameIdx>(statistic.second).find(gar_name);
        if ( statistic_it == std::get<kMatchedRuleNameIdx>(statistic.second).end()) {
          std::cout << "\t" << 0;
          continue;
        }
        std::cout << "\t" << statistic_it->second.second;
      }
      std::cout << std::endl;
    }

    std::cout << "original graph supp xy in support set";
    for (const auto& statistic : sampled_graph_statistic) {
      std::cout << "\t" << statistic.first;
    }
    std::cout << std::endl;
    std::cout << "support size";
    for (const auto& sampled_graph : sampled_graph_set) {
      std::cout << "\t" << sampled_graph.second.second.size();
    }
    std::cout << std::endl;
    // statistic by gar
    for (const auto& gar_name : gar_name_set) {
      // export support xy in support set
      std::cout << gar_name;
      for (const auto& statistic : sampled_graph_statistic) {
        auto statistic_it =  std::get<kContainedInSuppRuleNameIdx>(statistic.second).find(gar_name);
        if ( statistic_it == std::get<kContainedInSuppRuleNameIdx>(statistic.second).end()) {
          std::cout << "\t" << 0;
          continue;
        }
        std::cout << "\t" << statistic_it->second;
      }
      std::cout << std::endl;
    }

    std::cout << "sampled graph supp xy in support set";
    for (const auto& statistic : sampled_graph_statistic) {
      std::cout << "\t" << statistic.first;
    }
    std::cout << std::endl;
    std::cout << "support size";
    for (const auto& sampled_graph : sampled_graph_set) {
      std::cout << "\t" << sampled_graph.second.second.size();
    }
    std::cout << std::endl;
    // statistic by gar
    for (const auto& gar_name : gar_name_set) {
      // export support xy in support set
      std::cout << gar_name;
      for (const auto& statistic : sampled_graph_statistic) {
        auto statistic_it =  std::get<kMatchedContainedInSuppRuleNameIdx>(statistic.second).find(gar_name);
        if ( statistic_it == std::get<kMatchedContainedInSuppRuleNameIdx>(statistic.second).end()) {
          std::cout << "\t" << 0;
          continue;
        }
        std::cout << "\t" << statistic_it->second;
      }
      std::cout << std::endl;
    }

    std::cout << "graph_package_set.size(): "
              <<  graph_package_set.size()
              << std::endl;

    for (const auto& graph_package : graph_package_set){
      const auto& kGraphPackageName = graph_package.first;
      std::map<std::string, std::pair<int, int>> matched_rule; 
      std::map<std::string, int> 
                  contained_in_supp_rule, 
          matched_contained_in_supp_rule,
      not_matched_contained_in_supp_rule;
      for (const auto& sampled_graph_name : graph_package.second) {
        auto sampled_graph_statistic_it
           = sampled_graph_statistic.find(sampled_graph_name);
        assert(sampled_graph_statistic_it
            != sampled_graph_statistic.end());
        auto& statistic = *sampled_graph_statistic_it;
                              matched_rule
              .insert(std::get<kMatchedRuleNameIdx>(statistic.second).begin(),
                      std::get<kMatchedRuleNameIdx>(statistic.second).end());
                    contained_in_supp_rule
              .insert(std::get<kContainedInSuppRuleNameIdx>(statistic.second).begin(),
                      std::get<kContainedInSuppRuleNameIdx>(statistic.second).end());
            matched_contained_in_supp_rule
              .insert(std::get<kMatchedContainedInSuppRuleNameIdx>(statistic.second).begin(),
                      std::get<kMatchedContainedInSuppRuleNameIdx>(statistic.second).end());
        not_matched_contained_in_supp_rule
              .insert(std::get<kNotMatchedContainedInSuppRuleNameIdx>(statistic.second).begin(),
                      std::get<kNotMatchedContainedInSuppRuleNameIdx>(statistic.second).end());
      }
      std::cout << "\t" << kGraphPackageName
               << "\t" <<                       matched_rule.size()
               << "\t" <<             contained_in_supp_rule.size()
               << "\t" <<     matched_contained_in_supp_rule.size()
               << "\t" << not_matched_contained_in_supp_rule.size()
               << std::endl;
    }

    std::ofstream log_file(kLogFile);

    for (auto& gar_supp : gar_supp_on_original_graph){
      log_file << gar_supp.first << "\t" << gar_supp.second.first
                                 << "\t" << gar_supp.second.second
                                 << std::endl;
    }

    // statistic by sampled graph
    for (const auto& statistic : sampled_graph_statistic) {
      log_file << "\t" << statistic.first
               << "\t" << std::get<   kMatchedRuleNameIdx               >(statistic.second).size()
               << "\t" << std::get<          kContainedInSuppRuleNameIdx>(statistic.second).size()
               << "\t" << std::get<   kMatchedContainedInSuppRuleNameIdx>(statistic.second).size()
               << "\t" << std::get<kNotMatchedContainedInSuppRuleNameIdx>(statistic.second).size()
               << std::endl;
    }

    log_file << "supp x on sampled graph";
    for (const auto& statistic : sampled_graph_statistic) {
      log_file << "\t" << statistic.first;
    }
    log_file << std::endl;
    log_file << "support size";
    for (const auto& sampled_graph : sampled_graph_set) {
      log_file << "\t" << sampled_graph.second.second.size();
    }
    log_file << std::endl;
    // statistic by gar
    for (const auto& gar_name : gar_name_set) {
      // export support x
      log_file << gar_name;
      for (const auto& statistic : sampled_graph_statistic) {
        auto statistic_it =  std::get<kMatchedRuleNameIdx>(statistic.second).find(gar_name);
        if ( statistic_it == std::get<kMatchedRuleNameIdx>(statistic.second).end()) {
          log_file << "\t" << 0;
          continue;
        }
        log_file << "\t" << statistic_it->second.first;
      }
      log_file << std::endl;
    }

    log_file << "supp xy on sampled graph";
    for (const auto& statistic : sampled_graph_statistic) {
      log_file << "\t" << statistic.first;
    }
    log_file << std::endl;
    log_file << "support size";
    for (const auto& sampled_graph : sampled_graph_set) {
      log_file << "\t" << sampled_graph.second.second.size();
    }
    log_file << std::endl;
    // statistic by gar
    for (const auto& gar_name : gar_name_set) {
      // export support xy
      log_file << gar_name;
      for (const auto& statistic : sampled_graph_statistic) {
        auto statistic_it =  std::get<kMatchedRuleNameIdx>(statistic.second).find(gar_name);
        if ( statistic_it == std::get<kMatchedRuleNameIdx>(statistic.second).end()) {
          log_file << "\t" << 0;
          continue;
        }
        log_file << "\t" << statistic_it->second.second;
      }
      log_file << std::endl;
    }

    log_file << "original graph supp xy in support set";
    for (const auto& statistic : sampled_graph_statistic) {
      log_file << "\t" << statistic.first;
    }
    log_file << std::endl;
    log_file << "support size";
    for (const auto& sampled_graph : sampled_graph_set) {
      log_file << "\t" << sampled_graph.second.second.size();
    }
    log_file << std::endl;
    // statistic by gar
    for (const auto& gar_name : gar_name_set) {
      // export support xy in support set
      log_file << gar_name;
      for (const auto& statistic : sampled_graph_statistic) {
        auto statistic_it =  std::get<kContainedInSuppRuleNameIdx>(statistic.second).find(gar_name);
        if ( statistic_it == std::get<kContainedInSuppRuleNameIdx>(statistic.second).end()) {
          log_file << "\t" << 0;
          continue;
        }
        log_file << "\t" << statistic_it->second;
      }
      log_file << std::endl;
    }

    log_file << "sampled graph supp xy in support set";
    for (const auto& statistic : sampled_graph_statistic) {
      log_file << "\t" << statistic.first;
    }
    log_file << std::endl;
    log_file << "support size";
    for (const auto& sampled_graph : sampled_graph_set) {
      log_file << "\t" << sampled_graph.second.second.size();
    }
    log_file << std::endl;
    // statistic by gar
    for (const auto& gar_name : gar_name_set) {
      // export support xy in support set
      log_file << gar_name;
      for (const auto& statistic : sampled_graph_statistic) {
        auto statistic_it =  std::get<kMatchedContainedInSuppRuleNameIdx>(statistic.second).find(gar_name);
        if ( statistic_it == std::get<kMatchedContainedInSuppRuleNameIdx>(statistic.second).end()) {
          log_file << "\t" << 0;
          continue;
        }
        log_file << "\t" << statistic_it->second;
      }
      log_file << std::endl;
    }

    std::cout << "graph_package_set.size(): "
              <<  graph_package_set.size()
              << std::endl;

    for (const auto& graph_package : graph_package_set){
      const auto& kGraphPackageName = graph_package.first;
      std::map<std::string, std::pair<int, int>> matched_rule; 
      std::map<std::string, int> 
                  contained_in_supp_rule, 
          matched_contained_in_supp_rule,
      not_matched_contained_in_supp_rule;
      for (const auto& sampled_graph_name : graph_package.second) {
        auto sampled_graph_statistic_it
           = sampled_graph_statistic.find(sampled_graph_name);
        assert(sampled_graph_statistic_it
            != sampled_graph_statistic.end());
        auto& statistic = *sampled_graph_statistic_it;
                              matched_rule
              .insert(std::get<kMatchedRuleNameIdx>(statistic.second).begin(),
                      std::get<kMatchedRuleNameIdx>(statistic.second).end());
                    contained_in_supp_rule
              .insert(std::get<kContainedInSuppRuleNameIdx>(statistic.second).begin(),
                      std::get<kContainedInSuppRuleNameIdx>(statistic.second).end());
            matched_contained_in_supp_rule
              .insert(std::get<kMatchedContainedInSuppRuleNameIdx>(statistic.second).begin(),
                      std::get<kMatchedContainedInSuppRuleNameIdx>(statistic.second).end());
        not_matched_contained_in_supp_rule
              .insert(std::get<kNotMatchedContainedInSuppRuleNameIdx>(statistic.second).begin(),
                      std::get<kNotMatchedContainedInSuppRuleNameIdx>(statistic.second).end());
      }
      log_file << "\t" << kGraphPackageName
               << "\t" <<                       matched_rule.size()
               << "\t" <<             contained_in_supp_rule.size()
               << "\t" <<     matched_contained_in_supp_rule.size()
               << "\t" << not_matched_contained_in_supp_rule.size()
               << std::endl;
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
   * @brief Incremental evaluation for GarAnalysis.
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

#endif  // EXAMPLES_ANALYTICAL_APPS_GAR_ANALYSIS_GAR_ANALYSIS_H_