#ifndef EXAMPLES_ANALYTICAL_APPS_GAR_COVER_BY_FEATURE_GAR_COVER_BY_FEATURE_H_
#define EXAMPLES_ANALYTICAL_APPS_GAR_COVER_BY_FEATURE_GAR_COVER_BY_FEATURE_H_

#include <grape/grape.h>
#include <omp.h>

#include <functional>

#include "gundam/io/csvgraph.h"
#include "gundam/io/csvgraph.h"

#include "gundam/match/match.h"
#include "gundam/match/matchresult.h"

#include "gundam/graph_type/graph.h"
#include "gundam/graph_type/small_graph.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/large_graph2.h"

#include "gundam/type_getter/vertex_handle.h"

#include <yaml-cpp/yaml.h>

#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "gar_cover_by_feature/gar_cover_by_feature_context.h"

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

template <typename FRAG_T>
class GarCoverByFeature : public ParallelAppBase<FRAG_T, GarCoverByFeatureContext<FRAG_T>>,
                          public ParallelEngine {
 public:
  static constexpr LoadStrategy load_strategy = LoadStrategy::kLoadWholeGraph;

 private:
  using    DataGraph = typename GarCoverByFeatureContext<FRAG_T>::DataGraph;
  using      Pattern = typename GarCoverByFeatureContext<FRAG_T>::  Pattern;

 public:
  // specialize the templated worker.
  INSTALL_PARALLEL_WORKER(GarCoverByFeature<FRAG_T>, 
                          GarCoverByFeatureContext<FRAG_T>,
                          FRAG_T)
  using vertex_t = typename fragment_t::vertex_t;

  /**
   * @brief Partial evaluation for GarCoverByFeature.
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
    
    std::ifstream config_file(ctx.yaml_file_);

    if (!config_file.is_open()) {
      std::cout << "cannot open config file." << std::endl;
      return;
    }
    YAML::Node config = YAML::LoadFile(ctx.yaml_file_);

    // load the gar set  
    if (!config["GarDir"]){
      std::cout << "cannot get gar directory!" << std::endl;
      return;
    }
    const std::string kGarDir
             = config["GarDir"].as<std::string>();

    if (!config["GarNames"]){
      std::cout << "cannot get gar name!" << std::endl;
      return;
    }
    YAML::Node gar_names_config = config["GarNames"];

    std::vector<std::pair<GARPatternToDataGraph, std::string>> gar_set;
    gar_set.reserve(gar_names_config.size());
    for(int i=0; i < gar_names_config.size(); ++i){
      const std::string kGarName = gar_names_config[i].as<std::string>();
      const std::string kGarVFile = kGarDir + "/" + kGarName + "_v.csv",
                        kGarEFile = kGarDir + "/" + kGarName + "_e.csv",
                        kGarXFile = kGarDir + "/" + kGarName + "_x.csv",
                        kGarYFile = kGarDir + "/" + kGarName + "_y.csv";
      GARPatternToDataGraph gar_p2d;
      auto res = gar::ReadGAR(gar_p2d, kGarVFile, kGarEFile,
                                       kGarXFile, kGarYFile);
      if (res < 0){
        std::cout<<"loading gar: "<<kGarName<<" failed! :"<<res<<std::endl;
        return;
      }
      gar_set.emplace_back(std::move(gar_p2d), kGarName);
    }

    // load the feature set  
    if (!config["FeatureDir"]){
      std::cout << "cannot get feature directory!" << std::endl;
      return;
    }
    const std::string kFeatureDir
             = config["FeatureDir"].as<std::string>();

    if (!config["FeatureNames"]) {
      std::cout << "cannot get the name of features!" << std::endl;
      return;
    }
    YAML::Node feature_names_config = config["FeatureNames"];

    std::vector<std::pair<GARPatternToDataGraph, std::string>> feature_set;
    feature_set.reserve(feature_names_config.size());
    for(int i=0; i < feature_names_config.size(); ++i){
      const std::string kFeatureName = feature_names_config[i].as<std::string>();
      const std::string kFeatureVFile = kFeatureDir + "/" + kFeatureName + "_v.csv",
                        kFeatureEFile = kFeatureDir + "/" + kFeatureName + "_e.csv",
                        kFeatureXFile = kFeatureDir + "/" + kFeatureName + "_x.csv",
                        kFeatureYFile = kFeatureDir + "/" + kFeatureName + "_y.csv";
      GARPatternToDataGraph feature;
      auto res = gar::ReadGAR(feature, kFeatureVFile, kFeatureEFile,
                                             kFeatureXFile, kFeatureYFile);
      if (res < 0){
        std::cout<<"loading gar: "<<kFeatureName<<" failed! :"<<res<<std::endl;
        return;
      }
      feature_set.emplace_back(std::move(feature), kFeatureName);
    }

    const std::string kCoveredByFeatureAttributeKey = "covered_by_feature";
    const std::string kCoveredAttrValue = "";

    std::vector<std::string> uncovered_gar_set;

    for (auto& gar_pair : gar_set){
      auto& gar_p2d = gar_pair.first;
      Pattern gar_p2d_pattern(gar_p2d.pattern());
      const std::string& kGarName = gar_pair.second;
      for (auto vertex_it = gar_p2d_pattern.VertexBegin();
               !vertex_it.IsDone(); 
                vertex_it++){
        vertex_it->AddAttribute(kCoveredByFeatureAttributeKey,
                                kCoveredAttrValue);
      }
      for (auto edge_it = gar_p2d_pattern.EdgeBegin();
               !edge_it.IsDone(); 
                edge_it++){
        edge_it->AddAttribute(kCoveredByFeatureAttributeKey,
                              kCoveredAttrValue);
      }
      for (const auto& feature_pair : feature_set){
        const auto& feature = feature_pair.first;
        const std::string& kFeatureName = feature_pair.second;

        auto feature_rhs_pattern = RhsPattern(feature);
        auto gar_p2d_rhs_pattern = RhsPattern(gar_p2d);

        GUNDAM::MatchSet rhs_match_result(feature_rhs_pattern,
                                          gar_p2d_rhs_pattern);

        GUNDAM::MatchUsingMatch(feature_rhs_pattern, 
                                gar_p2d_rhs_pattern,
                                  rhs_match_result);

        GUNDAM::Match
          feature_pattern_to_rhs_match(feature.pattern(),
                                       feature_rhs_pattern,
                                       "same_id_map");

        GUNDAM::Match
          gar_p2d_rhs_to_pattern_match(gar_p2d_rhs_pattern,
                                       gar_p2d_pattern,
                                       "same_id_map");

        for (auto rhs_match_it = rhs_match_result.MatchBegin();
                 !rhs_match_it.IsDone(); 
                  rhs_match_it++){
          GUNDAM::MatchSet const_match_result(feature.pattern(), 
                                              gar_p2d_pattern);

          GUNDAM::Match feature_to_gar_rhs_partial_match(feature.pattern(), 
                                                     gar_p2d_rhs_pattern);

          feature_to_gar_rhs_partial_match 
            = (*rhs_match_it)(feature_pattern_to_rhs_match);

          GUNDAM::Match feature_to_gar_partial_match(feature.pattern(), 
                                                     gar_p2d_pattern);
          feature_to_gar_partial_match 
            = gar_p2d_rhs_to_pattern_match(feature_to_gar_rhs_partial_match);

          // *rhs_match_it now is a partial match from
          // feature pattern to gar pattern
          MatchUsingMatch(feature.pattern(), 
                          gar_p2d_pattern,
                          feature_to_gar_partial_match,
                          const_match_result);
          for (auto const_match_it = const_match_result.MatchBegin();
                   !const_match_it.IsDone();
                    const_match_it++){
            GUNDAM::Match<const Pattern, Pattern> match;
            for (auto map_it = const_match_it->MapBegin();
                     !map_it.IsDone();
                      map_it++){
              match.AddMap(map_it->src_handle(),
                           gar_p2d_pattern
                                  .FindVertex(map_it->dst_handle()->id()));
            }
            for (auto vertex_it  = feature_rhs_pattern.VertexBegin();
                     !vertex_it.IsDone();
                      vertex_it++) {
              auto src_ptr = feature.pattern().FindVertex(vertex_it->id());
              match.MapTo(src_ptr)->template attribute<std::string>(kCoveredByFeatureAttributeKey) += " " + kFeatureName;
            }
            for (auto edge_it  = feature_rhs_pattern.EdgeBegin();
                     !edge_it.IsDone();
                      edge_it++) {
              auto src_ptr = feature.pattern().FindVertex(edge_it->src_handle()->id());
              auto dst_ptr = feature.pattern().FindVertex(edge_it->dst_handle()->id());
              assert(src_ptr);
              assert(dst_ptr);

              auto src_ptr_in_gar = match.MapTo(src_ptr);
              auto dst_ptr_in_gar = match.MapTo(dst_ptr);
              assert(src_ptr_in_gar);
              assert(dst_ptr_in_gar);

              for (auto out_edge_it = src_ptr_in_gar->OutEdgeBegin();
                       !out_edge_it.IsDone();
                        out_edge_it++){
                if (out_edge_it->label() != out_edge_it->label()){
                  continue;
                }
                if (out_edge_it->dst_handle() != dst_ptr_in_gar){
                  continue;
                }
                // find that edge
                out_edge_it->template attribute<std::string>(kCoveredByFeatureAttributeKey) += " " + kFeatureName;
                // does not break here, can have duplicate edge between two vertexes
              }
            }
          }
        }
      }

      std::cout<<"# all features has been processed #"<<std::endl;
      
      // to verify whether all vertexes are covered
      bool all_vertexes_covered = true;
      for (auto vertex_it = gar_p2d_pattern.VertexBegin();
               !vertex_it.IsDone(); 
                vertex_it++){
        auto attr_ptr = vertex_it->FindAttribute(kCoveredByFeatureAttributeKey);
        assert(!attr_ptr.IsNull());
        if (attr_ptr->template value<std::string>() == kCoveredAttrValue){
          // this vertex is not covered
          if (all_vertexes_covered){
            // meet the uncovered vertex in the first time
            std::cout<<"# uncovered vertexes: "<<std::endl;
            all_vertexes_covered = true;
          }
          std::cout<<"\t"<<vertex_it->id();
          continue;
        }
      }
      std::cout<<std::endl; 
      if (all_vertexes_covered){
        std::cout<<"# all vertexes are covered #"<<std::endl;
      }

      // to verify whether all edges are covered
      bool all_edges_covered = true;
      for (auto edge_it = gar_p2d_pattern.EdgeBegin();
               !edge_it.IsDone(); 
                edge_it++){
        auto attr_ptr = edge_it->FindAttribute(kCoveredByFeatureAttributeKey);
        assert(!attr_ptr.IsNull());
        if (attr_ptr->template value<std::string>() == kCoveredAttrValue){
          // this vertex is not covered
          if (all_edges_covered){
            // meet the uncovered vertex in the first time
            std::cout<<"# uncovered edges: "<<std::endl;
            all_edges_covered = false;
          }
          std::cout<<"\t"<<edge_it->id();
          continue;
        }
      }
      std::cout<<std::endl; 
      if (all_edges_covered){
        std::cout<<"# all edges are covered #"<<std::endl;
      }

      if (!all_vertexes_covered || !all_edges_covered){
        uncovered_gar_set.emplace_back(kGarName);
      }

      const std::string kOutputGarVFile = kGarDir + "/" + kGarName + "_output_v.csv",
                        kOutputGarEFile = kGarDir + "/" + kGarName + "_output_e.csv";

      GUNDAM::WriteCSVGraph(gar_p2d_pattern, kOutputGarVFile,
                                               kOutputGarEFile);
    }

    std::cout<<"uncovered gar:"<<std::endl;
    for (const auto& uncovered_gar : uncovered_gar_set){
      std::cout<<uncovered_gar<<std::endl;
    }
    
    // try to cover the discovered rule using the given features
    // for (const auto& feature_name : feature_name_list){

    //   if (res < 0) {
    //     std::cout<<"load feature: "<<feature_name
    //              <<" failed! res: "<<res<<std::endl;
    //     continue;
    //   }

    // }Match<const Pattern, Pattern> match;
    //       for (auto map_it = const_match_it->MapBegin();
    //                !map_it.IsDone();
    //                 map_it++){
    //         match.AddMap(map_it->src_handle(),
    //                      gar_p2d_pattern
    //                             .FindVertex(map_it->dst_handle()->id()));
    //       }
    //       for (auto vertex_it  = feature_rhs_pattern.VertexBegin();
    //                !vertex_it.IsDone();
    //                 vertex_it++) {
    //         auto src_ptr = feature.pattern().FindVertex(vertex_it->id());
    //         match.MapTo(src_ptr)->AddAttribute(kCoveredByFeatureAttributeKey,
    //                                                kCoveredAttrValue);
    //       }
    //       for (auto edge_it  = feature_rhs_pattern.EdgeBegin();
    //                !edge_it.IsDone();
    //                 edge_it++) {
    //         auto src_ptr = feature.pattern().FindVertex(edge_it->src_handle()->id());
    //         auto dst_ptr = feature.pattern().FindVertex(edge_it->dst_handle()->id());
    //         assert(src_ptr);
    //         assert(dst_ptr);

    //         auto src_ptr_in_gar = match.MapTo(src_ptr);
    //         auto dst_ptr_in_gar = match.MapTo(dst_ptr);
    //         assert(src_ptr_in_gar);
    //         assert(dst_ptr_in_gar);

    //         for (auto out_edge_it = src_ptr_in_gar->OutEdgeBegin();
    //                  !out_edge_it.IsDone();
    //                   out_edge_it++){
    //           if (out_edge_it->label() != out_edge_it->label()){
    //             continue;
    //           }
    //           if (out_edge_it->dst_handle() != dst_ptr_in_gar){
    //             continue;
    //           }
    //           // find that edge
    //           out_edge_it->AddAttribute(kCoveredByFeatureAttributeKey,
    //                                     kCoveredAttrValue);
    //           // can have duplicate edge between two vertexes
    //         }
    //       }
    //     }
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
   * @brief Incremental evaluation for GarCoverByFeature.
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

#endif  // EXAMPLES_ANALYTICAL_APPS_GAR_COVER_BY_FEATURE_GAR_COVER_BY_FEATURE_H_