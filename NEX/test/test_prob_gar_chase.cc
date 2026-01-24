#include <iostream>

#include "include/gundam/graph_type/graph.h"
#include "include/gundam/graph_type/small_graph.h"
#include "include/gundam/graph_type/large_graph.h"

#include "include/gundam/algorithm/match_using_match.h"

#include "include/gundam/tool/same_pattern.h"
#include "include/gundam/tool/copy_to.h"

#include "include/gundam/io/csvgraph.h"

#include "include/gar/gar.h"
#include "include/gar/same_gar.h"

#include "include/prob_gar/csv_prob_gar.h"
#include "include/prob_gar/prob_gar.h"
#include "include/prob_gar/prob_gar_chase.h"

template<typename GraphType>
void AddIfNotExist(std::vector<GraphType>& existed_pattern_set,
                         const GraphType&      new_pattern){
  for (const auto& existed_pattern : existed_pattern_set) {
    if (GUNDAM::SamePattern(existed_pattern,
                                new_pattern)) {
      return;
    }
  }
  existed_pattern_set.emplace_back(new_pattern);
  return;
}

int main() {

  using    DataGraphType = GUNDAM::LargeGraph<int, char, std::string,
                                              int, char, std::string>;
  using GraphPatternType = GUNDAM::SmallGraph<int, char,
                                              int, char>;

  using ProbGarType = prob_gar::ProbGraphAssociationRule<GraphPatternType, 
                                                            DataGraphType>;

  // ####################
  // ###  test set 0  ###
  // ####################
  // 0.3::has_edge(V0,V2) :- vertex_A(V0), vertex_B(V1), vertex_B(V2), has_edge(V0,V1).

  // vertex_A(v0).
  // vertex_B(v1).
  // vertex_B(v2).
  // vertex_B(v3).

  // has_edge(v0,v1).

  // query(has_edge(v0,v1)).
  // query(has_edge(v0,v2)).
  // query(has_edge(v0,v3)).

  DataGraphType data_graph_0;
  data_graph_0.AddVertex(0, 'A');
  data_graph_0.AddVertex(1, 'B');
  data_graph_0.AddVertex(2, 'B');
  data_graph_0.AddVertex(3, 'B');
  data_graph_0.AddEdge(0, 1, 'a', 0);

  GraphPatternType graph_pattern_0;
  graph_pattern_0.AddVertex(0, 'A');
  graph_pattern_0.AddVertex(1, 'B');
  graph_pattern_0.AddVertex(2, 'B');
  graph_pattern_0.AddEdge(0, 1, 'a', 0);
  
  ProbGarType prob_gar_0(graph_pattern_0);
  prob_gar_0.AddY<gar::EdgeLiteral<GraphPatternType, 
                                      DataGraphType>>(0, 2, 'a');
  prob_gar_0.set_prob(0.30);

  std::map<typename GUNDAM::EdgeID<DataGraphType>::type, float> diff_edge_0;

  prob_gar::ProbGARChase(prob_gar_0, 
                       data_graph_0, 
                        diff_edge_0);

  std::string data_graph_0_str;
  data_graph_0_str << data_graph_0;

  std::cout << data_graph_0_str << std::endl;
  for (const auto& [edge_id, prob] : diff_edge_0) {
    std::cout << "\tedge:" << edge_id 
              << "\tprob:" << prob << std::endl;
    assert(fabs(prob - 0.363) < 0.02);
  }

  // ####################
  // ###  test set 1  ###
  // ####################
  // 0.3::has_edge(V0,V2) :- vertex_A(V0), vertex_B(V1), vertex_B(V2), has_edge(V0,V1).
  // 0.5::has_edge(V0,V1) :- vertex_A(V0), vertex_B(V1).

  // vertex_A(v0).
  // vertex_B(v1).
  // vertex_B(v2).
  // vertex_B(v3).

  // has_edge(v0,v1).

  // query(has_edge(v0,v1)).
  // query(has_edge(v0,v2)).
  // query(has_edge(v0,v3)).

  DataGraphType data_graph_1;
  data_graph_1.AddVertex(0, 'A');
  data_graph_1.AddVertex(1, 'B');
  data_graph_1.AddVertex(2, 'B');
  data_graph_1.AddVertex(3, 'B');
  data_graph_1.AddEdge(0, 1, 'a', 0);

  GraphPatternType graph_pattern_1;
  graph_pattern_1.AddVertex(0, 'A');
  graph_pattern_1.AddVertex(1, 'B');
  
  ProbGarType prob_gar_1(graph_pattern_1);
  prob_gar_1.AddY<gar::EdgeLiteral<GraphPatternType, 
                                      DataGraphType>>(0, 1, 'a');
  prob_gar_1.set_prob(0.50);

  std::vector<ProbGarType> prob_gar_set_1;
  prob_gar_set_1.emplace_back(prob_gar_1);
  prob_gar_set_1.emplace_back(prob_gar_0);

  std::map<typename GUNDAM::EdgeID<DataGraphType>::type, float> diff_edge_1;

  prob_gar::ProbGARChase(prob_gar_set_1, 
                           data_graph_1, 
                            diff_edge_1);

  std::string data_graph_1_str;
  data_graph_1_str << data_graph_1;

  std::cout << data_graph_1_str << std::endl;
  for (const auto& [edge_id, prob] : diff_edge_1) {
    std::cout << "\tedge:" << edge_id 
              << "\tprob:" << prob << std::endl;
    assert(fabs(prob - 0.718) < 0.02);
  }

  // ####################
  // ###  test set 2  ###
  // ####################
  // 0.3::has_edge(V0,V2) :- vertex_A(V0), vertex_B(V1), vertex_B(V2), has_edge(V0,V1).
  // 0.5::has_edge(V0,V1) :- vertex_A(V0), vertex_B(V1).

  // vertex_A(v0).
  // vertex_B(v1).
  // vertex_B(v2).
  // vertex_B(v3).

  // has_edge(v0,v1).
  // has_edge(v0,v2).

  // query(has_edge(v0,v1)).
  // query(has_edge(v0,v2)).
  // query(has_edge(v0,v3)).

  DataGraphType data_graph_2;
  data_graph_2.AddVertex(0, 'A');
  data_graph_2.AddVertex(1, 'B');
  data_graph_2.AddVertex(2, 'B');
  data_graph_2.AddVertex(3, 'B');
  data_graph_2.AddEdge(0, 1, 'a', 0);
  data_graph_2.AddEdge(0, 2, 'a', 1);

  std::vector<ProbGarType> prob_gar_set_2;
  prob_gar_set_2.emplace_back(prob_gar_1);
  prob_gar_set_2.emplace_back(prob_gar_0);

  std::map<typename GUNDAM::EdgeID<DataGraphType>::type, float> diff_edge_2;

  prob_gar::ProbGARChase(prob_gar_set_2, 
                           data_graph_2, 
                            diff_edge_2);

  std::string data_graph_2_str;
  data_graph_2_str << data_graph_2;

  std::cout << data_graph_2_str << std::endl;
  for (const auto& [edge_id, prob] : diff_edge_2) {
    std::cout << "\tedge:" << edge_id 
              << "\tprob:" << prob << std::endl;
    assert(fabs(prob - 0.755) < 0.02);
  }

  // ####################
  // ###  test set 3  ###
  // ####################
  // 0.3::has_edge(V0,V2) :- vertex_A(V0), vertex_A(V1), vertex_A(V2), has_edge(V0,V1).
  // 0.5::has_edge(V0,V1) :- vertex_A(V0), vertex_A(V1).

  // vertex_A(v0).
  // vertex_A(v1).
  // vertex_A(v2).
  // vertex_A(v3).

  // has_edge(v0,v1).

  DataGraphType data_graph_3;
  data_graph_3.AddVertex(0, 'A');
  data_graph_3.AddVertex(1, 'A');
  data_graph_3.AddVertex(2, 'A');
  data_graph_3.AddVertex(3, 'A');
  data_graph_3.AddEdge(0, 1, 'a', 0);
  
  GraphPatternType graph_pattern_3_0;
  graph_pattern_3_0.AddVertex(0, 'A');
  graph_pattern_3_0.AddVertex(1, 'A');
  graph_pattern_3_0.AddVertex(2, 'A');
  graph_pattern_3_0.AddEdge(0, 1, 'a', 0);
  
  ProbGarType prob_gar_3_0(graph_pattern_3_0);
  prob_gar_3_0.AddY<gar::EdgeLiteral<GraphPatternType, 
                                        DataGraphType>>(0, 2, 'a');
  prob_gar_3_0.set_prob(0.10);

  GraphPatternType graph_pattern_3_1;
  graph_pattern_3_1.AddVertex(0, 'A');
  graph_pattern_3_1.AddVertex(1, 'A');
  
  ProbGarType prob_gar_3_1(graph_pattern_3_1);
  prob_gar_3_1.AddY<gar::EdgeLiteral<GraphPatternType, 
                                        DataGraphType>>(0, 1, 'a');
  prob_gar_3_1.set_prob(0.10);

  std::vector<ProbGarType> prob_gar_set_3;
  prob_gar_set_3.emplace_back(prob_gar_3_0);
  prob_gar_set_3.emplace_back(prob_gar_3_1);

  std::map<typename GUNDAM::EdgeID<DataGraphType>::type, float> diff_edge_3;

  prob_gar::ProbGARChase(prob_gar_set_3, 
                           data_graph_3, 
                            diff_edge_3);

  std::string data_graph_3_str;
  data_graph_3_str << data_graph_3;

  std::cout << data_graph_3_str << std::endl;
  for (const auto& [edge_id, prob] : diff_edge_3) {
    std::cout << "\tedge:" << edge_id 
              << "\tprob:" << prob << std::endl;
    // assert(fabs(prob - 0.783) < 0.02);
  }
  return 0;
}