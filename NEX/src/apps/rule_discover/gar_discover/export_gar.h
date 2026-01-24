#ifndef EXAMPLES_ANALYTICAL_APPS_GAR_DISCOVER_EXPORT_GAR_H_
#define EXAMPLES_ANALYTICAL_APPS_GAR_DISCOVER_EXPORT_GAR_H_

#include "gundam/graph_type/small_graph.h"
#include "gundam/graph_type/graph.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/large_graph2.h"

#include "gundam/tool/connected.h"

#include "gar/gar.h"
#include "gar/csv_gar.h"
#include "gar/gpar_new.h"

#include "prob_gar/prob_gar.h"
#include "prob_gar/csv_prob_gar.h"

#include "rule_discover/gar_discover/generate_tree.h"

namespace grape{

namespace _gar_discover {

template<typename GARType>
void ExportGARSet(const std::vector<GARType>& gar_set,
                  const std::string& output_gar_dir,
                  int level,
                  int worker_id,
                  std::string gar_prefix = "gar_level_"){

  const std::string kGarFilePrefix 
            = output_gar_dir 
             + "/" + gar_prefix + std::to_string(level)
                   + "_worker_" + std::to_string(worker_id);

  const std::string v_set_file = kGarFilePrefix + "_v_set.csv";
  const std::string e_set_file = kGarFilePrefix + "_e_set.csv";
  const std::string x_set_file = kGarFilePrefix + "_x_set.csv";
  const std::string y_set_file = kGarFilePrefix + "_y_set.csv";

  gar::WriteGARSet(gar_set, v_set_file, e_set_file,
                            x_set_file, y_set_file);
  return;
}

template<typename ProbGARType>
void ExportProbGARSet(const std::vector<ProbGARType>& prob_gar_set,
                      const std::string& output_gar_dir,
                      int level,
                      int worker_id,
                      std::string gar_prefix = "gar_level_"){

  const std::string kGarFilePrefix 
            = output_gar_dir 
             + "/" + gar_prefix + std::to_string(level)
                   + "_worker_" + std::to_string(worker_id);

  const std::string v_set_file = kGarFilePrefix + "_v_set.csv";
  const std::string e_set_file = kGarFilePrefix + "_e_set.csv";
  const std::string x_set_file = kGarFilePrefix + "_x_set.csv";
  const std::string y_set_file = kGarFilePrefix + "_y_set.csv";
  const std::string p_set_file = kGarFilePrefix + "_p_set.csv";

  prob_gar::WriteProbGARSet(prob_gar_set, v_set_file, e_set_file,
                                          x_set_file, y_set_file,
                                                      p_set_file);
  return;
}

template<typename GraphPatternType,
         typename    DataGraphType,
         typename    SubstructureLevelType>
void ExportGeneratedGars(
      const SubstructureLevelType& current_pattern_edge_size,
      const GenerateTreeLevel<GraphPatternType,
                                 DataGraphType>& current_generate_level,
            const Restriction<GraphPatternType,
                                 DataGraphType>& restriction,
      const std::string& kOutputGarDir,
      const int worker_id){

  using DataGraphTypeLiteralType = gar::LiteralInfo<GraphPatternType, 
                                                       DataGraphType>;

  using GARType = gar::GraphAssociationRule<GraphPatternType, 
                                               DataGraphType>;

  using ProbGarType = prob_gar::ProbGraphAssociationRule<
                                    GraphPatternType, 
                                       DataGraphType>;

  util::Info("export generate level: "
            + std::to_string(current_pattern_edge_size));

  // for not specifed confidence bound
  std::vector<GARType> gar_set;

  // for specifed confidence bound
  std::vector<ProbGarType> prob_gar_set;

  // #pragma omp parallel for schedule(dynamic)
  for (size_t current_generate_node_idx = 0; 
              current_generate_node_idx < static_cast<int>(current_generate_level.size()); 
              current_generate_node_idx++) {

    const auto& generate_node = current_generate_level.node(current_generate_node_idx);

    const auto& generated_pattern = generate_node.const_pattern();

    // assert(GUNDAM::Connected(generated_pattern));

    for (auto literal_tree_it  = generate_node.LiteralTreesBegin();
              literal_tree_it != generate_node.LiteralTreesEnd();
              literal_tree_it++) {

      auto& literal_tree = *literal_tree_it;

      constexpr typename GUNDAM::VertexID<LiteralTreeType>::type kLiteralTreeRootId = 0;

      using LiteralVertexHandleType = typename GUNDAM::VertexHandle<const LiteralTreeType>::type;
      
      auto root_handle = literal_tree.FindVertex(kLiteralTreeRootId);
      assert(root_handle->CountInVertex() == 0);
      std::stack<std::pair<LiteralVertexHandleType, bool>> literal_stack;

      DataGraphTypeLiteralType y_literal 
          = root_handle->template const_attribute<DataGraphTypeLiteralType>(_gar_discover::kLiteralKey);

      GARType gar(generated_pattern);

      gar.AddY(y_literal);

      gar_set.emplace_back(gar);

      if (restriction.specified_confidence_bound()) {
        float confidence
            = root_handle->template const_attribute<float>(_gar_discover::kConfKey);
        if (confidence >= restriction.confidence_bound()) {
          prob_gar_set.emplace_back(gar, confidence);
        }
      }

      #ifndef NDEBUG
      // used to find whether the same vertex has been visited twice
      std::set<LiteralVertexHandleType> visited;
      visited.emplace(root_handle);
      #endif // NDEBUG
      
      for (auto out_vertex_cit = root_handle->OutVertexBegin();
               !out_vertex_cit.IsDone();
                out_vertex_cit++) {
        literal_stack.emplace(out_vertex_cit, true);
        #ifndef NDEBUG
        visited.emplace(out_vertex_cit);
        #endif // NDEBUG
      }

      std::vector<DataGraphTypeLiteralType> existed_literals;

      while (!literal_stack.empty()) {
        auto [ current_literal_handle,
               current_literal_considered_first_time ] 
                     = literal_stack.top();
        literal_stack.pop();
        if (!current_literal_considered_first_time) {
          // recycle
          // the label of the vertex in the literal tree represents
          // the level of it 
          assert(existed_literals.size() 
              == current_literal_handle->label() - 1); // literal held in root does not contained in it
          existed_literals.pop_back();
          continue;
        }

        literal_stack.emplace(current_literal_handle, false);
        // expanding
        existed_literals.emplace_back(current_literal_handle
                ->template const_attribute<
                                DataGraphTypeLiteralType>(_gar_discover::kLiteralKey));

        // the label of the vertex in the literal tree represents
        // the level of it
        assert(existed_literals.size() 
            == current_literal_handle->label() - 1); // literal held in root does not contained in it

        GARType temp_gar(gar);
        // generate a new gar
        for (const auto& literal : existed_literals){
          temp_gar.AddX(literal);
        }
        gar_set.emplace_back(temp_gar);

        if (restriction.specified_confidence_bound()) {
          float confidence
              = current_literal_handle->template const_attribute<float>(_gar_discover::kConfKey);
          if (confidence >= restriction.confidence_bound()) {
            prob_gar_set.emplace_back(temp_gar, confidence);
          }
        }

        for (auto out_literal_vertex_cit = current_literal_handle->OutVertexBegin();
                 !out_literal_vertex_cit.IsDone();
                  out_literal_vertex_cit++) {
          // should not have visited this vertex before
          assert(visited.find(out_literal_vertex_cit) == visited.cend());
          // add to stack
          literal_stack.emplace(out_literal_vertex_cit, true);
        }
      }
    }
  }
  util::Info("gar_set.size(): " + std::to_string(gar_set.size()));
  if (!restriction.specified_confidence_bound()) {
    if (gar_set.empty()) {
      return;
    }
    ExportGARSet(gar_set, kOutputGarDir, current_pattern_edge_size, worker_id);
    return;
  }
  if (prob_gar_set.empty()) {
    return;
  }
  ExportProbGARSet(prob_gar_set, kOutputGarDir, current_pattern_edge_size, worker_id);
  return;
}

} // namespace _gar_discover

} // namespace grape

#endif // _EXPORT_GAR_H
