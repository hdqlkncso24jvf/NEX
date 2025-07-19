#ifndef EXAMPLES_ANALYTICAL_APPS_GAR_SUPP_GAR_SUPP_CONTEXT_H_
#define EXAMPLES_ANALYTICAL_APPS_GAR_SUPP_GAR_SUPP_CONTEXT_H_

#include <grape/app/context_base.h>
#include <grape/grape.h>

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

#include "util/file.h"

#include "gar/csv_gar.h"

#include "gundam/graph_type/graph.h"
#include "gundam/graph_type/small_graph.h"

namespace grape {

/**
 * @brief Context for the parallel version of GarSupp.
 *
 * @tparam FRAG_T
 */
template <typename FRAG_T>
class GarSuppContext : public VertexDataContext<FRAG_T, int64_t> {
 public:
  using oid_t = typename FRAG_T::oid_t;
  using vid_t = typename FRAG_T::vid_t;

  explicit GarSuppContext(const FRAG_T& fragment)
   : VertexDataContext<FRAG_T, int64_t>(fragment, true), 
        current_process_data_graph_idx_(0),
                     remove_charactors_{'*', '.', '"', '/', '\\', '[', ']', ':', ';', '|', ','}{
    return;
  }

  void Init(ParallelMessageManager& messages, std::string yaml_file,
            int frag_num) {
    this->yaml_file_ = yaml_file;
    this->fid_ = this->fragment().fid();
    this->frag_num_ = frag_num;
#ifdef PROFILING
    preprocess_time = 0;
    exec_time = 0;
    postprocess_time = 0;
#endif
  }

  void Output(std::ostream& os) {
#ifdef PROFILING
    VLOG(2) << "preprocess_time: " << preprocess_time << "s.";
    VLOG(2) << "exec_time: " << exec_time << "s.";
    VLOG(2) << "postprocess_time: " << postprocess_time << "s.";
#endif
  }

#ifdef PROFILING
  double preprocess_time = 0;
  double exec_time = 0;
  double postprocess_time = 0;
#endif

  using VertexIDType = uint32_t;
  using   EdgeIDType = uint32_t;

  using VertexLabelType = uint32_t;
  using   EdgeLabelType = uint32_t;

  static constexpr std::string_view kRulePosfix    = "_v.csv";
  static constexpr std::string_view kRuleSetPosfix = "_v_set.csv";

  static constexpr std::string_view kSuppPosfix    = "_supp.txt";
  static constexpr std::string_view kSuppSetPosfix = "_supp_set.txt";

  class DataGraphFile {
  private:
    std::string data_graph_dir_,
                data_graph_name_,
                ml_literal_edges_file_;

  public:
    DataGraphFile(const std::string& data_graph_dir,
                  const std::string& data_graph_name)
                   :data_graph_dir_ (data_graph_dir),
                    data_graph_name_(data_graph_name),
              ml_literal_edges_file_(""){
      return;
    }

    DataGraphFile(const std::string& data_graph_dir,
                  const std::string& data_graph_name,
                  const std::string& ml_literal_edges_file)
                   :data_graph_dir_ (data_graph_dir),
                    data_graph_name_(data_graph_name),
              ml_literal_edges_file_(ml_literal_edges_file){
      assert(this->ml_literal_edges_file_ != "");
      return;
    }
    
    const std::string& data_graph_dir() const{
      return this->data_graph_dir_;
    }
    
    const std::string& data_graph_name() const{
      return this->data_graph_name_;
    }

    const std::string& ml_literal_edges_file() const{
      return this->ml_literal_edges_file_;
    }

    const bool has_ml_literal_edges_file() const {
      return this->ml_literal_edges_file_ != "";
    }

    inline const std::string DataGraphPrefixToString() const {
      return this->data_graph_dir_ + "/" + this->data_graph_name_;
    }
  };

  template <typename   Pattern,
            typename DataGraph,
            typename LiteralInfoContainer>
  inline bool NeedPreserve(const gar::GraphAssociationRule<Pattern, 
                                                         DataGraph>& gar,
                           const LiteralInfoContainer& preserve_edge_literal_types,
                           const LiteralInfoContainer&   remove_edge_literal_types) const {

    if (preserve_edge_literal_types.empty()
       && remove_edge_literal_types.empty()){
      return true;
    }

    gar::Literal<Pattern, DataGraph>* y_literal_ptr 
                               = *gar.y_literal_set().begin();

    bool need_preserve = true;

    switch (y_literal_ptr->type()){
    case gar::LiteralType::kEdgeLiteral:
      {
        auto edge_literal_ptr
          = static_cast<gar::EdgeLiteral<Pattern, 
                                         DataGraph>*>(y_literal_ptr);
        
        auto y_info = edge_literal_ptr->info();

        auto y_info_tuple = std::make_tuple(
                              gar.pattern()
                                 .FindVertex(y_info.x_id())->label(),
                              y_info.edge_label(),
                              gar.pattern()
                                 .FindVertex(y_info.y_id())->label());

        if (!preserve_edge_literal_types.empty()) {
          // std::cout<<"ctx.preserve_edge_literal_types_ is not empty"<<std::endl;
          if (preserve_edge_literal_types.find(y_info_tuple)
           == preserve_edge_literal_types.end()){
            // does not 
            need_preserve = false;
            break;
          }
        }
        
        if (!remove_edge_literal_types.empty()) {
          // std::cout<<"ctx.remove_edge_literal_types_ is not empty"<<std::endl;
          if (remove_edge_literal_types.find(y_info_tuple)
           != remove_edge_literal_types.end()){
            need_preserve = false;
            break;
          }
        }
      }
      break;
    case gar::LiteralType::kVariableLiteral:
    case gar::LiteralType::kAttrValueLiteral:
    case gar::LiteralType::kConstantLiteral:
    default:
      need_preserve = false;
      // assert(false);
    }
    return need_preserve;
  }

  // using Pattern   = GUNDAM::LargeGraph2<VertexIDType, VertexLabelType, std::string,
  //                                         EdgeIDType,   EdgeLabelType, std::string>;
  using Pattern = GUNDAM::SmallGraph<VertexIDType, VertexLabelType, 
                                       EdgeIDType,   EdgeLabelType>;
  // using DataGraph = GUNDAM::LargeGraph2<VertexIDType, VertexLabelType, std::string,
  //                                         EdgeIDType,   EdgeLabelType, std::string>;

  // using Pattern = GUNDAM::SmallGraph<VertexIDType, VertexLabelType, 
  //                                      EdgeIDType,   EdgeLabelType>;

  using DataGraph = GUNDAM::Graph<
      GUNDAM::SetVertexIDType<VertexIDType>,
      GUNDAM::SetVertexAttributeStoreType<GUNDAM::AttributeType::kGrouped>,
      // GUNDAM::SetVertexAttributeStoreType<GUNDAM::AttributeType::kSeparated>,
      GUNDAM::SetVertexLabelType<VertexLabelType>,
      GUNDAM::SetVertexLabelContainerType<GUNDAM::ContainerType::Vector>,
      GUNDAM::SetVertexIDContainerType<GUNDAM::ContainerType::Map>,
      GUNDAM::SetVertexPtrContainerType<GUNDAM::ContainerType::Vector>,
      GUNDAM::SetEdgeLabelContainerType<GUNDAM::ContainerType::Vector>,
      GUNDAM::SetVertexAttributeKeyType<std::string>,
      GUNDAM::SetEdgeIDType<EdgeIDType>,
      GUNDAM::SetEdgeAttributeStoreType<GUNDAM::AttributeType::kGrouped>,
      // GUNDAM::SetEdgeAttributeStoreType<GUNDAM::AttributeType::kSeparated>,
      GUNDAM::SetEdgeLabelType<EdgeLabelType>,
      GUNDAM::SetEdgeAttributeKeyType<std::string>>;

  using GARPatternToDataGraph 
      = gar::GraphAssociationRule<Pattern, DataGraph>;

  std::string yaml_file_;
  int fid_;
  int frag_num_;

  // extra posfix when support candiadate is specified in
  // data graph
  std::string supp_candidate_posfix_;

  // process num for each worker
  std::map<int, int> process_num_;

  // the prefix or prefixes of the  rules
  std::set<std::string> rule_prefix_set_;

  // specify what to do
  std::string action_;

  // specify which kind of right-hand-side literals
  // should be removed or preserved
  // it should be noticed that only one set would be
  // non-empty
  std::set<std::tuple<VertexLabelType,
                        EdgeLabelType,
                      VertexLabelType>> 
            preserve_edge_literal_types_,
              remove_edge_literal_types_;

  // once a literal is specified to be preserved
  // the candidate of its support in data graph could be
  // denoted
  std::set<std::vector<VertexIDType>> supp_candidates_;

  // the gar dirs
  std::vector<std::string> gar_dirs_;

  double time_limit_;
  double time_limit_per_supp_;

  inline void AddDataGraphFile(const std::string& data_graph_dir,
                               const std::string& data_graph_name){
    this->data_graph_files_.emplace_back(data_graph_dir,
                                         data_graph_name);
    return;
  }

  inline void AddDataGraphFile(const std::string& data_graph_dir,
                               const std::string& data_graph_name,
                               const std::string& ml_literal_edges_file){
    this->data_graph_files_.emplace_back(data_graph_dir,
                                         data_graph_name,
                                         ml_literal_edges_file);
    return;
  }

  inline void ResetProcessingDataGraph() {
    this->current_process_data_graph_idx_ = 0;
    return;
  }

  inline const DataGraphFile& CurrentProcessDataGraphFile() const {
    assert(!this->AllDataGraphProcessed());
    return this->data_graph_files_[this->current_process_data_graph_idx_];
  }

  inline void MoveToNextDataGraph() {
    assert(this->current_process_data_graph_idx_ 
         < this->data_graph_files_.size());
    this->current_process_data_graph_idx_++;
    return;
  }

  inline bool AllDataGraphProcessed() const {
    assert(this->current_process_data_graph_idx_ 
        <= this->data_graph_files_.size());
    return this->current_process_data_graph_idx_ 
        == this->data_graph_files_.size();
  }

  inline std::vector<std::string> QualifiedGarPathSet(const std::string& gar_dir, bool has_support_file) const {
    std::vector<std::string> qualified_gar_path_set_in_one_dir;

    std::set<std::string> dir_files;

    util::GetFiles(gar_dir, dir_files);
    for (const auto rule_prefix : this->rule_prefix_set_){
      for (auto gar_file : dir_files){
        const bool isPrefix = rule_prefix.size() 
                              <= gar_file.size() 
            && std::mismatch(rule_prefix.begin(), 
                             rule_prefix. end (),
                                gar_file.begin(), 
                                gar_file. end ()).first == rule_prefix.end();
        const bool isGarPosfix = kRulePosfix.size() 
                                 <= gar_file.size() 
               && std::mismatch(kRulePosfix.begin(), 
                                kRulePosfix. end (),
                                   gar_file. end () - kRulePosfix.size(), 
                                   gar_file. end ()).first == kRulePosfix.end();
        if (!isPrefix || !isGarPosfix){
          continue;
        }

        gar_file = gar_file.substr(0, gar_file.length()
                                 - kRulePosfix.size());

        const std::string kGarName = gar_dir + "/" + gar_file;

        const std::string kSuppFileName 
                    = this->SuppFileOnCurrentDataGraph(kGarName);

        if (has_support_file){
          // select all gar with support file
          if (!util::ExistsFile(kSuppFileName)){
            continue;
          }
        }
        else{
          // select all gar without support file
          if (util::ExistsFile(kSuppFileName)){
            continue;
          }
        }

        if (this->preserve_edge_literal_types_.empty()
         && this->  remove_edge_literal_types_.empty()){
          qualified_gar_path_set_in_one_dir.emplace_back(kGarName);
          continue;
        }

        GARPatternToDataGraph gar;
        
        const std::string 
          gar_v_file = kGarName + "_v.csv",
          gar_e_file = kGarName + "_e.csv",
          gar_x_file = kGarName + "_x.csv",
          gar_y_file = kGarName + "_y.csv";
        auto res = gar::ReadGAR(gar, gar_v_file, 
                                     gar_e_file,
                                     gar_x_file, 
                                     gar_y_file);
        std::cout<<"GAR loaded"<<std::endl;
        if (res < 0) {
          std::cout << "Read GAR error: " << res << std::endl;
          continue;
        }

        if (gar.y_literal_set().Count() != 1){
          std::cout<<" has illegal literals in y : "
                  << gar.y_literal_set().Count() <<std::endl;
          continue;
        }

        if (!NeedPreserve(gar, this->preserve_edge_literal_types_,
                               this->  remove_edge_literal_types_)){
          std::cout<<"gar does not need to be preserved"<<std::endl;
          continue; 
        }
        qualified_gar_path_set_in_one_dir.emplace_back(kGarName);
      }
    }
    return qualified_gar_path_set_in_one_dir;
  }

  template <typename GraphPatternType,
            typename    DataGraphType>
  int LoadGars(const std::vector<std::vector<std::string>>& gar_name_set,
                     std::vector<std::tuple<std::string, // gar_supp_file_name
                                            std::string, // gar_name
                                            gar::GraphAssociationRule<
                                                 GraphPatternType,
                                                    DataGraphType>
                                           >
                                >& gar_set){
    using GarType = gar::GraphAssociationRule<GraphPatternType,
                                                 DataGraphType>;
    gar_set.clear();
    for (const auto& gar_name : gar_name_set) {
      assert(!gar_name.empty());
      const std::string& kGarFile = *(gar_name.begin());
      if (gar_name.size() == 1) {
        GarType gar;
        gar::ReadGAR(gar, kGarFile + "_v.csv",
                          kGarFile + "_e.csv",
                          kGarFile + "_x.csv",
                          kGarFile + "_y.csv");
        const std::string kGarSuppFileName 
                      = this->SuppFileOnCurrentDataGraph(kGarFile);
        gar_set.emplace_back(kGarSuppFileName, kGarFile, std::move(gar));
        continue;
      }
      assert(gar_name.size() > 1);
      std::set<std::string> gar_name_set_to_load;
      for (int gar_idx = 1; gar_idx < gar_name.size(); gar_idx++){
        auto [gar_name_set_to_load_it,
              gar_name_set_to_load_ret]
            = gar_name_set_to_load.emplace(gar_name[gar_idx]);
        if (!gar_name_set_to_load_ret){
          std::cout << "duplicated gar name: " << gar_name[gar_idx] << std::endl;
          return -1;
        }
      }
      assert(gar_name_set_to_load.size() == gar_name.size() - 1);

      std::vector<  GarType  > temp_gar_set;
      std::vector<std::string> temp_gar_name_set;
      gar::ReadGARSet(temp_gar_set, 
                      temp_gar_name_set,
                           kGarFile + "_v_set.csv",
                           kGarFile + "_e_set.csv",
                           kGarFile + "_x_set.csv",
                           kGarFile + "_y_set.csv");

      const std::string kGarSuppSetFileName 
                    = this->SuppSetFileOnCurrentDataGraph(kGarFile);
      assert(temp_gar_set.size() 
          == temp_gar_name_set.size());
      gar_set.reserve(gar_set.size() + gar_name_set_to_load.size());
      for (int gar_idx = 0; gar_idx < temp_gar_set.size(); gar_idx++){
        const std::string& kGarName = temp_gar_name_set[gar_idx];
        if (gar_name_set_to_load.find(kGarName) 
         == gar_name_set_to_load.end()) {
          // this gar does not need to be loaded
          continue;
        }
        // this gar need to be loaded
        gar_set.emplace_back(kGarSuppSetFileName,
                             temp_gar_name_set[gar_idx],
                                  temp_gar_set[gar_idx]);
      }
    }
    return 0;
  }

  int CollectEvaluatedGar(const std::string& supp_set_file,
                        std::set<std::string>& evaluated_gar_name) const {
    evaluated_gar_name.clear();
    if (!util::ExistsFile(supp_set_file)){
      // does not have this file 
      return 0;
    }
    std::ifstream supp_file(supp_set_file);

    int counter = 0;
    while (supp_file) {
      std::string s;
      if (!std::getline( supp_file, s ))
        break;

      std::istringstream ss( s );
      std::vector <std::string> record;
      std::string buf; 
      while (ss >> buf)
        record.emplace_back(buf);

      if (record.size() != 3) {
        std::cout << "illegal support set file: "
                  << supp_set_file 
                  << " at line: " << counter << std::endl;
        for (const auto& rec : record) {
          std::cout << "\t" << rec;
        }
        std::cout << std::endl;
        return -1;
      }

      const std::string& gar_name = record[0];

      auto [ evaluated_gar_name_it,
             evaluated_gar_name_ret ]
           = evaluated_gar_name.emplace(gar_name);

      if (!evaluated_gar_name_ret) {
        std::cout << "duplicate gar_name: "   << gar_name
                  << " in support set file: " << supp_set_file
                  << std::endl;
        return -1;
      }
      counter++;
    }
    return 0;
  }

  inline std::vector<std::vector<std::string>> 
    QualifiedGarSet(const std::string& gar_dir, bool has_support_file) const {
    std::vector<std::vector<std::string>> qualified_gar_path_set_in_one_dir;

    std::set<std::string> dir_files;

    util::GetFiles(gar_dir, dir_files);
    for (const auto rule_prefix : this->rule_prefix_set_){
      for (auto gar_file : dir_files){
        const bool isPrefix = rule_prefix.size() 
                              <= gar_file.size() 
             && std::mismatch(rule_prefix.begin(), 
                              rule_prefix. end (),
                                 gar_file.begin(), 
                                 gar_file. end ()).first == rule_prefix.end();
        const bool isGarPosfix = kRulePosfix.size() 
                                 <= gar_file.size() 
                && std::mismatch(kRulePosfix.begin(), 
                                 kRulePosfix. end (),
                                    gar_file.end() - kRulePosfix.size(), 
                                    gar_file.end()).first == kRulePosfix.end();
        if (!isPrefix){
          continue;
        }

        if (isGarPosfix){
          // load seperated single gar
          gar_file = gar_file.substr(0, gar_file.length()
                                   - kRulePosfix.size());

          const std::string kGarName = gar_dir + "/" + gar_file;

          const std::string kSuppFileName 
                     = this->SuppFileOnCurrentDataGraph(kGarName);

          if (has_support_file){
            // select all gar with support file
            if (!util::ExistsFile(kSuppFileName)){
              continue;
            }
          }
          else{
            // select all gar without support file
            if (util::ExistsFile(kSuppFileName)){
              continue;
            }
          }

          if (this->preserve_edge_literal_types_.empty()
           && this->  remove_edge_literal_types_.empty()){
            qualified_gar_path_set_in_one_dir.emplace_back(std::vector<std::string>{kGarName});
            continue;
          }

          GARPatternToDataGraph gar;
          
          const std::string 
            gar_v_file = kGarName + "_v.csv",
            gar_e_file = kGarName + "_e.csv",
            gar_x_file = kGarName + "_x.csv",
            gar_y_file = kGarName + "_y.csv";
          auto res = gar::ReadGAR(gar, gar_v_file, 
                                       gar_e_file,
                                       gar_x_file, 
                                       gar_y_file);
          std::cout<<"GAR loaded"<<std::endl;
          if (res < 0) {
            std::cout << "Read GAR error: " << res << std::endl;
            continue;
          }

          if (gar.y_literal_set().Count() != 1){
            std::cout<<" has illegal literals in y : "
                    << gar.y_literal_set().Count() <<std::endl;
            continue;
          }

          if (!NeedPreserve(gar, this->preserve_edge_literal_types_,
                                 this->  remove_edge_literal_types_)){
            std::cout<<"gar does not need to be preserved"<<std::endl;
            continue; 
          }
          qualified_gar_path_set_in_one_dir.emplace_back(std::vector<std::string>{kGarName});
          continue;
        }

        const bool isGarSetPosfix = kRuleSetPosfix.size() 
                                       <= gar_file.size() 
                   && std::mismatch(kRuleSetPosfix.begin(), 
                                    kRuleSetPosfix. end (),
                                    gar_file.end()         - kRuleSetPosfix.size(), 
                                    gar_file.end()).first == kRuleSetPosfix.end();
        if (!isGarSetPosfix){
          continue;
        }
        gar_file = gar_file.substr(0, gar_file.length()
                              - kRuleSetPosfix.size());

        const std::string kGarName = gar_dir + "/" + gar_file;
        std::vector<GARPatternToDataGraph> gar_set;
        std::vector<std::string> gar_name_list;
        const std::string 
          gar_v_set_file = kGarName + "_v_set.csv",
          gar_e_set_file = kGarName + "_e_set.csv",
          gar_x_set_file = kGarName + "_x_set.csv",
          gar_y_set_file = kGarName + "_y_set.csv";
        auto res = gar::ReadGARSet(gar_set, 
                                   gar_name_list,
                                   gar_v_set_file,
                                   gar_e_set_file,
                                   gar_x_set_file,
                                   gar_y_set_file);

        assert(gar_set.size() == gar_name_list.size());
        const std::string kSuppSetFileName
                   = this->SuppSetFileOnCurrentDataGraph(kGarName);

        std::set<std::string> evaluated_gar_name_set;

        auto ret = this->CollectEvaluatedGar(kSuppSetFileName, 
                                             evaluated_gar_name_set);
        if (ret < 0){
          return std::vector<std::vector<std::string>>();
        }
        
        std::vector<std::string> gar_process;
        gar_process.emplace_back(kGarName);
        for (int gar_idx = 0; gar_idx < gar_name_list.size(); gar_idx++){
          const std::string& gar_name = gar_name_list[gar_idx];
          if (!has_support_file
           && (evaluated_gar_name_set.find(gar_name)
            != evaluated_gar_name_set.end())){
            // to collect gar that has not been evaluted
            // but this gar has been evaluated
            continue;
          }
          if ( has_support_file
           && (evaluated_gar_name_set.find(gar_name)
            == evaluated_gar_name_set.end())){
            // to collect gar that has been evaluted
            // but this gar has not been evaluated
            continue;
          }
          // this gar has not been evaluted
          auto& gar = gar_set[gar_idx];

          if (!NeedPreserve(gar, this->preserve_edge_literal_types_,
                                 this->  remove_edge_literal_types_)){
            // this gar does not need to be preserved
            std::cout << "gar: " << kGarName + "_" + gar_name 
                      << " does not need to be preserved"<<std::endl;
            continue; 
          }
          // this gar need to be preserved
          gar_process.emplace_back(gar_name);
        }
        assert(gar_process.size() >= 1);
        if (gar_process.size() == 1){
          // no gar in this gar set need to be evaluated
          continue;
        }
        qualified_gar_path_set_in_one_dir.emplace_back(std::move(gar_process));
      }
    }
    return qualified_gar_path_set_in_one_dir;
  }

  inline std::string SuppFilePrefixOnCurrentDataGraph(
            const std::string& gar_path) const {

    std::string data_graph_dir_name = this->CurrentProcessDataGraphFile().data_graph_dir();
    while (data_graph_dir_name.back() == '/'){
      data_graph_dir_name.pop_back();
    }
    std::size_t found = data_graph_dir_name.find_last_of('/');

    data_graph_dir_name = data_graph_dir_name.substr(found+1);

    std::string supp_posfix = this->supp_candidate_posfix_ // could be ""
                            + "_" + data_graph_dir_name;

    for (const auto& charactor : this->remove_charactors_){
      std::replace(supp_posfix.begin(), 
                   supp_posfix.end(), charactor, '_');
    }
    supp_posfix = supp_posfix + "_" 
                              + this->CurrentProcessDataGraphFile().data_graph_name();

    return gar_path + supp_posfix;
  }

  inline std::string SuppFileOnCurrentDataGraph(const std::string& gar_path) const {
    if (this->pivot_all_vertexes_) {
      return this->SuppFilePrefixOnCurrentDataGraph(gar_path) + "_pivot_all_vertexes_supp.txt";
    }
    return this->SuppFilePrefixOnCurrentDataGraph(gar_path) + "_supp.txt";
  }

  inline std::string SuppSetFileOnCurrentDataGraph(const std::string& gar_path) const{
    if (this->pivot_all_vertexes_) {
      return this->SuppFilePrefixOnCurrentDataGraph(gar_path) + "_pivot_all_vertexes_supp_set.txt";
    }
    return this->SuppFilePrefixOnCurrentDataGraph(gar_path) + "_supp_set.txt";
  }

  inline void GatherSuppOnCurrentDataGraph(
          const std::vector<std::string>& qualified_gar_path,
                std::  map <std::string, 
                            std::pair<int, int>>& qualified_gar_supp_set) const {
    assert(qualified_gar_path.size() >= 1);
    assert(qualified_gar_supp_set.empty());
    std::cout << "qualified_gar_path.size(): "
              <<  qualified_gar_path.size()
              << std::endl;
    if (qualified_gar_path.size() == 1) {
      const std::string kSuppFile = this->SuppFileOnCurrentDataGraph(*qualified_gar_path.begin());
      assert(util::ExistsFile(kSuppFile));
      std::ifstream supp_file(kSuppFile);
      
      std::vector<std::string> vect;
      vect.reserve(3);
      std::string str;
      while (supp_file >> str) {
        std::cout << "str: " << str << std::endl;
        vect.emplace_back(str);
      }
      if (vect.size() != 3){
        std::cout << "illegal file: "
                  << kSuppFile << std::endl;
        return;
      }
      assert(std::stoi(vect[1]) 
          <= std::stoi(vect[2]));
      auto [qualified_gar_supp_set_it,
            qualified_gar_supp_set_ret]
          = qualified_gar_supp_set.emplace(vect[0], 
                       std::pair(std::stoi(vect[1]), 
                                 std::stoi(vect[2])));
      // should added successfully
      if (!qualified_gar_supp_set_ret){
        std::cout << "* duplicate gar: "
                  << vect[0] << std::endl;
      }
      return;
    }
    assert(qualified_gar_path.size() > 1);
    const std::string kSuppSetFile 
               = this->SuppSetFileOnCurrentDataGraph(*qualified_gar_path.begin());

    assert(util::ExistsFile(kSuppSetFile));
    std::ifstream supp_file(kSuppSetFile);

    while (supp_file) {
      std::string s;
      if (!std::getline( supp_file, s ))
        break;

      std::istringstream ss( s );
      std::vector <std::string> record;
      std::string buf; 
      while (ss >> buf)
        record.emplace_back(buf);

      std::vector <std::string> vect;
      vect.reserve(3);
      for (const auto& str : record){
        vect.emplace_back(str);
      }
      if (vect.size() != 3){
        std::cout << "illegal file: "
                  << kSuppSetFile << std::endl;
        return;
      }
      assert(std::stoi(vect[1]) 
          <= std::stoi(vect[2]));
      const std::string kGarName = (*qualified_gar_path.begin()) + "@" + vect[0];
      std::cout << " kGarName: "
                <<   kGarName << std::endl;
      auto [ qualified_gar_supp_set_it,
             qualified_gar_supp_set_ret]
           = qualified_gar_supp_set.emplace(kGarName, 
                        std::pair(std::stoi(vect[1]), 
                                  std::stoi(vect[2])));
      if (!qualified_gar_supp_set_ret){
        std::cout << "# duplicate gar: "
                  << kGarName << std::endl;
        return;
      }
    }
    return;
  }

  inline std::string ConfFilePathOnCurrentDataGraph(const std::string& gar_dir)  const {
    std::string data_graph_dir_name = this->CurrentProcessDataGraphFile().data_graph_dir();
    while (data_graph_dir_name.back() == '/'){
      data_graph_dir_name.pop_back();
    }
    std::size_t found = data_graph_dir_name.find_last_of('/');
    std::cout<<data_graph_dir_name<<std::endl;

    data_graph_dir_name = data_graph_dir_name.substr(found+1);

    std::string confidence_file_path = gar_dir + this->supp_candidate_posfix_
                                          + "_" + data_graph_dir_name;

    for (const auto& charactor : this->remove_charactors_){
      std::replace(confidence_file_path.begin(), 
                   confidence_file_path. end (), charactor, '_');
    }
    confidence_file_path = "./" + confidence_file_path 
                          + "_" + this->CurrentProcessDataGraphFile().data_graph_name();

    if (!this->preserve_edge_literal_types_.empty()){
      confidence_file_path += "_preserve";
      for (const auto& edge_literal_type 
            : this->preserve_edge_literal_types_){
        confidence_file_path += "_" + std::to_string(std::get<0>(edge_literal_type)) 
                              + "_" + std::to_string(std::get<1>(edge_literal_type)) 
                              + "_" + std::to_string(std::get<2>(edge_literal_type));
      }
    }
    else if (!this->remove_edge_literal_types_.empty()){
      confidence_file_path += "_remove";
      for (const auto& edge_literal_type 
              : this->remove_edge_literal_types_){
        confidence_file_path += "_" + std::to_string(std::get<0>(edge_literal_type)) 
                              + "_" + std::to_string(std::get<1>(edge_literal_type)) 
                              + "_" + std::to_string(std::get<2>(edge_literal_type));
      }
    }
    if (this->pivot_all_vertexes_) {
      confidence_file_path += "_pivot_all_vertexes";
    }
    confidence_file_path += "_confidence.txt";
    return confidence_file_path;
  }

  bool pivot_all_vertexes_;

 private:
  // mark the current processing data graph
  int current_process_data_graph_idx_;

  // the data graph files
  std::vector<DataGraphFile> data_graph_files_;

  std::vector<char> remove_charactors_;
};

}  // namespace grape

#endif  // EXAMPLES_ANALYTICAL_APPS_GAR_SUPP_GAR_SUPP_CONTEXT_H_
