#include <iostream>
#include <string>

#include "gtest/gtest.h"

#include "gundam/tool/diameter.h"

#include "gundam/graph_type/small_graph.h"
#include "gundam/graph_type/large_graph.h"
#include "gundam/graph_type/large_graph2.h"
#include "gundam/graph_type/graph.h"

#include "test_data/test_pattern_set.h"

template<class GraphType>
void TestDiameter(){

  GraphType g;

  std::cout << "GraphType: " << typeid(g).name() << std::endl;

  ConstructGraph0(g);
  auto diameter_0 = GUNDAM::Diameter<true>(g);
  ASSERT_EQ(diameter_0, 1);

  ConstructGraph1(g);
  auto diameter_1 = GUNDAM::Diameter<true>(g);
  ASSERT_EQ(diameter_1, 1);
  
  ConstructGraph2(g);
  auto diameter_2 = GUNDAM::Diameter<true>(g);
  ASSERT_EQ(diameter_2, 2);

  ConstructGraph3(g);
  auto diameter_3 = GUNDAM::Diameter<true>(g);
  ASSERT_EQ(diameter_3, 4);

  ConstructGraph4(g);
  auto diameter_4 = GUNDAM::Diameter<true>(g);
  ASSERT_EQ(diameter_4, 4);
  
  ConstructGraph5(g);
  auto diameter_5 = GUNDAM::Diameter<true>(g);
  ASSERT_EQ(diameter_5, 2);

  ConstructGraph6(g);
  auto diameter_6 = GUNDAM::Diameter<true>(g);
  ASSERT_EQ(diameter_6, 2);

  ConstructGraph7(g);
  auto diameter_7 = GUNDAM::Diameter<true>(g);
  ASSERT_EQ(diameter_7, 3);
  return;
}

TEST(TestGUNDAM, TestDiameter) {
  using namespace GUNDAM;

  using G1 = LargeGraph<uint32_t, uint32_t, std::string, 
                        uint64_t, uint32_t, std::string>;

  using G2 = LargeGraph2<uint32_t, uint32_t, std::string, 
                         uint64_t, uint32_t, std::string>;  

  using G3 = SmallGraph<uint32_t, uint32_t, 
                        uint64_t, uint32_t>;

  using G4 = Graph<SetVertexIDType<uint32_t>, 
                   SetVertexLabelType<uint32_t>,
                   SetVertexAttributeKeyType<std::string>, 
                   SetEdgeIDType<uint64_t>,
                   SetEdgeLabelType<uint32_t>, 
                   SetEdgeAttributeKeyType<std::string>>;

  using G5 = Graph<SetVertexIDType<uint32_t>, 
                   SetVertexLabelType<uint32_t>,
                   SetVertexAttributeKeyType<std::string>,
                   SetEdgeIDType<uint64_t>,
                   SetEdgeLabelType<uint32_t>, 
                   SetEdgeAttributeKeyType<std::string>, 
                   SetVertexLabelContainerType<GUNDAM::ContainerType::Map>,
                   SetVertexIDContainerType<GUNDAM::ContainerType::Map>>;

  using G6 = Graph<SetVertexIDType<uint32_t>, 
                   SetVertexLabelType<uint32_t>,
                   SetVertexAttributeKeyType<std::string>,
                   SetEdgeIDType<uint64_t>,
                   SetEdgeLabelType<uint32_t>, 
                   SetEdgeAttributeKeyType<std::string>, 
                   SetVertexPtrContainerType<GUNDAM::ContainerType::Map>,
                   SetEdgeLabelContainerType<GUNDAM::ContainerType::Map>>;

  using G7 = Graph<SetVertexIDType<uint32_t>, 
                   SetVertexLabelType<uint32_t>,
                   SetVertexAttributeKeyType<std::string>,
                   SetEdgeIDType<uint64_t>,
                   SetEdgeLabelType<uint32_t>, 
                   SetEdgeAttributeKeyType<std::string>, 
                   SetVertexLabelContainerType<GUNDAM::ContainerType::Map>,
                   SetVertexIDContainerType<GUNDAM::ContainerType::Map>, 
                   SetVertexPtrContainerType<GUNDAM::ContainerType::Map>,
                   SetEdgeLabelContainerType<GUNDAM::ContainerType::Map>>;

  TestDiameter<G1>();
  TestDiameter<G2>();
  TestDiameter<G3>();
  TestDiameter<G4>();
  TestDiameter<G5>();
  TestDiameter<G6>();
  TestDiameter<G7>();
}