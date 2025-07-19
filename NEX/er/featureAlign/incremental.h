#ifndef HER_INC_H_
#define HER_INC_H_

#include "her/spair.h"
#include <ostream>
#include <queue>
#include <vector>
#include <stack>

namespace her
{
	template <typename GRAPH>
	class DepGraph
	{
		using vertex_t = typename GRAPH::vertex_t;
		using nodePair = typename her::VertexPair<vertex_t>;
		using edge = typename std::pair<nodePair, nodePair>;
		using SCC = typename std::pair<std::unordered_set<nodePair>, std::unordered_set<edge>>;

	public:
		void insertNode(nodePair p)
		{
			auto u = p.u(), v = p.v();
			nodeSet.insert(p);
			if (vertex_to_matches.find(v) == vertex_to_matches.end())
			{
				vertex_to_matches[v] = {};
			}
			if (vertex_to_matches.find(u) == vertex_to_matches.end())
			{
				vertex_to_matches[u] = {};
			}
			vertex_to_matches[u].insert(p);
			vertex_to_matches[v].insert(p);
		}

		void insertEdge(nodePair p, nodePair q)
		{
			if (nodeSet.find(p) == nodeSet.end())
			{
				insertNode(p);
				nextNodes[p] = {};
				precNodes[p] = {};
			}
			if (nodeSet.find(q) == nodeSet.end())
			{
				insertNode(q);
				nextNodes[q] = {};
				precNodes[q] = {};
			}
			nextNodes[p].insert(q);
			precNodes[q].insert(p);
			edgeSet.insert(edge(p, q));
		}

		void deleteEdge(nodePair p, nodePair q)
		{
			nextNodes[p].erase(q);
			precNodes[q].erase(p);
			edgeSet.erase(edge(p, q));
		}

		void loadDepGraph(std::unordered_set<nodePair> candidates, std::string fileName)
		{
			std::ifstream dp(fileName);
			for (auto i : candidates)
			{
				insertNode(i);
			}
			vertex_t u1, v1, u2, v2;
			while (dp >> u1 >> v1 >> u2 >> v2)
			{
				insertEdge(nodePair(u1, v1), nodePair(u2, v2));
			}
			dp.close();
		}

		void storeDepGraph(std::string fileName)
		{
			std::ofstream dp(fileName);
			vertex_t u1, v1, u2, v2;
			for (auto e : edgeSet)
			{
				u1 = e.first.u();
				v1 = e.first.v();
				u2 = e.second.u();
				v2 = e.second.v();
				dp << u1 << " " << v1 << " " << u2 << " " << v2 << std::endl;
			}
			dp.close();
		}

		std::unordered_set<nodePair> getRelatedMatches(std::unordered_set<vertex_t> affected_vertexes)
		{
			std::unordered_set<nodePair> res;
			for (auto vertex : affected_vertexes)
			{
				for (auto match : vertex_to_matches[vertex])
				{
					res.insert(match);
				}
			}
			return res;
		}

		std::unordered_set<nodePair> getAffectedMatches(std::unordered_set<vertex_t> affected_vertexes)
		{
			std::unordered_set<nodePair> allAffectedMatches = getRelatedMatches(affected_vertexes);
			std::unordered_set<nodePair> allMatches = allAffectedMatches;
			std::unordered_set<nodePair> newMatchesSet = allAffectedMatches;
			long unsigned int prevSize = 0;
			while (prevSize != allMatches.size())
			{
				std::unordered_set<nodePair> tmp;
				prevSize = allMatches.size();
				for (auto i : newMatchesSet)
				{
					auto prevOfi = getPrec(i);
					for (auto j : prevOfi)
					{
						tmp.insert(j);
						allMatches.insert(j);
					}
				}
				newMatchesSet = tmp;
			}
			return allMatches;
		}

		std::unordered_set<nodePair> getPrec(nodePair p)
		{
			return precNodes[p];
		}

		std::unordered_set<nodePair> getNext(nodePair p)
		{
			return nextNodes[p];
		}

		int getNodesNum()
		{
			return nodeSet.size();
		}

		int getEdgesNum()
		{
			return edgeSet.size();
		}

		std::unordered_set<nodePair> getNodeset()
		{
			return nodeSet;
		}

		std::unordered_set<edge> getEdgeset()
		{
			return edgeSet;
		}

		std::unordered_set<edge> judge(std::unordered_set<nodePair> &nodes, std::unordered_set<edge> &newEdgeLists)
		{
			std::unordered_set<edge> rt_edges = {};
			for (auto i = newEdgeLists.begin(); i != newEdgeLists.end();)
			{
				nodePair u = (*i).first, v = (*i).second;
				if (nodes.find(u) != nodes.end() && nodes.find(v) != nodes.end())
				{
					rt_edges.insert(*i);
					newEdgeLists.erase(*i++);
				}
				else if (nodes.find(u) == nodes.end() || nodes.find(v) == nodes.end())
				{
					newEdgeLists.erase(*i++);
				}
				else
				{
					++i;
				}
			}
			return rt_edges;
		}

		// using tarjan algorithm to find all scces
		void tarjan(nodePair x, int tot, std::unordered_map<nodePair, int> &DFN,
					std::unordered_map<nodePair, int> &LOW, std::stack<nodePair> &nodepair_stack,
					std::unordered_map<nodePair, int> &is_visit, std::vector<SCC> &rt_getscc,
					std::unordered_map<nodePair, int> &can_finish, std::unordered_set<edge> &newEdgeLists)
		{
			if (can_finish[x] == 1)
			{
				return;
			}
			DFN[x] = LOW[x] = ++tot; // initialize new nodes
			nodepair_stack.push(x);	 // push in stack
			is_visit[x] = 1;
			auto iter = nextNodes[x].begin(); // get all outwards nodes
			while (iter != nextNodes[x].end())
			{
				if (!DFN[*iter])
				{																								 // not visited
					tarjan(*iter, tot, DFN, LOW, nodepair_stack, is_visit, rt_getscc, can_finish, newEdgeLists); // dfs
					LOW[x] = std::min(LOW[x], LOW[*iter]);
				}
				else if (is_visit[*iter])
				{ // in stack and visited。
					LOW[x] = std::min(LOW[x], DFN[*iter]);
				}
				iter++;
			}
			if (LOW[x] == DFN[x])
			{
				if (false && nodepair_stack.top() == x)
				{
					nodepair_stack.pop();
				}
				else
				{
					std::unordered_set<nodePair> rt_node;
					while (!(nodepair_stack.top() == x))
					{
						can_finish[nodepair_stack.top()] = 1;
						rt_node.insert(nodepair_stack.top());
						nodepair_stack.pop();
					}
					can_finish[nodepair_stack.top()] = 1;
					rt_node.insert(nodepair_stack.top());
					nodepair_stack.pop();
					std::unordered_set<edge> rt_edge = judge(rt_node, newEdgeLists);
					if (!rt_edge.empty())
					{
						rt_getscc.push_back(SCC(rt_node, rt_edge));
					}
				}
			}
		}

		std::vector<SCC> getSCC(std::unordered_set<edge> newEdgeLists)
		{
			std::vector<SCC> rt_getscc;
			std::unordered_map<nodePair, int> can_finish; // whether can end directly
			for (auto i = nodeSet.begin(); i != nodeSet.end(); i++)
			{
				can_finish[*i] = 0;
			}
			while (!newEdgeLists.empty())
			{
				int tot = 0;
				std::unordered_map<nodePair, int> DFN, LOW; // tarjan
				std::stack<nodePair> nodepair_stack;		// nodes stack
				std::unordered_map<nodePair, int> is_visit; // is in stack
				auto cur_edge = newEdgeLists.begin();
				nodePair first_node = (*cur_edge).first;
				tarjan(first_node, tot, DFN, LOW, nodepair_stack, is_visit,
					   rt_getscc, can_finish, newEdgeLists);
			}
			return rt_getscc;
		}

	private:
		std::unordered_set<nodePair> nodeSet;								  // all nodes are in this set
		std::unordered_set<edge> edgeSet;									  // all nodes are in this set
		std::unordered_map<nodePair, std::unordered_set<nodePair>> nextNodes; // all nodes' outwards edges are in this set
		std::unordered_map<nodePair, std::unordered_set<nodePair>> precNodes; // all nodes' inwards edges are in this set
		std::unordered_map<vertex_t, std::unordered_set<nodePair>> vertex_to_matches;
	};

	template <typename GRAPH, typename H_V, typename H_P, typename H_R>
	class IncProcedure
	{
		using vertex_t = typename GRAPH::vertex_t;
		using nodePair = typename her::VertexPair<vertex_t>;
		using edge = typename std::pair<nodePair, nodePair>;
		using SCC = typename std::pair<std::unordered_set<nodePair>, std::unordered_set<edge>>;
		using descendants_t = std::vector<std::pair<vertex_t, depth_t>>;

	public:
		IncProcedure(GRAPH &gd, GRAPH &g, H_V &h_v, H_P &h_p, H_R &h_r)
			: gd_(gd), g_(g), h_v_(h_v), h_p_(h_p), h_r_(h_r)
		{
		}

		void InitParams(double sigma, double delta, int k, std::unordered_set<nodePair> &result)
		{
			sigma_ = sigma;
			delta_ = delta;
			k_ = k;

			for (auto i : result)
			{
				vertex_t u = i.u(), v = i.v();

				descendants_t u_descendants;
				if (ecache_gd_.find(u) == ecache_gd_.end())
				{
					ecache_gd_[u] = h_r_(gd_, u, k_, false);
				}
				u_descendants = ecache_gd_[u];

				for (auto u1_depth_pair : u_descendants)
				{
					vertex_t u1 = u1_depth_pair.first;
					if (precMap_gd.find(u1) == precMap_gd.end())
					{
						precMap_gd[u1] = {};
					}
					precMap_gd[u1].insert(u);
					if (nextMap_gd.find(u) == nextMap_gd.end())
					{
						nextMap_gd[u] = {};
					}
					nextMap_gd[u].insert(u1);
				}

				descendants_t v_descendants;
				if (ecache_g_.find(v) == ecache_g_.end())
				{
					ecache_g_[v] = h_r_(g_, v, k_, true);
				}
				v_descendants = ecache_g_[v];

				for (auto v1_depth_pair : v_descendants)
				{
					vertex_t v1 = v1_depth_pair.first;
					if (precMap_g.find(v1) == precMap_g.end())
					{
						precMap_g[v1] = {};
					}
					precMap_g[v1].insert(v);
					if (nextMap_g.find(v) == nextMap_g.end())
					{
						nextMap_g[v] = {};
					}
					nextMap_g[v].insert(v1);
				}
			}
		}

		DepGraph<GRAPH> getDependenciesGraph(std::unordered_set<nodePair> result)
		{
			DepGraph<GRAPH> dg;
			for (auto c : result)
			{
				dg.insertNode(c);
			}
			for (auto i : result)
			{
				for (auto j : result)
				{
					vertex_t u_1 = i.u(), v_1 = i.v();
					vertex_t u_2 = j.u(), v_2 = j.v();
					if (nextMap_gd[u_1].find(u_2) != nextMap_gd[u_1].end() &&
						nextMap_g[v_1].find(v_2) != nextMap_g[v_1].end())
					{
						dg.insertEdge(i, j);
					}
				}
			}

			return dg;
		}

		std::pair<std::unordered_set<edge>, std::unordered_set<edge>> getDeltaGraph(
			std::unordered_set<nodePair> &affectedMatches, DepGraph<GRAPH> &Gamma)
		{
			std::unordered_set<edge> insertedEdges;
			std::unordered_set<edge> deletedEdges;
			std::unordered_set<edge> newEdgesSet;

			for (auto affectedMatch : affectedMatches)
			{
				vertex_t u_1 = affectedMatch.u(), v_1 = affectedMatch.v();
				std::unordered_set<nodePair> relatedMatches;
				for (auto match : Gamma.getRelatedMatches(nextMap_gd[u_1]))
				{
					relatedMatches.insert(match);
				}
				for (auto match : Gamma.getRelatedMatches(nextMap_g[v_1]))
				{
					relatedMatches.insert(match);
				}
				for (auto otherMatch : relatedMatches)
				{
					vertex_t u_2 = otherMatch.u(), v_2 = otherMatch.v();
					if (nextMap_gd[u_1].find(u_2) != nextMap_gd[u_1].end() &&
						nextMap_g[v_1].find(v_2) != nextMap_g[v_1].end())
					{
						newEdgesSet.insert(edge(affectedMatch, otherMatch));
					}
				}
			}

			std::unordered_set<edge> oldEdgesSet;
			for (auto match : affectedMatches)
			{
				for (auto outwards : Gamma.getNext(match))
				{
					oldEdgesSet.insert(edge(match, outwards));
				}
			}

			for (auto e : newEdgesSet)
			{
				if (oldEdgesSet.find(e) == oldEdgesSet.end())
				{
					insertedEdges.insert(e);
				}
			}
			for (auto e : oldEdgesSet)
			{
				if (newEdgesSet.find(e) == newEdgesSet.end())
				{
					deletedEdges.insert(e);
				}
			}
			return {insertedEdges, deletedEdges};
		}

		bool getvCache(vertex_t u, vertex_t v)
		{
			// auto u_and_v = nodePair(u, v);
			// return hvResult.find(u_and_v) != hvResult.end();
			auto key = nodePair(u, v);
			if (hv_cache.find(key) == hv_cache.end())
			{
				hv_cache[key] = h_v_(gd_, u, g_, v);
			}
			return hv_cache[key];
			// return h_v_(gd_, u, g_, v);
		}

		double getpCache(vertex_t u1, vertex_t u2, vertex_t v1, vertex_t v2)
		{
			auto u1_and_v1 = nodePair(u1, v1);
			auto u2_and_v2 = nodePair(u2, v2);
			auto key = std::make_pair(u1_and_v1, u2_and_v2);
			if (hp_cache.find(key) == hp_cache.end())
			{
				hp_cache[key] = h_p_(gd_, u1, u2, g_, v1, v2);
			}
			return hp_cache[key];
			// return h_p_(gd_, u1, u2, g_, v1, v2);
		}

		bool checkPM(vertex_t u, vertex_t v, std::unordered_set<nodePair> &psbMatches, std::unordered_set<nodePair> &S)
		{
			double sum = 0;
			descendants_t u_descendants;
			if (ecache_gd_.find(u) == ecache_gd_.end())
			{
				ecache_gd_[u] = h_r_(gd_, u, k_, false);
			}
			u_descendants = ecache_gd_[u];

			descendants_t v_descendants;
			if (ecache_g_.find(v) == ecache_g_.end())
			{
				ecache_g_[v] = h_r_(g_, v, k_, true);
			}
			v_descendants = ecache_g_[v];

			// auto begin = GetCurrentTime();
			for (auto u1_depth_pair : u_descendants)
			{
				vertex_t u1 = u1_depth_pair.first;
				depth_t u1_depth = u1_depth_pair.second;
				std::vector<std::pair<vertex_t, depth_t>> l;
				for (auto v1_depth : v_descendants)
				{
					vertex_t v1 = v1_depth.first;
					if (getvCache(u1, v1) >= sigma_)
					{
						// LOG(INFO) << h_v_(gd_, u1, g_, v1);
						l.push_back(v1_depth);
					}
				}
				// sort list by path score in descending order
				std::sort(l.begin(), l.end(),
						  [this, u, u1, v](const std::pair<vertex_t, depth_t> &a,
										   const std::pair<vertex_t, depth_t> &b)
						  {
							  return getpCache(u, u1, v, a.first) >
									 getpCache(u, u1, v, b.first);
						  });
				for (auto &v1_depth_pair : l)
				{
					vertex_t v1 = v1_depth_pair.first;
					depth_t v1_depth = v1_depth_pair.second;
					nodePair key1(u1, v1), key2(u, v);
					depth_t depth = std::max(u1_depth, v1_depth);
					if (psbMatches.find(key1) != psbMatches.end() || S.find(key2) != S.end())
					{
						sum += getpCache(u, u1, v, v1) / depth;
						if (sum >= delta_)
						{
							// checkTime += (GetCurrentTime() - begin);
							return true;
						}
						break;
					}
				}
			}
			// checkTime += (GetCurrentTime() - begin);
			return false;
		}

		std::vector<nodePair> MarkAFF(DepGraph<GRAPH> &Gamma, std::unordered_set<nodePair> &Pi,
									  edge e, std::unordered_set<nodePair> &S)
		{
			std::vector<nodePair> deltaC = {};
			std::unordered_set<nodePair> A_m = Pi;
			nodePair u1_v1 = e.first, u2_v2 = e.second;
			A_m.insert(u2_v2);
			if (!checkPM(u1_v1.u(), u1_v1.v(), A_m, S))
			{
				return deltaC;
			}
			deltaC.push_back(u1_v1);
			std::queue<nodePair> delta_uvs;
			delta_uvs.push(u1_v1);
			A_m.insert(u1_v1);
			while (!delta_uvs.empty())
			{
				nodePair cur_uv = delta_uvs.front();
				vertex_t c_u = cur_uv.u(), c_v = cur_uv.v();
				delta_uvs.pop();
				for (nodePair p_uv : S)
				{
					vertex_t p_u = p_uv.u(), p_v = p_uv.v();
					if (precMap_gd[c_u].find(p_u) != precMap_gd[c_u].end() && precMap_g[c_v].find(p_v) != precMap_g[c_v].end())
					{
						if (Pi.find(p_uv) == Pi.end() && checkPM(p_u, p_v, A_m, S))
						{
							deltaC.push_back(p_uv);
							delta_uvs.push(p_uv);
							A_m.insert(p_uv);
						}
					}
				}
			}
			return deltaC;
		}

		void topologic(std::unordered_set<nodePair> precSet, std::vector<nodePair> &nodesInOrder,
					   std::unordered_set<nodePair> &Pi, DepGraph<GRAPH> &Gamma)
		{
			std::unordered_map<nodePair, bool> isVisited, isFinish;
			for (auto i : Gamma.getNodeset())
			{
				isVisited[i] = false;
				isFinish[i] = false;
			}

			std::queue<nodePair> q;
			while (!precSet.empty())
			{
				auto iter = precSet.begin();

				q.push(*iter);
				while (!q.empty())
				{
					auto node = q.front();
					if (isFinish[node] == true)
					{
						if (isVisited[node] == false)
						{
							isVisited[node] = true;
							nodesInOrder.push_back(node);
							if (precSet.find(node) != precSet.end())
							{
								precSet.erase(node);
							}
						}
						q.pop();
					}
					for (auto outwards : Gamma.getNext(node))
					{
						if (Pi.find(outwards) != Pi.end() && !isVisited[outwards] && !isFinish[outwards])
						{
							q.push(outwards);
						}
					}
					isFinish[node] = true;
				}
			}
		}

		std::vector<nodePair> CheckCand(
			DepGraph<GRAPH> &Gamma, std::unordered_set<nodePair> &Pi,
			std::unordered_set<nodePair> C)
		{

			std::unordered_set<nodePair> deltaC = {};
			std::queue<nodePair> deltaSet = {};

			for (auto i = C.begin(); i != C.end();)
			{
				auto u_v = *i;
				if (!checkPM(u_v.u(), u_v.v(), Pi, C))
				{
					deltaC.insert(u_v);
					deltaSet.push(u_v);
					C.erase(*i++);
				}
				else
				{
					++i;
				}
			}

			while (!deltaSet.empty())
			{
				nodePair cur_uv = deltaSet.front();
				vertex_t c_u = cur_uv.u(), c_v = cur_uv.v();
				deltaSet.pop();
				for (auto i = C.begin(); i != C.end();)
				{
					nodePair p_uv = *i;
					bool needToErase = false;
					vertex_t p_u = p_uv.u(), p_v = p_uv.v();
					if (precMap_gd[c_u].find(p_u) != precMap_gd[c_u].end() &&
						precMap_g[c_v].find(p_v) != precMap_g[c_v].end())
					{
						if (Pi.find(p_uv) == Pi.end() && checkPM(p_u, p_v, Pi, C))
						{
							deltaC.insert(p_uv);
							deltaSet.push(p_uv);
							needToErase = true;
						}
					}
					if (needToErase)
					{
						C.erase(*i++);
					}
					else
					{
						++i;
					}
				}
			}

			std::vector<nodePair> result;
			for (auto &match : C)
			{
				if (deltaC.find(match) == deltaC.end())
				{
					result.push_back(match);
				}
			}
			return result;
		}

		std::pair<std::unordered_set<nodePair>, std::unordered_set<nodePair>> IncMatch(
			std::unordered_set<nodePair> &Pi, DepGraph<GRAPH> &Gamma,
			std::unordered_set<edge> deletedEdges, std::unordered_set<edge> insertedEdges)
		{

			std::unordered_set<nodePair> Pi_plus, Pi_sub, C;
			std::unordered_set<nodePair> rtSet, emptySet;
			// update Gamma
			for (edge e : deletedEdges)
			{
				Gamma.deleteEdge(e.first, e.second);
			}
			for (edge e : insertedEdges)
			{
				Gamma.insertEdge(e.first, e.second);
			}

			// process Delta_Gamma-
			for (auto e : deletedEdges)
			{
				auto u_1_v_1 = e.first, u_2_v_2 = e.second;
				if (Pi.find(u_1_v_1) == Pi.end() || Pi.find(u_2_v_2) == Pi.end())
				{
					continue;
				}
				// precSet.insert(u_1_v_1);
				// std::unordered_set<nodePair> precSet;
				// precSet.insert(u_1_v_1);
				// std::vector<nodePair> nodesInOrder;
				// topologic(precSet, nodesInOrder, Pi, Gamma);
				// for(auto u_v: nodesInOrder){
				auto u_v = u_1_v_1;
				std::unordered_set<nodePair> nodesOfS = {u_v}, newNodesSet = {u_v};
				long unsigned int prevSize = 0;
				while (prevSize != nodesOfS.size())
				{
					std::unordered_set<nodePair> tmp;
					prevSize = nodesOfS.size();
					for (auto i : newNodesSet)
					{
						auto prevOfi = Gamma.getPrec(i);
						for (auto j : prevOfi)
						{
							if (Pi.find(j) != Pi.end())
							{
								tmp.insert(j);
								nodesOfS.insert(j);
							}
						}
					}
					newNodesSet = tmp;
				}
				rtSet = DecMatch(Gamma, Pi, u_v, nodesOfS);
				for (auto m : rtSet)
				{
					Pi_sub.insert(m);
				}
				// }
			}

			// process Delta_Gamma+
			while (true)
			{
				bool needToEnd = true;
				for (auto i = insertedEdges.begin(); i != insertedEdges.end();)
				{
					auto u_1_v_1 = i->first, u_2_v_2 = i->second;
					if (Pi.find(u_1_v_1) == Pi.end() && Pi.find(u_2_v_2) == Pi.end())
					{
						++i;
					}
					else if (Pi.find(u_1_v_1) != Pi.end())
					{
						insertedEdges.erase(*i++);
					}
					else
					{
						needToEnd = false;
						std::stack<nodePair> nodesOfS;
						nodesOfS.push(u_1_v_1);
						while (!nodesOfS.empty())
						{
							auto u_v = nodesOfS.top();
							nodesOfS.pop();
							if (checkPM(u_v.u(), u_v.v(), Pi, emptySet))
							{
								Pi.insert(u_v);
								Pi_plus.insert(u_v);
								for (auto i : Gamma.getPrec(u_v))
								{
									if (Pi.find(i) == Pi.end())
									{
										nodesOfS.push(i);
									}
								}
							}
						}
						insertedEdges.erase(*i++);
					}
				}
				if (needToEnd)
				{
					break;
				}
			}

			std::vector<SCC> inserted_SCCs = Gamma.getSCC(insertedEdges);
			for (SCC S : inserted_SCCs)
			{
				std::unordered_set<nodePair> nodesOfS = S.first;
				std::unordered_set<edge> newEdges = S.second;
				for (edge e : newEdges)
				{
					C.insert(e.first);
					C.insert(e.second);
					std::vector<nodePair> deltaC = MarkAFF(Gamma, nodesOfS, e, nodesOfS);
					for (nodePair i : deltaC)
					{
						Pi.insert(i);
					}
				}
				std::vector<nodePair> deltaPi = CheckCand(Gamma, Pi, C);
				for (nodePair i : deltaPi)
				{
					Pi_plus.insert(i);
				}
				C.clear();
			}

			return {Pi_sub, Pi_plus};
		}

		std::unordered_set<nodePair> DecMatch(DepGraph<GRAPH> &Gamma, std::unordered_set<nodePair> &Pi,
											  nodePair u1_v1, std::unordered_set<nodePair> &S)
		{
			std::unordered_set<nodePair> deltaC, emptySet;
			if (checkPM(u1_v1.u(), u1_v1.v(), Pi, emptySet))
			{
				return deltaC;
			}
			deltaC.insert(u1_v1);
			std::queue<nodePair> delta_uvs;
			delta_uvs.push(u1_v1);
			Pi.erase(u1_v1);
			while (!delta_uvs.empty())
			{
				nodePair cur_uv = delta_uvs.front();
				vertex_t c_u = cur_uv.u(), c_v = cur_uv.v();
				delta_uvs.pop();
				for (nodePair p_uv : S)
				{
					vertex_t p_u = p_uv.u(), p_v = p_uv.v();
					if (precMap_gd[c_u].find(p_u) != precMap_gd[c_u].end() &&
						precMap_g[c_v].find(p_v) != precMap_g[c_v].end())
					{
						if (Pi.find(p_uv) != Pi.end() && !checkPM(p_u, p_v, Pi, emptySet))
						{
							deltaC.insert(p_uv);
							delta_uvs.push(p_uv);
							Pi.erase(p_uv);
						}
					}
				}
			}

			return deltaC;
		}

	private:
		GRAPH gd_;
		GRAPH g_;
		H_V h_v_;
		H_P h_p_;
		H_R h_r_;
		double sigma_{};
		double delta_{};
		int k_{};
		std::unordered_map<vertex_t, std::unordered_set<vertex_t>> precMap_gd;
		std::unordered_map<vertex_t, std::unordered_set<vertex_t>> precMap_g;
		std::unordered_map<vertex_t, std::unordered_set<vertex_t>> nextMap_gd;
		std::unordered_map<vertex_t, std::unordered_set<vertex_t>> nextMap_g;
		std::unordered_map<vertex_t, descendants_t> ecache_gd_;
		std::unordered_map<vertex_t, descendants_t> ecache_g_;
		std::unordered_map<nodePair, double> hv_cache;
		std::unordered_map<std::pair<nodePair, nodePair>, double> hp_cache;
	};
}
#endif