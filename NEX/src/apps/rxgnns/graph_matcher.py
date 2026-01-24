from copy import deepcopy
import random
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional

class Node:
    def __init__(self, id, label, attributes=None):
        self.id = id
        self.label = label
        self.attributes = attributes or {}

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = set()
        self._in_edges = None
        self._out_edges = None

    def add_node(self, node):
        self.nodes[node.id] = node
        self._in_edges = None
        self._out_edges = None

    def add_edge(self, source_id, target_id):
        if source_id in self.nodes and target_id in self.nodes:
            self.edges.add((source_id, target_id))
            self._in_edges = None
            self._out_edges = None
        else:
            raise ValueError(f"Node {source_id} or {target_id} not in graph")

    def get_in_edges(self, node_id):
        if self._in_edges is None:
            self._in_edges = {}
            for src, tgt in self.edges:
                if tgt not in self._in_edges:
                    self._in_edges[tgt] = set()
                self._in_edges[tgt].add(src)
        return self._in_edges.get(node_id, set())

    def get_out_edges(self, node_id):
        if self._out_edges is None:
            self._out_edges = {}
            for src, tgt in self.edges:
                if src not in self._out_edges:
                    self._out_edges[src] = set()
                self._out_edges[src].add(tgt)
        return self._out_edges.get(node_id, set())

class Pattern:
    def __init__(self, pivot_id=None):
        self.graph = Graph()
        self.pivot_id = pivot_id

    def add_node(self, node):
        self.graph.add_node(node)
        if self.pivot_id is None:
            self.pivot_id = node.id
        return self

    def add_edge(self, source_id, target_id):
        self.graph.add_edge(source_id, target_id)
        return self

    def set_pivot(self, pivot_id):
        if pivot_id in self.graph.nodes:
            self.pivot_id = pivot_id
        else:
            raise ValueError(f"Pivot node {pivot_id} not in pattern graph")
        return self

class Predicate:
    def evaluate(self, mapping, data_graph, query_pattern):
        raise NotImplementedError("Subclasses must implement evaluate()")

    def description(self):
        raise NotImplementedError("Subclasses must implement description()")

    def get_involved_nodes(self):
        raise NotImplementedError("Subclasses must implement get_involved_nodes()")

class PointWisePredicate(Predicate):
    def __init__(self, node_id, evaluate_func, description_str=None):
        self.node_id = node_id
        self.evaluate_func = evaluate_func
        self._description = description_str

    def evaluate(self, mapping, data_graph, query_pattern):
        if self.node_id not in mapping:
            return False
        data_id = mapping[self.node_id]
        if data_id not in data_graph.nodes:
            return False
        data_node = data_graph.nodes[data_id]
        return self.evaluate_func(data_node)

    def description(self):
        return self._description or f"Point condition on node {self.node_id}"

    def get_involved_nodes(self):
        return {self.node_id}

class AttributePredicate(PointWisePredicate):
    def __init__(self, node_id, attribute, value, operator="=="):
        self.node_id = node_id
        self.attribute = attribute
        self.value = value
        self.operator = operator
        description_str = f"{node_id}.{attribute} {operator} {value}"
        super().__init__(node_id, None, description_str)

    def evaluate(self, mapping, data_graph, query_pattern):
        if self.node_id not in mapping:
            return False
        data_id = mapping[self.node_id]
        if data_id not in data_graph.nodes:
            return False
        data_node = data_graph.nodes[data_id]
        if self.attribute not in data_node.attributes or data_node.attributes[self.attribute] is None:
            return False
        node_value = data_node.attributes[self.attribute]
        try:
            if self.operator == "==":
                result = node_value == self.value
            elif self.operator == "!=":
                result = node_value != self.value
            elif self.operator == ">":
                result = node_value > self.value
            elif self.operator == ">=":
                result = node_value >= self.value
            elif self.operator == "<":
                result = node_value < self.value
            elif self.operator == "<=":
                result = node_value <= self.value
            else:
                raise ValueError(f"Unsupported operator {self.operator}")
            return result
        except TypeError:
            return False

class WLPredicate(Predicate):
    def __init__(self, node_id, is_negated=False, gnn_attr='gnn_prediction'):
        self.node_id = node_id
        self.is_negated = is_negated
        self.gnn_attr = gnn_attr
        self.color_map = {}

    def evaluate(self, mapping, data_graph, query_pattern):
        if self.node_id not in mapping:
            return False
        data_id = mapping[self.node_id]
        if data_id not in data_graph.nodes:
            return False
        subgraph_nodes = self.extract_local_neighborhood(data_graph, data_id, 3)
        local_graph = self.create_local_subgraph(data_graph, subgraph_nodes)
        node_colors = self.compute_local_wl_colors(local_graph)
        node_color = node_colors.get(data_id)
        node = data_graph.nodes[data_id]
        node_gnn = node.attributes.get(self.gnn_attr, False)
        for other_id, other_color in node_colors.items():
            if other_id != data_id and other_color == node_color:
                other_node = data_graph.nodes[other_id]
                other_gnn = other_node.attributes.get(self.gnn_attr, False)
                if node_gnn != other_gnn:
                    return not self.is_negated
        return self.is_negated

    def extract_local_neighborhood(self, graph, start_node, max_hops=3):
        result = {start_node}
        frontier = {start_node}
        for _ in range(max_hops):
            next_frontier = set()
            for node in frontier:
                in_edges = graph.get_in_edges(node)
                out_edges = graph.get_out_edges(node)
                next_frontier.update(in_edges)
                next_frontier.update(out_edges)
            frontier = next_frontier - result
            result.update(frontier)
            if not frontier:
                break
        return result

    def create_local_subgraph(self, graph, nodes):
        local_graph = {}
        for node_id in nodes:
            if node_id in graph.nodes:
                local_graph[node_id] = {
                    'label': graph.nodes[node_id].label,
                    'in_edges': list(graph.get_in_edges(node_id) & nodes),
                    'out_edges': list(graph.get_out_edges(node_id) & nodes)
                }
        return local_graph

    def compute_local_wl_colors(self, local_graph):
        colors = {node_id: str(data['label']) for node_id, data in local_graph.items()}
        for _ in range(1):
            new_colors = {}
            for node_id, data in local_graph.items():
                in_colors = sorted([colors.get(n, '') for n in data['in_edges']])
                out_colors = sorted([colors.get(n, '') for n in data['out_edges']])
                neighbor_colors = ''.join(in_colors) + '|' + ''.join(out_colors)
                new_colors[node_id] = colors[node_id] + neighbor_colors
            if all(new_colors[n] == colors[n] for n in colors):
                break
            colors = new_colors
        color_map = {}
        unique_colors = {}
        for node_id, color in colors.items():
            if color not in unique_colors:
                unique_colors[color] = len(unique_colors)
            color_map[node_id] = unique_colors[color]
        return color_map

    def description(self):
        return f"{'¬' if self.is_negated else ''}1WL({self.node_id})"

    def get_involved_nodes(self):
        return {self.node_id}

class PairWisePredicate(Predicate):
    def __init__(self, node1_id, node2_id, compare_func=None, description_str=None):
        self.node1_id = node1_id
        self.node2_id = node2_id
        self.compare_func = compare_func
        self._description = description_str

    def evaluate(self, mapping, data_graph, query_pattern):
        if self.compare_func is None:
            raise NotImplementedError("Subclass must implement evaluate() or provide compare_func")
        if self.node1_id not in mapping or self.node2_id not in mapping:
            return False
        data_id1 = mapping[self.node1_id]
        data_id2 = mapping[self.node2_id]
        if data_id1 not in data_graph.nodes or data_id2 not in data_graph.nodes:
            return False
        data_node1 = data_graph.nodes[data_id1]
        data_node2 = data_graph.nodes[data_id2]
        return self.compare_func(data_node1, data_node2)

    def description(self):
        return self._description or f"Relation between {self.node1_id} and {self.node2_id}"

    def get_involved_nodes(self):
        return {self.node1_id, self.node2_id}

class AttributeComparisonPredicate(PairWisePredicate):
    def __init__(self, node1_id, attr1, node2_id, attr2, operator="=="):
        self.node1_id = node1_id
        self.attr1 = attr1
        self.node2_id = node2_id
        self.attr2 = attr2
        self.operator = operator
        description_str = f"{node1_id}.{attr1} {operator} {node2_id}.{attr2}"
        super().__init__(node1_id, node2_id, None, description_str)

    def evaluate(self, mapping, data_graph, query_pattern):
        if self.node1_id not in mapping or self.node2_id not in mapping:
            return False
        data_id1 = mapping[self.node1_id]
        data_id2 = mapping[self.node2_id]
        if data_id1 not in data_graph.nodes or data_id2 not in data_graph.nodes:
            return False
        data_node1 = data_graph.nodes[data_id1]
        data_node2 = data_graph.nodes[data_id2]
        if self.attr1 not in data_node1.attributes or self.attr2 not in data_node2.attributes:
            return False
        value1 = data_node1.attributes[self.attr1]
        value2 = data_node2.attributes[self.attr2]
        try:
            if self.operator == "==":
                return value1 == value2
            elif self.operator == "!=":
                return value1 != value2
            elif self.operator == ">":
                return value1 > value2
            elif self.operator == ">=":
                return value1 >= value2
            elif self.operator == "<":
                return value1 < value2
            elif self.operator == "<=":
                return value1 <= value2
            else:
                raise ValueError(f"Unsupported operator {self.operator}")
        except TypeError:
            return False

class RxGNNs:
    def __init__(self, pattern, model_predicate=None):
        self.pattern = pattern
        self.model_predicate = model_predicate
        self.preconditions = []

    def add_precondition(self, predicate):
        self.preconditions.append(predicate)
        return self

    def add_preconditions(self, predicates):
        self.preconditions.extend(predicates)
        return self

    def description(self):
        precond_desc = " ∧ ".join(p.description() for p in self.preconditions)
        model_desc = self.model_predicate.description() if self.model_predicate else "¬M(x₀)"
        return f"Q[x̄,x₀]({precond_desc} → {model_desc})"

class Matcher:
    def __init__(self, data_graph, debug=False):
        self.data_graph = data_graph
        self.debug = debug

    def get_all_matching_pivots(self, pattern, pivot_id=None, predicates=None) -> Set[str]:
        if pivot_id is None:
            pivot_id = pattern.pivot_id
        if not pivot_id:
            raise ValueError("Pattern must have a pivot node")
        pivot_label = pattern.graph.nodes[pivot_id].label
        candidates = [
            nid for nid, node in self.data_graph.nodes.items()
            if node.label == pivot_label
        ]
        if self.debug:
            pass
        matching_pivots = set()
        for candidate_id in candidates:
            if self._has_pattern_match(pattern, pivot_id, candidate_id):
                matching_pivots.add(candidate_id)
        if self.debug:
            pass
        if predicates:
            matching_pivots = self._apply_predicate_filter(
                matching_pivots, pattern, pivot_id, predicates
            )
            if self.debug:
                pass
        return matching_pivots

    def _has_pattern_match(self, pattern, pivot_id, candidate_pivot_id) -> bool:
        initial_mapping = {pivot_id: candidate_pivot_id}
        try:
            mapping = self._find_one_mapping(pattern, initial_mapping)
            return mapping is not None
        except:
            return False

    def _find_one_mapping(self, pattern, initial_mapping) -> Optional[Dict[str, str]]:
        query_graph = pattern.graph
        fixed_nodes = set(initial_mapping.keys())
        remaining_nodes = [n for n in query_graph.nodes.keys() if n not in fixed_nodes]
        def dfs_match(current_mapping, remaining):
            if not remaining:
                for src, tgt in query_graph.edges:
                    data_src = current_mapping[src]
                    data_tgt = current_mapping[tgt]
                    if (data_src, data_tgt) not in self.data_graph.edges:
                        return None
                return current_mapping
            q_node = remaining[0]
            q_label = query_graph.nodes[q_node].label
            candidates = self._get_candidates_for_node(
                q_node, q_label, current_mapping, query_graph
            )
            for cand in candidates:
                if self._is_compatible(q_node, cand, current_mapping, query_graph):
                    new_mapping = current_mapping.copy()
                    new_mapping[q_node] = cand
                    result = dfs_match(new_mapping, remaining[1:])
                    if result is not None:
                        return result
            return None
        return dfs_match(initial_mapping, remaining_nodes)

    def _get_candidates_for_node(self, q_node, q_label, current_mapping, query_graph) -> List[str]:
        candidates = []
        for matched_q, matched_d in current_mapping.items():
            if (matched_q, q_node) in query_graph.edges:
                for edge in self.data_graph.edges:
                    if edge[0] == matched_d:
                        cand = edge[1]
                        if cand not in current_mapping.values():
                            if self.data_graph.nodes[cand].label == q_label:
                                candidates.append(cand)
            elif (q_node, matched_q) in query_graph.edges:
                for edge in self.data_graph.edges:
                    if edge[1] == matched_d:
                        cand = edge[0]
                        if cand not in current_mapping.values():
                            if self.data_graph.nodes[cand].label == q_label:
                                candidates.append(cand)
        candidates = list(set(candidates))
        if not candidates:
            for nid, node in self.data_graph.nodes.items():
                if node.label == q_label and nid not in current_mapping.values():
                    candidates.append(nid)
            if len(candidates) > 100:
                candidates = candidates[:100]
        return candidates

    def _is_compatible(self, q_node, d_node, current_mapping, query_graph) -> bool:
        if self.data_graph.nodes[d_node].label != query_graph.nodes[q_node].label:
            return False
        if d_node in current_mapping.values():
            return False
        for matched_q, matched_d in current_mapping.items():
            if (matched_q, q_node) in query_graph.edges:
                if (matched_d, d_node) not in self.data_graph.edges:
                    return False
            if (q_node, matched_q) in query_graph.edges:
                if (d_node, matched_d) not in self.data_graph.edges:
                    return False
        return True

    def _apply_predicate_filter(self, pivots, pattern, pivot_id, predicates) -> Set[str]:
        filtered = set()
        for pid in pivots:
            passes = True
            for pred in predicates:
                involved_nodes = pred.get_involved_nodes()
                if len(involved_nodes) == 1 and pivot_id in involved_nodes:
                    mapping = {pivot_id: pid}
                    try:
                        if not pred.evaluate(mapping, self.data_graph, pattern.graph):
                            passes = False
                            break
                    except:
                        passes = False
                        break
            if passes:
                filtered.add(pid)
        return filtered

    def evaluate_rule(self, rxgnns, verbose=False, gnn_attr='gnn_prediction'):
        pattern = rxgnns.pattern
        preconditions = rxgnns.preconditions
        pivot_id = pattern.pivot_id
        if not pivot_id:
            raise ValueError("Pattern must have a pivot node")
        if verbose or self.debug:
            pass
        matching_pivots = self.get_all_matching_pivots(
            pattern,
            pivot_id,
            preconditions
        )
        support = len(matching_pivots)
        if verbose or self.debug:
            pass
        gnn_false_count = 0
        gnn_true_count = 0
        satisfies_gnn_false_mappings = []
        for pid in matching_pivots:
            node = self.data_graph.nodes[pid]
            if gnn_attr in node.attributes:
                gnn_pred = node.attributes[gnn_attr]
                is_false = self._is_gnn_false(gnn_pred)
                if is_false:
                    gnn_false_count += 1
                    satisfies_gnn_false_mappings.append({pivot_id: pid})
                else:
                    gnn_true_count += 1
        confidence = (gnn_false_count / support) if support > 0 else 0
        if verbose or self.debug:
            pass
        result = {
            'support': support,
            'confidence': confidence,
            'support_count': support,
            'confidence_count': (support, gnn_false_count),
            'satisfies_preconditions': [{pivot_id: pid} for pid in matching_pivots],
            'satisfies_gnn_false': satisfies_gnn_false_mappings,
            'total_mappings': len(matching_pivots),
            'unique_pivots_preconditions': support,
            'unique_pivots_gnn_false': gnn_false_count
        }
        return result

    def _is_gnn_false(self, gnn_pred):
        if isinstance(gnn_pred, bool):
            return gnn_pred == False
        elif isinstance(gnn_pred, (int, float)):
            return gnn_pred == 0
        elif isinstance(gnn_pred, str):
            return gnn_pred.lower() in ['false', '0', 'no']
        return False

    def find_homomorphic_mappings(self, pattern, max_mappings=None,
                                  pivot_only=False, predicates=None):
        if max_mappings is None:
            max_mappings = float('inf')
        mappings = []
        query_graph = pattern.graph
        pivot_id = pattern.pivot_id
        if not query_graph.nodes:
            return mappings
        match_order = self._determine_match_order(query_graph, pivot_id)
        def backtrack(current_mapping, remaining_nodes):
            if len(mappings) >= max_mappings:
                return
            if not remaining_nodes:
                if predicates:
                    if all(pred.evaluate(current_mapping, self.data_graph, query_graph)
                           for pred in predicates):
                        if pivot_only and pivot_id:
                            mappings.append({pivot_id: current_mapping[pivot_id]})
                        else:
                            mappings.append(current_mapping.copy())
                else:
                    if pivot_only and pivot_id:
                        mappings.append({pivot_id: current_mapping[pivot_id]})
                    else:
                        mappings.append(current_mapping.copy())
                return
            q_node = remaining_nodes[0]
            q_label = query_graph.nodes[q_node].label
            candidates = self._get_candidates_for_node(
                q_node, q_label, current_mapping, query_graph
            )
            for cand in candidates:
                if self._is_compatible(q_node, cand, current_mapping, query_graph):
                    current_mapping[q_node] = cand
                    backtrack(current_mapping, remaining_nodes[1:])
                    del current_mapping[q_node]
                    if len(mappings) >= max_mappings:
                        return
        backtrack({}, match_order)
        return mappings

    def _determine_match_order(self, query_graph, pivot_id=None):
        node_degrees = {}
        for node_id in query_graph.nodes:
            in_edges = query_graph.get_in_edges(node_id)
            out_edges = query_graph.get_out_edges(node_id)
            node_degrees[node_id] = len(in_edges) + len(out_edges)
        order = []
        if pivot_id is not None and pivot_id in query_graph.nodes:
            order.append(pivot_id)
        remaining = [n for n in query_graph.nodes if n != pivot_id]
        remaining.sort(key=lambda n: node_degrees[n], reverse=True)
        order.extend(remaining)
        return order

    def get_pivot_matches(self, rxgnns):
        pivot_id = rxgnns.pattern.pivot_id
        if not pivot_id:
            return []
        result = self.evaluate_rule(rxgnns)
        mappings = result.get('satisfies_gnn_false', result.get('satisfies_preconditions', []))
        return [(m[pivot_id], m) for m in mappings if pivot_id in m]

    def get_pattern_support(self, pattern):
        matching_pivots = self.get_all_matching_pivots(pattern, pattern.pivot_id, predicates=None)
        return len(matching_pivots)

    def get_pattern_confidence(self, pattern):
        matching_pivots = self.get_all_matching_pivots(pattern, pattern.pivot_id, predicates=None)
        if not matching_pivots:
            return 0.0
        return 1.0

    def merge_patterns(self, pattern1, pattern2, nodes_to_merge):
        result_pattern = Pattern(pattern1.pivot_id)
        for node_id, node in pattern1.graph.nodes.items():
            result_pattern.graph.add_node(deepcopy(node))
        for edge in pattern1.graph.edges:
            result_pattern.graph.add_edge(edge[0], edge[1])
        node_mapping = {}
        for (node1, node2) in nodes_to_merge:
            node_mapping[node2] = node1
        for node_id, node in pattern2.graph.nodes.items():
            if node_id not in node_mapping:
                new_id = f"{node_id}_p2"
                node_mapping[node_id] = new_id
                new_node = deepcopy(node)
                new_node.id = new_id
                result_pattern.graph.add_node(new_node)
        for edge in pattern2.graph.edges:
            src = node_mapping.get(edge[0], edge[0])
            tgt = node_mapping.get(edge[1], edge[1])
            if (src, tgt) not in result_pattern.graph.edges:
                result_pattern.graph.add_edge(src, tgt)
        return result_pattern