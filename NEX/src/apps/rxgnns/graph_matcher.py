from copy import deepcopy

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

        if self.operator == "==":
            return node_value == self.value
        elif self.operator == "!=":
            return node_value != self.value
        elif self.operator == ">":
            return node_value > self.value
        elif self.operator == ">=":
            return node_value >= self.value
        elif self.operator == "<":
            return node_value < self.value
        elif self.operator == "<=":
            return node_value <= self.value
        else:
            raise ValueError(f"Unsupported operator {self.operator}")


class WLPredicate(Predicate):
    def __init__(self, node_id, is_negated=False):
        self.node_id = node_id
        self.is_negated = is_negated
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
        node_gnn = node.attributes.get('GNN', 0)

        for other_id, other_color in node_colors.items():
            if other_id != data_id and other_color == node_color:
                other_node = data_graph.nodes[other_id]
                other_gnn = other_node.attributes.get('GNN', 0)

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
        model_desc = self.model_predicate.description() if self.model_predicate else "true"
        return f"Q[x̄,x₀]({precond_desc} → {model_desc})"


class VF2State:
    def __init__(self, query_graph, data_graph):
        self.query_graph = query_graph
        self.data_graph = data_graph

        self.mapping = {}

        self.reverse = {}

        self.candidates_query = set(query_graph.nodes.keys())
        self.candidates_data = set(data_graph.nodes.keys())

        self.matched_query = set()
        self.matched_data = set()

    def is_complete(self):
        return len(self.matched_query) == len(self.query_graph.nodes)

    def candidate_pairs(self):
        if self.matched_query:
            query_neighbors = set()
            for q_id in self.matched_query:
                query_neighbors.update(self.query_graph.get_out_edges(q_id))
                query_neighbors.update(self.query_graph.get_in_edges(q_id))

            query_candidates = query_neighbors - self.matched_query

            if query_candidates:
                for q_id in query_candidates:
                    for d_id in self.candidates_data - self.matched_data:
                        if self.is_feasible(q_id, d_id):
                            yield q_id, d_id
                return

        for q_id in self.candidates_query - self.matched_query:
            for d_id in self.candidates_data - self.matched_data:
                if self.is_feasible(q_id, d_id):
                    yield q_id, d_id

    def is_feasible(self, q_id, d_id):
        if self.query_graph.nodes[q_id].label != self.data_graph.nodes[d_id].label:
            return False

        if d_id in self.reverse:
            return False

        for q_in in self.query_graph.get_in_edges(q_id):
            if q_in in self.matched_query:
                d_in = self.mapping[q_in]
                if (d_in, d_id) not in self.data_graph.edges:
                    return False

        for q_out in self.query_graph.get_out_edges(q_id):
            if q_out in self.matched_query:
                d_out = self.mapping[q_out]
                if (d_id, d_out) not in self.data_graph.edges:
                    return False

        return True

    def add_pair(self, q_id, d_id):
        self.mapping[q_id] = d_id
        self.reverse[d_id] = q_id
        self.matched_query.add(q_id)
        self.matched_data.add(d_id)

    def remove_pair(self, q_id, d_id):
        del self.mapping[q_id]
        del self.reverse[d_id]
        self.matched_query.remove(q_id)
        self.matched_data.remove(d_id)

    def get_mapping(self):
        return self.mapping.copy()


class Matcher:
    def __init__(self, data_graph):
        self.data_graph = data_graph

    def find_homomorphic_mappings(self, pattern, max_mappings=1000, pivot_only=False):
        mappings = []
        query_graph = pattern.graph
        pivot_id = pattern.pivot_id

        if not query_graph.nodes:
            return mappings

        match_order = self._determine_match_order(query_graph, pivot_id)

        state = VF2State(query_graph, self.data_graph)

        def match(state, depth=0):
            if depth == len(match_order):
                if pivot_only and pivot_id is not None:
                    mappings.append({pivot_id: state.mapping[pivot_id]})
                else:
                    mappings.append(state.get_mapping())
                return

            if len(mappings) >= max_mappings:
                return

            q_id = match_order[depth]

            candidates = self._get_compatible_candidates(state, q_id)

            for d_id in candidates:
                state.add_pair(q_id, d_id)

                match(state, depth + 1)

                state.remove_pair(q_id, d_id)

        match(state)
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

    def _get_compatible_candidates(self, state, q_id):
        query_node = state.query_graph.nodes[q_id]
        candidates = []

        connected_to_matched = False
        matched_predecessors = []
        matched_successors = []

        for prev_q_id in state.matched_query:
            if (prev_q_id, q_id) in state.query_graph.edges:
                connected_to_matched = True
                matched_predecessors.append(prev_q_id)
            if (q_id, prev_q_id) in state.query_graph.edges:
                connected_to_matched = True
                matched_successors.append(prev_q_id)

        if connected_to_matched:
            possible_candidates = set(d_id for d_id in state.data_graph.nodes
                                      if d_id not in state.matched_data and
                                      state.data_graph.nodes[d_id].label == query_node.label)

            for prev_q_id in matched_predecessors:
                prev_d_id = state.mapping[prev_q_id]
                possible_candidates = {d_id for d_id in possible_candidates
                                       if (prev_d_id, d_id) in state.data_graph.edges}
                if not possible_candidates:
                    return []

            for succ_q_id in matched_successors:
                succ_d_id = state.mapping[succ_q_id]
                possible_candidates = {d_id for d_id in possible_candidates
                                       if (d_id, succ_d_id) in state.data_graph.edges}
                if not possible_candidates:
                    return []

            candidates = list(possible_candidates)
        else:
            for d_id, d_node in state.data_graph.nodes.items():
                if d_id not in state.matched_data and d_node.label == query_node.label:
                    candidates.append(d_id)

        return candidates

    def _sort_predicates_by_type(self, preconditions):
        predicate_priorities = {
            AttributePredicate: 1,
            AttributeComparisonPredicate: 2,
            WLPredicate: 3,
        }

        return sorted(preconditions,
                      key=lambda p: predicate_priorities.get(type(p), 999))

    def _expand_mapping_with_pruning(self, initial_mapping, pattern, sorted_preconditions):
        mapping = initial_mapping.copy()

        unmapped_nodes = [n for n in pattern.graph.nodes if n not in mapping]
        if not unmapped_nodes:
            for pred in sorted_preconditions:
                if not pred.evaluate(mapping, self.data_graph, pattern.graph):
                    return None
            return mapping

        def expand(current_mapping, nodes_to_map):
            if not nodes_to_map:
                for pred in sorted_preconditions:
                    if not pred.evaluate(current_mapping, self.data_graph, pattern.graph):
                        return None
                return current_mapping

            node = nodes_to_map[0]
            remaining_nodes = nodes_to_map[1:]

            candidates = []
            for d_id, d_node in self.data_graph.nodes.items():
                if d_id in current_mapping.values():
                    continue
                if d_node.label != pattern.graph.nodes[node].label:
                    continue

                valid = True
                for mapped_q in current_mapping:
                    mapped_d = current_mapping[mapped_q]
                    if (mapped_q, node) in pattern.graph.edges:
                        if (mapped_d, d_id) not in self.data_graph.edges:
                            valid = False
                            break
                    if (node, mapped_q) in pattern.graph.edges:
                        if (d_id, mapped_d) not in self.data_graph.edges:
                            valid = False
                            break
                if valid:
                    candidates.append(d_id)

            for d_id in candidates:
                current_mapping[node] = d_id
                result = expand(current_mapping.copy(), remaining_nodes)
                if result:
                    return result
                del current_mapping[node]

            return None

        return expand(mapping, unmapped_nodes)

    def evaluate_rule(self, rxgnns, verbose=False):
        pattern = rxgnns.pattern
        preconditions = rxgnns.preconditions
        pivot_id = pattern.pivot_id

        sorted_preconditions = self._sort_predicates_by_type(preconditions)
        pivot_mappings = self.find_homomorphic_mappings(pattern, pivot_only=True)

        satisfies_preconditions = []
        for pivot_mapping in pivot_mappings:
            full_mapping = self._expand_mapping_with_pruning(pivot_mapping, pattern, sorted_preconditions)
            if full_mapping:
                satisfies_preconditions.append(full_mapping)

        satisfies_gnn = []
        for mapping in satisfies_preconditions:
            pivot_data_id = mapping[pivot_id]
            pivot_node = self.data_graph.nodes[pivot_data_id]
            if pivot_node.attributes.get('gnn_prediction', False):
                satisfies_gnn.append(mapping)

        support = len(satisfies_gnn)
        confidence = (support / len(satisfies_preconditions)) if satisfies_preconditions else 0

        result = {
            'support': support,
            'confidence': confidence,
            'support_count': support,
            'confidence_count': (len(satisfies_preconditions), support),
            'satisfies_preconditions': satisfies_preconditions,
            'satisfies_gnn': satisfies_gnn,
            'total_mappings': len(pivot_mappings)
        }

        return result

    def get_pivot_matches(self, rxgnns):
        pivot_id = rxgnns.pattern.pivot_id
        if not pivot_id:
            return []

        result = self.evaluate_rule(rxgnns)

        mappings = result.get('satisfies_all', result.get('satisfies_preconditions', []))

        return [(m[pivot_id], m) for m in mappings if pivot_id in m]

    def get_pattern_support(self, pattern):
        mappings = self.find_homomorphic_mappings(pattern, pivot_only=True)
        return len(mappings)

    def get_pattern_confidence(self, pattern):
        mappings = self.find_homomorphic_mappings(pattern, pivot_only=True)
        if not mappings:
            return 0
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

