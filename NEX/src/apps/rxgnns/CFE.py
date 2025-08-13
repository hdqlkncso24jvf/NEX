import copy
from collections import defaultdict, deque
import random
import heapq
import numpy as np
from sklearn.cluster import KMeans
from graph_matcher import Matcher
from graph_matcher import Graph, Node, AttributeComparisonPredicate, Pattern, WLPredicate, AttributePredicate, RxGNNs

class CounterfactualExplainer:

    def __init__(self, threshold=5, alpha=0.5, lambda_factor=1000.0, max_iterations=3, num_clusters=5):
        self.threshold = threshold
        self.alpha = alpha
        self.lambda_factor = lambda_factor
        self.max_iterations = max_iterations
        self.num_clusters = num_clusters

    def explain(self, subgraph, rxgnn_rule):
        pattern = rxgnn_rule.pattern
        pivot_id = pattern.pivot_id
        preconditions = rxgnn_rule.preconditions

        attr_to_modify = {}
        edge_removals = []

        dag = self._build_dag(pattern.graph, pivot_id)
        reverse_dag = self._reverse_dag(dag)

        cand_space = self._initialize_candidate_space(subgraph, pattern.graph, preconditions, pivot_id)

        refined_cand_space = self._refine_candidate_space(
            cand_space, pattern.graph, dag, reverse_dag, preconditions
        )

        if not refined_cand_space.get(pivot_id, []):
            return attr_to_modify, edge_removals

        pivot_node = None
        for mapping in self._find_homomorphic_mappings(pattern.graph, subgraph, refined_cand_space):
            if pivot_id in mapping:
                pivot_node = mapping[pivot_id]
                break

        if pivot_node is None:
            return attr_to_modify, edge_removals

        iterations = 0
        max_perturbations = 10

        while (pivot_id in refined_cand_space and
               pivot_node in refined_cand_space[pivot_id] and
               iterations < max_perturbations):

            best_perturbation = self._select_best_perturbation_two_phase(
                subgraph, pattern.graph, refined_cand_space,
                dag, reverse_dag, preconditions, pivot_id
            )

            if not best_perturbation:
                break

            perturbation_type, perturbation_info = best_perturbation

            if perturbation_type == 'attr':
                node_id, attr_name = perturbation_info
                if node_id not in attr_to_modify:
                    attr_to_modify[node_id] = []
                if attr_name not in attr_to_modify[node_id]:
                    attr_to_modify[node_id].append(attr_name)

                if node_id in refined_cand_space[pivot_id]:
                    refined_cand_space[pivot_id].remove(node_id)

            elif perturbation_type == 'edge':
                source_id, target_id = perturbation_info
                edge_removals.append((source_id, target_id))

                if (source_id, target_id) in subgraph.edges:
                    subgraph.edges.remove((source_id, target_id))

            refined_cand_space = self._initialize_candidate_space(subgraph, pattern.graph, preconditions, pivot_id)
            refined_cand_space = self._refine_candidate_space(
                refined_cand_space, pattern.graph, dag, reverse_dag, preconditions
            )

            iterations += 1

        return attr_to_modify, edge_removals

    def _build_dag(self, graph, root_id):
        dag = {node_id: [] for node_id in graph.nodes}
        visited = set([root_id])
        queue = [root_id]

        while queue:
            node_id = queue.pop(0)

            neighbors = []
            for edge in graph.edges:
                if edge[0] == node_id and edge[1] not in visited:
                    neighbors.append(edge[1])
                elif edge[1] == node_id and edge[0] not in visited:
                    neighbors.append(edge[0])

            label_freq = defaultdict(int)
            for n_id in graph.nodes:
                label = graph.nodes[n_id].label
                label_freq[label] += 1

            neighbors.sort(key=lambda x: (label_freq[graph.nodes[x].label], -len(self._get_edges(graph, x))))

            for neighbor in neighbors:
                dag[node_id].append(neighbor)
                visited.add(neighbor)
                queue.append(neighbor)

        return dag

    def _reverse_dag(self, dag):
        reverse_dag = {node: [] for node in dag}

        for node, children in dag.items():
            for child in children:
                reverse_dag[child].append(node)

        return reverse_dag

    def _initialize_candidate_space(self, data_graph, pattern_graph, preconditions, pivot_id):
        cand_space = {}

        attr_predicates = {}
        for pred in preconditions:
            if isinstance(pred, AttributePredicate):
                if pred.node_id not in attr_predicates:
                    attr_predicates[pred.node_id] = []
                attr_predicates[pred.node_id].append(pred)

        for pattern_node_id, pattern_node in pattern_graph.nodes.items():
            cand_space[pattern_node_id] = []

            for data_node_id, data_node in data_graph.nodes.items():
                if data_node.label != pattern_node.label:
                    continue

                if len(self._get_edges(data_graph, data_node_id)) < len(
                        self._get_edges(pattern_graph, pattern_node_id)):
                    continue

                valid = True
                if pattern_node_id in attr_predicates:
                    for pred in attr_predicates[pattern_node_id]:
                        mapping = {pattern_node_id: data_node_id}
                        if not pred.evaluate(mapping, data_graph, pattern_graph):
                            valid = False
                            break

                for pred in preconditions:
                    if isinstance(pred, AttributePredicate) and pred.node_id == pattern_node_id:
                        mapping = {pattern_node_id: data_node_id}
                        if not pred.evaluate(mapping, data_graph, pattern_graph):
                            valid = False
                            break

                    elif isinstance(pred, WLPredicate) and pred.node_id == pattern_node_id:
                        mapping = {pattern_node_id: data_node_id}
                        if not pred.evaluate(mapping, data_graph, pattern_graph):
                            valid = False
                            break

                if valid:
                    cand_space[pattern_node_id].append(data_node_id)

        return cand_space

    def _refine_candidate_space(self, cand_space, pattern_graph, dag, reverse_dag, preconditions):
        refined_cand = copy.deepcopy(cand_space)

        old_size = sum(len(candidates) for candidates in refined_cand.values())

        for _ in range(self.max_iterations):
            refined_cand = self._dag_graph_dp(refined_cand, pattern_graph, dag, preconditions)

            if any(len(candidates) == 0 for candidates in refined_cand.values()):
                return refined_cand

            refined_cand = self._dag_graph_dp(refined_cand, pattern_graph, reverse_dag, preconditions)

            if any(len(candidates) == 0 for candidates in refined_cand.values()):
                return refined_cand

            new_size = sum(len(candidates) for candidates in refined_cand.values())
            if new_size == old_size:
                break

            old_size = new_size

        return refined_cand

    def _dag_graph_dp(self, cand_space, pattern_graph, dag, preconditions):
        refined_cand = copy.deepcopy(cand_space)

        topo_order = self._topological_sort(dag)

        for node_id in reversed(topo_order):
            valid_candidates = []

            for candidate_id in refined_cand[node_id]:
                valid = True

                for child_id in dag[node_id]:
                    valid_child_candidates = []

                    for child_candidate_id in refined_cand[child_id]:
                        edge_valid = self._check_edge_constraint(
                            pattern_graph, node_id, candidate_id, child_id, child_candidate_id
                        )

                        if not edge_valid:
                            continue

                        var_valid = True
                        for pred in preconditions:
                            if isinstance(pred, AttributeComparisonPredicate) and \
                                    ((pred.node1_id == node_id and pred.node2_id == child_id) or
                                     (pred.node1_id == child_id and pred.node2_id == node_id)):

                                mapping = {
                                    node_id: candidate_id,
                                    child_id: child_candidate_id
                                }

                                if not pred.evaluate(mapping, None, pattern_graph):
                                    var_valid = False
                                    break

                        if var_valid:
                            valid_child_candidates.append(child_candidate_id)

                    if not valid_child_candidates:
                        valid = False
                        break

                    refined_cand[child_id] = valid_child_candidates

                if valid:
                    valid_candidates.append(candidate_id)

            refined_cand[node_id] = valid_candidates

        return refined_cand

    def _encode_neighborhood(self, data_graph, node_id, max_hops=3):
        neighbors = set()
        current = {node_id}

        for hop in range(max_hops):
            next_hop = set()
            for n in current:
                for edge in data_graph.edges:
                    if edge[0] == n and edge[1] not in neighbors and edge[1] != node_id:
                        next_hop.add(edge[1])
                    elif edge[1] == n and edge[0] not in neighbors and edge[0] != node_id:
                        next_hop.add(edge[0])

            neighbors.update(next_hop)
            current = next_hop
            if not current:
                break

        degree = len(self._get_edges(data_graph, node_id))
        label_counts = defaultdict(int)

        for n in neighbors:
            if n in data_graph.nodes:
                label_counts[data_graph.nodes[n].label] += 1

        all_labels = sorted(set(data_graph.nodes[n].label for n in data_graph.nodes))
        label_vector = [label_counts.get(label, 0) for label in all_labels]

        features = [degree] + label_vector

        return np.array(features)

    def _cluster_candidates(self, data_graph, pattern_node_id, candidates):
        if len(candidates) <= 1:
            return {0: candidates}

        features = []
        for node_id in candidates:
            feature_vector = self._encode_neighborhood(data_graph, node_id)
            features.append(feature_vector)

        if not features:
            return {0: []}

        max_len = max(len(f) for f in features)
        padded_features = []
        for f in features:
            if len(f) < max_len:
                padded = np.pad(f, (0, max_len - len(f)), 'constant')
                padded_features.append(padded)
            else:
                padded_features.append(f)

        k = min(self.num_clusters, len(candidates))
        if k <= 1:
            return {0: candidates}

        try:
            features_array = np.array(padded_features)
            kmeans = KMeans(n_clusters=k, random_state=0).fit(features_array)
            labels = kmeans.labels_

            clusters = defaultdict(list)
            for i, node_id in enumerate(candidates):
                clusters[labels[i]].append(node_id)

            return dict(clusters)
        except Exception as e:
            return {0: candidates}

    def _select_best_perturbation_two_phase(self, data_graph, pattern_graph, cand_space, dag, reverse_dag,
                                            preconditions, pivot_id):
        clusters_by_pattern_node = {}
        for pattern_node_id, candidates in cand_space.items():
            if not candidates:
                continue

            clusters_by_pattern_node[pattern_node_id] = self._cluster_candidates(
                data_graph, pattern_node_id, candidates
            )

        representative_perturbations = []

        for pattern_node_id, clusters in clusters_by_pattern_node.items():
            relevant_predicates = []
            for pred in preconditions:
                if isinstance(pred, AttributePredicate) and pred.node_id == pattern_node_id:
                    relevant_predicates.append(pred)

            if not relevant_predicates:
                continue

            for cluster_id, candidates in clusters.items():
                if not candidates:
                    continue

                representative = candidates[0]

                for pred in relevant_predicates:
                    attr_name = pred.attribute

                    cost = self._estimate_attr_modification_cost(data_graph, pattern_node_id, representative, attr_name)
                    effectiveness = self._estimate_representative_effectiveness(
                        pattern_node_id, cluster_id, len(candidates), pattern_graph, pivot_id
                    )

                    if cost > 0:
                        representative_perturbations.append((
                            effectiveness / cost,
                            ('attr', pattern_node_id, representative, attr_name, cluster_id)
                        ))

        edge_perturbations = []
        for pattern_edge in pattern_graph.edges:
            source_pattern_id, target_pattern_id = pattern_edge[0], pattern_edge[1]

            if source_pattern_id in cand_space and target_pattern_id in cand_space:
                for source_candidate_id in cand_space[source_pattern_id]:
                    for target_candidate_id in cand_space[target_pattern_id]:
                        if (source_candidate_id, target_candidate_id) in data_graph.edges:
                            cost = self.lambda_factor
                            effectiveness = self._estimate_perturbation_effectiveness(
                                source_pattern_id, source_candidate_id, cand_space, pattern_graph, pivot_id
                            )

                            edge_perturbations.append((
                                effectiveness / cost,
                                ('edge', (source_candidate_id, target_candidate_id))
                            ))

        if representative_perturbations:
            representative_perturbations.sort(reverse=True)
            _, (pert_type, pattern_node_id, _, _, cluster_id) = representative_perturbations[0]

            if pert_type == 'attr':
                best_cluster_candidates = clusters_by_pattern_node[pattern_node_id][cluster_id]

                detailed_perturbations = []
                for candidate_id in best_cluster_candidates:
                    original_cand_space = copy.deepcopy(cand_space)

                    test_cand_space = copy.deepcopy(cand_space)
                    if candidate_id in test_cand_space.get(pattern_node_id, []):
                        test_cand_space[pattern_node_id].remove(candidate_id)

                    new_cand_space = self._refine_candidate_space(
                        test_cand_space, pattern_graph, dag, reverse_dag, preconditions
                    )

                    effect_score = self._calculate_effect_score(original_cand_space, new_cand_space)

                    for pred in preconditions:
                        if isinstance(pred, AttributePredicate) and pred.node_id == pattern_node_id:
                            attr_name = pred.attribute
                            cost = self._estimate_attr_modification_cost(data_graph, pattern_node_id, candidate_id,
                                                                         attr_name)

                            if cost > 0:
                                detailed_perturbations.append((
                                    effect_score / cost,
                                    ('attr', (candidate_id, attr_name))
                                ))

                if detailed_perturbations:
                    detailed_perturbations.sort(reverse=True)
                    return detailed_perturbations[0][1]

        if edge_perturbations:
            edge_perturbations.sort(reverse=True)
            return edge_perturbations[0][1]

        return None

    def _calculate_effect_score(self, original_cand_space, new_cand_space):
        if not original_cand_space or not new_cand_space:
            return 0.0

        reduction_ratios = []
        for node_id in original_cand_space:
            orig_size = len(original_cand_space.get(node_id, []))
            new_size = len(new_cand_space.get(node_id, []))

            if orig_size > 0:
                reduction_ratio = 1.0 - (new_size / orig_size)
                reduction_ratios.append(reduction_ratio)

        return max(reduction_ratios) if reduction_ratios else 0.0

    def _estimate_representative_effectiveness(self, pattern_node_id, cluster_id, cluster_size, pattern_graph,
                                               pivot_id):
        distance = self._compute_distance(pattern_graph, pattern_node_id, pivot_id)
        distance_factor = self.alpha ** distance

        cluster_importance = min(1.0, cluster_size / 10.0)

        position_importance = 3.0 if pattern_node_id == pivot_id else 2.0 if self._are_neighbors(pattern_graph,
                                                                                                 pattern_node_id,
                                                                                                 pivot_id) else 1.0

        return distance_factor * position_importance * cluster_importance

    def _estimate_attr_modification_cost(self, data_graph, pattern_node_id, data_node_id, attr_name):
        if data_node_id not in data_graph.nodes:
            return 1.0

        node = data_graph.nodes[data_node_id]
        if not hasattr(node, 'attributes') or attr_name not in node.attributes:
            return 1.0

        attr_values = []
        gnn_labels = []

        for node_id, node_data in data_graph.nodes.items():
            if hasattr(node_data, 'attributes') and attr_name in node_data.attributes:
                attr_value = node_data.attributes[attr_name]
                attr_values.append(attr_value)

                gnn_label = node_data.attributes.get('gnn_label', 0)
                gnn_labels.append(gnn_label)

        if not attr_values:
            return 1.0

        try:
            gini_value = self._calculate_gini(attr_values, gnn_labels)

            normalized_cost = 1.0 + 9.0 * gini_value
            return normalized_cost
        except:
            return 1.0

    def _calculate_gini(self, attr_values, gnn_labels):
        if not attr_values or len(attr_values) != len(gnn_labels):
            return 0.0

        unique_values = sorted(set(attr_values))
        if len(unique_values) <= 1:
            return 0.0

        best_gini = 0.0
        for i in range(len(unique_values) - 1):
            split_value = (unique_values[i] + unique_values[i + 1]) / 2

            left_indices = [j for j, val in enumerate(attr_values) if val <= split_value]
            right_indices = [j for j, val in enumerate(attr_values) if val > split_value]

            if not left_indices or not right_indices:
                continue

            p_left = len(left_indices) / len(attr_values)
            p_right = len(right_indices) / len(attr_values)

            left_labels = [gnn_labels[j] for j in left_indices]
            right_labels = [gnn_labels[j] for j in right_indices]

            gini_left = self._calculate_group_gini(left_labels)
            gini_right = self._calculate_group_gini(right_labels)

            gini_split = p_left * gini_left + p_right * gini_right
            gini_improvement = self._calculate_group_gini(gnn_labels) - gini_split

            best_gini = max(best_gini, gini_improvement)

        return best_gini

    def _calculate_group_gini(self, labels):
        if not labels:
            return 0.0

        total = len(labels)
        label_counts = {}

        for label in labels:
            if label not in label_counts:
                label_counts[label] = 0
            label_counts[label] += 1

        gini = 1.0
        for count in label_counts.values():
            p = count / total
            gini -= p * p

        return gini

    def _estimate_perturbation_effectiveness(self, pattern_node_id, data_node_id, cand_space, pattern_graph, pivot_id):
        distance = self._compute_distance(pattern_graph, pattern_node_id, pivot_id)

        distance_factor = self.alpha ** distance

        position_importance = 3.0 if pattern_node_id == pivot_id else 2.0 if self._are_neighbors(pattern_graph,
                                                                                                 pattern_node_id,
                                                                                                 pivot_id) else 1.0

        return distance_factor * position_importance

    def _compute_distance(self, graph, source_id, target_id):
        if source_id == target_id:
            return 0

        visited = {source_id: 0}
        queue = deque([source_id])

        while queue:
            node_id = queue.popleft()
            current_distance = visited[node_id]

            neighbors = set()
            for edge in graph.edges:
                if edge[0] == node_id:
                    neighbors.add(edge[1])
                elif edge[1] == node_id:
                    neighbors.add(edge[0])

            for neighbor in neighbors:
                if neighbor not in visited:
                    visited[neighbor] = current_distance + 1
                    queue.append(neighbor)

                    if neighbor == target_id:
                        return visited[neighbor]

        return float('inf')

    def _are_neighbors(self, graph, node1_id, node2_id):
        for edge in graph.edges:
            if (edge[0] == node1_id and edge[1] == node2_id) or (edge[0] == node2_id and edge[1] == node1_id):
                return True
        return False

    def _get_node_degree(self, graph, node_id):
        degree = 0
        for edge in graph.edges:
            if edge[0] == node_id or edge[1] == node_id:
                degree += 1
        return degree

    def _check_edge_constraint(self, pattern_graph, pattern_id1, data_id1, pattern_id2, data_id2):
        edge_in_pattern = False
        for edge in pattern_graph.edges:
            if (edge[0] == pattern_id1 and edge[1] == pattern_id2) or (
                    edge[0] == pattern_id2 and edge[1] == pattern_id1):
                edge_in_pattern = True
                break

        if not edge_in_pattern:
            return True

        return True

    def _get_edges(self, graph, node_id):
        edges = []
        for edge in graph.edges:
            if edge[0] == node_id or edge[1] == node_id:
                edges.append(edge)
        return edges

    def _topological_sort(self, dag):
        in_degree = {node: 0 for node in dag}
        for node in dag:
            for child in dag[node]:
                in_degree[child] = in_degree.get(child, 0) + 1

        queue = deque([node for node in in_degree if in_degree[node] == 0])
        result = []

        while queue:
            node = queue.popleft()
            result.append(node)

            for child in dag[node]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        return result

    def _find_homomorphic_mappings(self, pattern_graph, data_graph, cand_space):
        mappings = []

        if any(len(candidates) == 0 for candidates in cand_space.values()):
            return mappings

        def backtrack(mapping, pattern_nodes_left):
            if not pattern_nodes_left:
                mappings.append(mapping.copy())
                return

            pattern_node = pattern_nodes_left[0]
            remaining = pattern_nodes_left[1:]

            for data_node in cand_space[pattern_node]:
                if data_node in mapping.values():
                    continue

                valid = True
                for mapped_pattern_node, mapped_data_node in mapping.items():
                    edge_in_pattern = False
                    for edge in pattern_graph.edges:
                        if (edge[0] == pattern_node and edge[1] == mapped_pattern_node) or \
                                (edge[0] == mapped_pattern_node and edge[1] == pattern_node):
                            edge_in_pattern = True
                            break

                    if edge_in_pattern:
                        edge_in_data = False
                        for edge in data_graph.edges:
                            if (edge[0] == data_node and edge[1] == mapped_data_node) or \
                                    (edge[0] == mapped_data_node and edge[1] == data_node):
                                edge_in_data = True
                                break

                        if not edge_in_data:
                            valid = False
                            break

                if valid:
                    mapping[pattern_node] = data_node
                    backtrack(mapping, remaining)
                    del mapping[pattern_node]

        pattern_nodes = list(pattern_graph.nodes.keys())
        backtrack({}, pattern_nodes)

        return mappings


def generate_counterfactual_explanation(subgraph, rxgnn_rule):
    explainer = CounterfactualExplainer(
        threshold=5,
        alpha=0.5,
        lambda_factor=10.0
    )

    attr_changes, edge_removals = explainer.explain(subgraph, rxgnn_rule)

    return attr_changes, edge_removals