import random
import threading
import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from tqdm import tqdm
import os
import pickle
from transformers import AutoModelForCausalLM, AutoTokenizer

from CFE import CounterfactualExplainer
from graph_matcher import Matcher, Pattern, Node, Graph, AttributePredicate, \
    AttributeComparisonPredicate, RxGNNs

MODEL_DIR = "models"
RULES_DIR = "rules"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RULES_DIR, exist_ok=True)

import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_dim + 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PatternComposer:
    def __init__(self, matcher, motifs, gnn_model=None, model_path="models/pattern_composer.pt"):
        self.matcher = matcher
        self.motifs = motifs
        self.model_path = model_path
        self.gnn_model = gnn_model

        self.state_dim = 4
        if gnn_model is not None:
            self.state_dim += 32

        self.action_dim = 10

        self.dqn = DQN(self.state_dim, self.action_dim)
        if os.path.exists(self.model_path):
            self.dqn.load_state_dict(torch.load(self.model_path))
        else:
            self.train_dqn(max_episodes=100)

    def get_state(self, pattern):
        num_nodes = len(pattern.graph.nodes)
        num_edges = len(pattern.graph.edges)
        support = self.matcher.get_pattern_support(pattern)
        confidence = self.matcher.get_pattern_confidence(pattern)

        state = [num_nodes, num_edges, support, confidence]

        if self.gnn_model is not None:
            gnn_features = self.extract_gnn_features(pattern)
            state.extend(gnn_features)

        return torch.tensor(state, dtype=torch.float32)

    def extract_gnn_features(self, pattern):
        if self.gnn_model is None:
            return []

        try:
            with torch.no_grad():
                features = self.gnn_model(pattern)
                return features.numpy().tolist()
        except Exception as e:
            return [0.0] * 32

    def get_reward(self, old_pattern, new_pattern):
        old_support = self.matcher.get_pattern_support(old_pattern)
        new_support = self.matcher.get_pattern_support(new_pattern)

        old_confidence = self.matcher.get_pattern_confidence(old_pattern)
        new_confidence = self.matcher.get_pattern_confidence(new_pattern)

        confidence_change = new_confidence - old_confidence

        support_change = (new_support - old_support) / max(1, old_support)

        reward = confidence_change + 0.1 * support_change

        return reward

    def train_dqn(self, max_episodes=100, gamma=0.99, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.95):
        from tqdm import tqdm

        optimizer = optim.Adam(self.dqn.parameters(), lr=0.001)
        memory = []
        epsilon = epsilon_start

        episode_progress = tqdm(range(max_episodes), desc="DQN Training Progress")
        for episode in episode_progress:
            if len(self.motifs) < 2:
                continue

            pattern1 = random.choice(self.motifs)
            pattern2 = random.choice(self.motifs)
            while pattern2 == pattern1 and len(self.motifs) > 1:
                pattern2 = random.choice(self.motifs)

            merge_candidates = self.get_merge_candidates(pattern1, pattern2)
            if not merge_candidates:
                continue

            state = self.get_state(pattern1)

            if random.random() < epsilon:
                if random.random() < 0.3:
                    action_idx = self.action_dim
                else:
                    action_idx = random.randint(0, min(self.action_dim - 1, len(merge_candidates) - 1))
            else:
                with torch.no_grad():
                    q_values = self.dqn(state)
                    valid_actions = min(self.action_dim, len(merge_candidates))
                    action_idx = torch.argmax(q_values[:valid_actions + 1]).item()

            if action_idx == self.action_dim:
                reward = 0
                next_state = state
            else:
                try:
                    merge_nodes = merge_candidates[action_idx]
                    new_pattern = self.matcher.merge_patterns(pattern1, pattern2, [merge_nodes])

                    next_state = self.get_state(new_pattern)
                    reward = self.get_reward(pattern1, new_pattern)
                except Exception as e:
                    continue

            memory.append((state, action_idx, reward, next_state))

            if len(memory) > 32:
                batch_size = min(32, len(memory))
                batch = random.sample(memory, batch_size)

                states, actions, rewards, next_states = zip(*batch)

                states = torch.stack(states)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards = torch.tensor(rewards, dtype=torch.float32)
                next_states = torch.stack(next_states)

                with torch.no_grad():
                    next_q_values = self.dqn(next_states)
                    max_next_q = next_q_values.max(1)[0]
                    target_q = rewards + gamma * max_next_q

                q_values = self.dqn(states)
                current_q = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                loss = F.mse_loss(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            episode_progress.set_postfix(
                {'epsilon': f"{epsilon:.4f}", 'memory': len(memory), 'reward': f"{reward:.4f}"})

        torch.save(self.dqn.state_dict(), self.model_path)

    def get_merge_candidates(self, pattern1, pattern2):
        candidates = []

        if pattern1.pivot_id and pattern2.pivot_id:
            pivot1 = pattern1.graph.nodes[pattern1.pivot_id]
            pivot2 = pattern2.graph.nodes[pattern2.pivot_id]

            if pivot1.label == pivot2.label:
                candidates.append((pattern1.pivot_id, pattern2.pivot_id))
                return candidates
            else:
                return []

        for node1_id, node1 in pattern1.graph.nodes.items():
            for node2_id, node2 in pattern2.graph.nodes.items():
                if node1.label == node2.label and (node1_id, node2_id) not in candidates:
                    candidates.append((node1_id, node2_id))

        return candidates

    def select_best_merge_or_stop(self, pattern1, pattern2):
        merge_candidates = self.get_merge_candidates(pattern1, pattern2)
        if not merge_candidates:
            return None, True

        state = self.get_state(pattern1)

        with torch.no_grad():
            q_values = self.dqn(state)

            candidates_count = len(merge_candidates)

            valid_length = min(len(q_values), candidates_count + 1)
            valid_q_values = q_values[:valid_length]

            best_idx = torch.argmax(valid_q_values).item()

            if best_idx >= candidates_count:
                return None, True

            if best_idx < candidates_count:
                return merge_candidates[best_idx], False
            else:
                return merge_candidates[0], False

class PredicateSelector:
    def __init__(self, matcher=None, ppl_file="ppl.pickle", model_name="Qwen/Qwen2.5-1.5B"):
        self.matcher = matcher
        self.ppl_file = ppl_file
        self.ppl_cache = {}
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.prompt_template = ""

        try:
            if os.path.exists(ppl_file):
                with open(ppl_file, 'rb') as f:
                    self.ppl_cache = pickle.load(f)
        except Exception as e:
            self.ppl_cache = {}

        try:
            with open("prompt.txt", "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
        except Exception as e:
            print(f"Unable to read prompt.txt: {e}")
            self.prompt_template = "Rule: {rule}\nWhat is the perplexity of this rule?"

    def _load_model(self):
        if self.model is None or self.tokenizer is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")

            print(f"Loading model {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(device)
            self.model.eval()

    def _predicate_to_natural_language(self, predicate, pattern):
        try:
            if hasattr(predicate, "description"):
                pred_description = predicate.description()
            else:
                return "Unknown predicate"

            if isinstance(predicate, AttributePredicate):
                node_id = predicate.node_id
                node_label = "Unknown"
                if pattern and node_id in pattern.graph.nodes:
                    node_label = pattern.graph.nodes[node_id].label

                op_map = {
                    "==": "equals",
                    ">": "greater than",
                    "<": "less than",
                    ">=": "greater than or equal to",
                    "<=": "less than or equal to",
                    "!=": "not equal to"
                }

                operator = op_map.get(predicate.operator, predicate.operator)

                return f"Node '{node_id}' of type '{node_label}' has attribute '{predicate.attribute}' {operator} {predicate.value}"

            elif isinstance(predicate, AttributeComparisonPredicate):
                node1_id = predicate.node1_id
                node2_id = predicate.node2_id
                node1_label = "Unknown"
                node2_label = "Unknown"

                if pattern:
                    if node1_id in pattern.graph.nodes:
                        node1_label = pattern.graph.nodes[node1_id].label
                    if node2_id in pattern.graph.nodes:
                        node2_label = pattern.graph.nodes[node2_id].label

                op_map = {
                    "==": "equals",
                    ">": "greater than",
                    "<": "less than",
                    ">=": "greater than or equal to",
                    "<=": "less than or equal to",
                    "!=": "not equal to"
                }

                operator = op_map.get(predicate.operator, predicate.operator)

                return f"Node '{node1_id}' of type '{node1_label}' has attribute '{predicate.attr1}' {operator} attribute '{predicate.attr2}' of node '{node2_id}' of type '{node2_label}'"

            return pred_description

        except Exception as e:
            return f"Predicate conversion error: {str(e)}"

    def calculate_ppl(self, predicate, pattern):
        pred_key = predicate.description()

        if pred_key in self.ppl_cache:
            return self.ppl_cache[pred_key]

        if self.matcher is not None:
            try:
                temp_rule = RxGNNs(pattern)
                temp_rule.add_precondition(predicate)

                result = self.matcher.evaluate_rule(temp_rule)
                support = result.get('support', 0)

                if support > 0:
                    ppl_value = -support
                    self.ppl_cache[pred_key] = ppl_value
                    return ppl_value
            except Exception as e:
                pass

        natural_language = self._predicate_to_natural_language(predicate, pattern)

        prompt = self.prompt_template.format(rule=natural_language)

        try:
            self._load_model()

            device = next(self.model.parameters()).device

            encodings = self.tokenizer(prompt, return_tensors="pt").to(device)

            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

                neg_log_likelihood = outputs.loss.item()

                ppl = np.exp(neg_log_likelihood)

            self.ppl_cache[pred_key] = ppl
            return ppl

        except Exception as e:
            print(f"PPL calculation failed: {e}, using random value instead")
            random_ppl = random.randint(1, 100)
            self.ppl_cache[pred_key] = random_ppl
            return random_ppl

    def rank_predicates(self, predicates, pattern):
        scored_predicates = []

        for pred in predicates:
            ppl = self.calculate_ppl(pred, pattern)
            scored_predicates.append((pred, ppl))

        return sorted(scored_predicates, key=lambda x: x[1])

    def save_ppl_cache(self, filename=None):
        if filename is None:
            filename = self.ppl_file

        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.ppl_cache, f)
        except Exception as e:
            print(f"Failed to save PPL cache: {e}")

class RuleDiscovery:
    def __init__(self, data_graph, motifs=None, support_threshold=5, confidence_threshold=0.5,
                 max_verification_time=50, max_pattern_combinations=3, ppl_file="ppl.pickle",
                 sample_ratio=0.1, min_nodes=100, gnn_model=None):
        self.original_graph = data_graph

        self.motifs_input = motifs
        self.support_threshold = support_threshold
        self.confidence_threshold = confidence_threshold
        self.max_verification_time = max_verification_time
        self.max_pattern_combinations = max_pattern_combinations
        self.gnn_model = gnn_model

        self.data_graph = self.sample_graph(data_graph, sample_ratio, min_nodes)

        self.matcher = Matcher(self.data_graph)

        self.motifs = motifs if motifs is not None else self.generate_random_motifs(k=5, max_nodes=3)

        self.composer = PatternComposer(self.matcher, self.motifs, gnn_model)
        self.predicate_selector = PredicateSelector(self.matcher, ppl_file)

        self.discovered_rules = []
        self.lock = threading.Lock()
        self.max_rules = 100
        self.pattern_combination_count = 0

    def sample_graph(self, original_graph, sample_ratio=0.1, min_nodes=100):
        if sample_ratio >= 1.0:
            return original_graph

        original_node_count = len(original_graph.nodes)
        target_node_count = max(min_nodes, int(original_node_count * sample_ratio))

        if target_node_count >= original_node_count:
            return original_graph

        from graph_matcher import Graph, Node
        sampled_graph = Graph()

        visited_nodes = set()

        pivot_label = None
        if self.motifs_input:
            for motif in self.motifs_input:
                if motif.pivot_id and motif.pivot_id in motif.graph.nodes:
                    pivot_label = motif.graph.nodes[motif.pivot_id].label
                    break

        potential_seeds = []
        if pivot_label:
            for node_id, node in original_graph.nodes.items():
                if node.label == pivot_label:
                    potential_seeds.append(node_id)

        if not potential_seeds:
            potential_seeds = list(original_graph.nodes.keys())

        num_seeds = min(len(potential_seeds), max(1, int(target_node_count ** 0.5)))
        seed_nodes = random.sample(potential_seeds, num_seeds)

        queue = deque(seed_nodes)
        for seed in seed_nodes:
            visited_nodes.add(seed)

        while queue and len(visited_nodes) < target_node_count:
            current_node_id = queue.popleft()

            neighbors = set()
            for edge in original_graph.edges:
                if edge[0] == current_node_id:
                    neighbors.add(edge[1])
                elif edge[1] == current_node_id:
                    neighbors.add(edge[0])

            neighbors = list(neighbors)
            random.shuffle(neighbors)

            for neighbor in neighbors:
                if neighbor not in visited_nodes:
                    visited_nodes.add(neighbor)
                    queue.append(neighbor)

                if len(visited_nodes) >= target_node_count:
                    break

        for node_id in visited_nodes:
            if node_id in original_graph.nodes:
                node = original_graph.nodes[node_id]
                sampled_graph.add_node(Node(
                    node_id,
                    node.label,
                    node.attributes.copy()
                ))

        for edge in original_graph.edges:
            if edge[0] in visited_nodes and edge[1] in visited_nodes:
                sampled_graph.add_edge(edge[0], edge[1])

        actual_ratio = len(sampled_graph.nodes) / original_node_count

        return sampled_graph

    def generate_random_motifs(self, k, max_nodes):
        if k <= 0 or max_nodes <= 0:
            raise ValueError("k and max_nodes must be positive integers")

        motifs = []
        seen_motifs = set()
        matcher = Matcher(self.data_graph)

        all_nodes = list(self.data_graph.nodes.keys())
        gnn_true_nodes = [nid for nid, node in self.data_graph.nodes.items()
                          if node.attributes.get('gnn_prediction', False)]

        attempts = 0
        max_attempts = k * 10

        while len(motifs) < k and attempts < max_attempts:
            attempts += 1

            if gnn_true_nodes and random.random() < 0.8:
                start_node_id = random.choice(gnn_true_nodes)
            else:
                start_node_id = random.choice(all_nodes)

            pattern = Pattern(pivot_id=f'm{len(motifs)}_0')
            visited_nodes = {start_node_id: f'm{len(motifs)}_0'}
            pattern.add_node(Node(visited_nodes[start_node_id],
                                  self.data_graph.nodes[start_node_id].label, {}))

            current_node = start_node_id
            for _ in range(max_nodes - 1):
                out_neighbors = list(self.data_graph.get_out_edges(current_node))
                in_neighbors = list(self.data_graph.get_in_edges(current_node))
                neighbors = out_neighbors + in_neighbors

                if not neighbors:
                    break

                next_node = random.choice(neighbors)
                if next_node in visited_nodes:
                    src_id = visited_nodes[current_node]
                    tgt_id = visited_nodes[next_node]
                    if (src_id, tgt_id) not in pattern.graph.edges:
                        pattern.add_edge(src_id, tgt_id)
                else:
                    new_node_id = f'm{len(motifs)}_{len(visited_nodes)}'
                    visited_nodes[next_node] = new_node_id
                    pattern.add_node(Node(new_node_id,
                                          self.data_graph.nodes[next_node].label, {}))
                    if next_node in out_neighbors:
                        pattern.add_edge(visited_nodes[current_node], new_node_id)
                    if next_node in in_neighbors:
                        pattern.add_edge(new_node_id, visited_nodes[current_node])

                current_node = next_node

            if len(pattern.graph.nodes) < 1:
                continue

            mappings = matcher.find_homomorphic_mappings(pattern, pivot_only=True)
            support = len(mappings)

            if support <= self.support_threshold:
                continue

            motif_key = self._serialize_pattern(pattern)
            if motif_key in seen_motifs:
                continue

            seen_motifs.add(motif_key)
            motifs.append(pattern)

        return motifs

    def _serialize_pattern(self, pattern):
        nodes_str = sorted([f"{nid}:{pattern.graph.nodes[nid].label}"
                            for nid in pattern.graph.nodes])
        edges_str = sorted([f"{src}->{tgt}" for src, tgt in pattern.graph.edges])
        return "|".join(nodes_str) + "#" + "|".join(edges_str)

    def generate_predicates(self, pattern):
        from tqdm import tqdm

        predicates = []

        label_attributes = {}
        attribute_values = {}

        for node_id, node in tqdm(self.data_graph.nodes.items(), desc="Analyzing node attributes", leave=False):
            label = node.label
            if label not in label_attributes:
                label_attributes[label] = set()
                attribute_values[label] = {}

            for attr, value in node.attributes.items():
                label_attributes[label].add(attr)

                if attr not in attribute_values[label]:
                    attribute_values[label][attr] = set()
                attribute_values[label][attr].add(value)

        for node_id, node in tqdm(pattern.graph.nodes.items(), desc="Generating node predicates", leave=False):
            label = node.label

            if label not in label_attributes:
                continue

            for attr in label_attributes[label]:
                values = attribute_values[label][attr]

                if all(isinstance(v, (int, float)) for v in values if v is not None):
                    numeric_values = [v for v in values if v is not None]
                    if not numeric_values:
                        continue

                    min_val = min(numeric_values)
                    max_val = max(numeric_values)

                    if min_val != max_val:
                        thresholds = [
                            min_val + (max_val - min_val) * 0.25,
                            min_val + (max_val - min_val) * 0.5,
                            min_val + (max_val - min_val) * 0.75
                        ]

                        for threshold in thresholds:
                            predicates.append(AttributePredicate(node_id, attr, threshold, '<'))
                            predicates.append(AttributePredicate(node_id, attr, threshold, '>'))

                elif all(isinstance(v, str) for v in values if v is not None):
                    if len(values) <= 10:
                        for value in values:
                            if value is not None:
                                predicates.append(AttributePredicate(node_id, attr, value, '=='))

                elif all(isinstance(v, bool) for v in values if v is not None):
                    predicates.append(AttributePredicate(node_id, attr, True, '=='))

                if attr == 'gnn_prediction' and node_id == pattern.pivot_id:
                    predicates.append(AttributePredicate(node_id, attr, False, '=='))

        node_pairs = [(n1, n2) for n1 in pattern.graph.nodes.keys() for n2 in pattern.graph.nodes.keys() if
                      n1 != n2]

        pair_progress = tqdm(node_pairs, desc="Generating comparison predicates", leave=False)
        for node1_id, node2_id in pair_progress:
            node1_label = pattern.graph.nodes[node1_id].label
            node2_label = pattern.graph.nodes[node2_id].label

            pair_progress.set_postfix(
                {"node1": f"{node1_id}({node1_label})", "node2": f"{node2_id}({node2_label})"})

            if node1_label not in label_attributes or node2_label not in label_attributes:
                continue

            common_attrs = set()
            for attr1 in label_attributes[node1_label]:
                for attr2 in label_attributes[node2_label]:
                    if attr1 == attr2:
                        values1 = attribute_values[node1_label][attr1]
                        values2 = attribute_values[node2_label][attr2]

                        if (all(isinstance(v, (int, float)) for v in values1 if v is not None) and
                            all(isinstance(v, (int, float)) for v in values2 if v is not None)) or \
                                (all(isinstance(v, str) for v in values1 if v is not None) and
                                 all(isinstance(v, str) for v in values2 if v is not None)):
                            common_attrs.add((attr1, attr2))

            for attr1, attr2 in common_attrs:
                predicates.append(AttributeComparisonPredicate(node1_id, attr1, node2_id, attr2, '=='))
                predicates.append(AttributeComparisonPredicate(node1_id, attr1, node2_id, attr2, '>'))
                predicates.append(AttributeComparisonPredicate(node1_id, attr1, node2_id, attr2, '<'))

        return predicates

    def evaluate_with_timeout(self, rule):
        from tqdm import tqdm
        import threading

        result = {'support': 0, 'confidence': 0}
        error_occurred = [False]
        completion_event = threading.Event()

        def evaluation_task():
            nonlocal result
            try:
                progress = tqdm(desc="Rule evaluation", leave=False, position=0)
                progress.set_description("Starting rule evaluation...")

                result = self.matcher.evaluate_rule(rule)

                progress.set_description("Rule evaluation completed")
                progress.update(1)
                progress.close()
            except Exception as e:
                error_occurred[0] = True
            finally:
                completion_event.set()

        thread = threading.Thread(target=evaluation_task)
        thread.daemon = True

        start_time = time.time()
        thread.start()

        timeout_bar = tqdm(total=self.max_verification_time, desc="Waiting for rule verification", leave=False, position=1)

        for _ in range(self.max_verification_time):
            if completion_event.is_set():
                break
            time.sleep(1)
            timeout_bar.update(1)

        timeout_bar.close()

        if not completion_event.is_set():
            elapsed = time.time() - start_time
            return {'support': 0, 'confidence': 0}

        if error_occurred[0]:
            return {'support': 0, 'confidence': 0}

        support = result.get('support', 0)
        confidence = result.get('confidence', 0.0)

        return result

    def level_wise_mining(self, pattern, thread_id):
        from tqdm import tqdm

        try:
            predicates = self.generate_predicates(pattern)

            ranked_predicates = self.predicate_selector.rank_predicates(predicates, pattern)

            gnn_pred = None
            for pred, _ in ranked_predicates:
                if (isinstance(pred, AttributePredicate) and
                        pred.node_id == pattern.pivot_id and
                        pred.attribute == 'gnn_prediction' and
                        pred.value == True and
                        pred.operator == '=='):
                    gnn_pred = pred
                    break

            if not gnn_pred and pattern.pivot_id:
                try:
                    gnn_pred = AttributePredicate(pattern.pivot_id, 'gnn_prediction', True, '==')
                except Exception as e:
                    return

            if not gnn_pred:
                return

            ranked_predicates = [(p, s) for p, s in ranked_predicates if p != gnn_pred]

            if not ranked_predicates:
                return

            base_rule = RxGNNs(pattern)

            result = self.evaluate_with_timeout(base_rule)

            if 'support' not in result:
                result['support'] = 0
            if 'confidence' not in result:
                result['confidence'] = 0

            if (result['support'] >= self.support_threshold and
                    result['confidence'] >= self.confidence_threshold):
                with self.lock:
                    if len(self.discovered_rules) < self.max_rules:
                        self.discovered_rules.append(base_rule)

            from collections import deque
            queue = deque([set()])

            explored_combinations = set([frozenset()])

            exploration_progress = tqdm(desc=f"Thread {thread_id}: Predicate exploration", position=thread_id % 10)
            combo_count = 0

            while queue and len(self.discovered_rules) < self.max_rules:
                current_predicates = queue.popleft()
                combo_count += 1
                exploration_progress.update(1)
                exploration_progress.set_description(f"Thread {thread_id}: Explored {combo_count} combinations")

                current_rule = RxGNNs(pattern)
                for pred in current_predicates:
                    current_rule.add_precondition(pred)

                current_combo = frozenset(p.description() for p in current_predicates)

                try:
                    result = self.evaluate_with_timeout(current_rule)
                    if 'support' not in result:
                        result['support'] = 0
                    if 'confidence' not in result:
                        result['confidence'] = 0

                    exploration_progress.set_postfix({
                        'support': result['support'],
                        'confidence': f"{result['confidence']:.4f}",
                        'pred_count': len(current_predicates)
                    })
                except Exception as e:
                    continue

                if (result['support'] >= self.support_threshold and
                        result['confidence'] >= self.confidence_threshold):
                    with self.lock:
                        if len(self.discovered_rules) < self.max_rules:
                            self.discovered_rules.append(current_rule)
                            new_rule_count = len(self.discovered_rules)

                for pred, score in ranked_predicates:
                    if pred in current_predicates:
                        continue

                    new_predicates = current_predicates.copy()
                    new_predicates.add(pred)

                    new_combo = frozenset(p.description() for p in new_predicates)
                    if new_combo in explored_combinations:
                        continue

                    explored_combinations.add(new_combo)

                    queue.append(new_predicates)

                    if len(queue) > 1000:
                        break

            exploration_progress.close()
        except Exception as e:
            pass

    def worker(self, pattern_idx, thread_id):
        from tqdm import tqdm

        try:
            if pattern_idx < 0 or pattern_idx >= len(self.motifs):
                return

            pattern = self.motifs[pattern_idx]

            if len(self.discovered_rules) >= self.max_rules:
                return

            try:
                self.level_wise_mining(pattern, thread_id)
                current_rules = len(self.discovered_rules)
            except Exception as e:
                pass

            if len(self.discovered_rules) >= self.max_rules:
                return

            with self.lock:
                if self.pattern_combination_count >= self.max_pattern_combinations:
                    return

            combine_progress = tqdm(enumerate(self.motifs), desc=f"Thread {thread_id}: Pattern combinations",
                                    position=thread_id % 10)

            for i, other_pattern in combine_progress:
                if i == pattern_idx:
                    continue

                with self.lock:
                    if self.pattern_combination_count >= self.max_pattern_combinations:
                        break
                    self.pattern_combination_count += 1
                    current_count = self.pattern_combination_count

                combine_progress.set_postfix({
                    "combinations": current_count,
                    "target pattern": i,
                    "discovered rules": len(self.discovered_rules)
                })

                try:
                    best_merge, should_stop = self.composer.select_best_merge_or_stop(pattern, other_pattern)

                    if should_stop:
                        combine_progress.set_postfix({"status": "stop merging", "combinations": current_count})
                        continue

                    if best_merge:
                        combine_progress.set_postfix({"status": "merging", "combinations": current_count})
                        merged_pattern = self.matcher.merge_patterns(pattern, other_pattern, [best_merge])

                        try:
                            self.level_wise_mining(merged_pattern, thread_id)
                            current_rules = len(self.discovered_rules)
                        except Exception as e:
                            pass
                except Exception as e:
                    pass

            combine_progress.close()

        except Exception as e:
            pass

    def discover(self, num_threads=4):
        from tqdm import tqdm

        start_time = time.time()

        work_items = []

        for i in range(len(self.motifs)):
            work_items.append((i, f"Single Motif-{i}"))

        for i in range(len(self.motifs)):
            for j in range(i + 1, len(self.motifs)):
                work_items.append((-1, f"Combination-{i}-{j}"))

        progress_bar = tqdm(total=len(work_items), desc="Overall mining progress")
        rules_progress = tqdm(total=self.max_rules, desc="Discovered rules")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for idx, (pattern_idx, name) in enumerate(work_items):
                if pattern_idx >= 0:
                    futures.append(executor.submit(self.worker, pattern_idx, idx))
                else:
                    pass

            for future in futures:
                try:
                    future.result()
                    progress_bar.update(1)
                    current_rules = len(self.discovered_rules)
                    rules_progress.n = min(current_rules, self.max_rules)
                    rules_progress.refresh()
                except Exception as e:
                    progress_bar.update(1)

        progress_bar.close()
        rules_progress.close()

        end_time = time.time()

        duration = end_time - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)

        self.save_rules()

        return self.discovered_rules

    def save_rules(self, filename="rules/discovered_rules.pkl"):
        serializable_rules = []

        for rule in self.discovered_rules:
            new_rule = RxGNNs(rule.pattern)
            new_rule.add_preconditions(rule.preconditions)
            serializable_rules.append(new_rule)

        try:
            with open(filename, 'wb') as f:
                pickle.dump(serializable_rules, f)
        except Exception as e:
            fallback_filename = f"{filename}.txt"
            with open(fallback_filename, 'w', encoding='utf-8') as f:
                for i, rule in enumerate(self.discovered_rules):
                    f.write(f"Rule {i + 1}: {rule.description()}\n")

    def load_rules(self, filename="rules/discovered_rules.pkl"):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.discovered_rules = pickle.load(f)
            return True
        return False