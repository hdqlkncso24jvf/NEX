import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_max_pool
import numpy as np
import pickle
import time
import argparse
from tqdm import tqdm
from collections import defaultdict, Counter
import math
import copy
import random


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=3, dropout=0.5,
                 gnn_type='GCN', readout='mean', num_classes=2, device='cuda'):
        super(GNNModel, self).__init__()

        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.readout = readout

        self.convs = nn.ModuleList()

        if gnn_type == 'GCN':
            self.convs.append(GCNConv(input_dim, hidden_dim))
        elif gnn_type == 'GAT':
            self.convs.append(GATConv(input_dim, hidden_dim))
        elif gnn_type == 'GIN':
            self.convs.append(GINConv(nn=nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            )))
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")

        for i in range(num_layers - 1):
            if gnn_type == 'GCN':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'GAT':
                self.convs.append(GATConv(hidden_dim, hidden_dim))
            elif gnn_type == 'GIN':
                self.convs.append(GINConv(nn=nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.to(device)

    def forward(self, x, edge_index, batch=None, center_idx=None):
        center_features = None
        if center_idx is not None:
            center_features = x[center_idx].clone()

        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if center_idx is not None:
            center_features = x[center_idx]

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

        if self.readout == 'mean':
            readout_features = global_mean_pool(x, batch)
        elif self.readout == 'max':
            readout_features = global_max_pool(x, batch)
        else:
            raise ValueError(f"Unknown readout function: {self.readout}")

        if center_features is not None:
            if readout_features.size(0) > 1:
                center_batch = batch[center_idx]
                graph_readout = readout_features[center_batch].unsqueeze(0)
            else:
                graph_readout = readout_features

            combined_features = torch.cat([graph_readout, center_features.unsqueeze(0)], dim=1)
        else:
            combined_features = torch.cat([readout_features, readout_features], dim=1)

        x = self.classifier(combined_features)

        return x


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


class GNNInterpreter:
    def __init__(self, model, device='cuda', learning_rate=0.01, embedding_dim=None, max_nodes=10):
        self.model = model
        self.device = device
        self.embedding_dim = embedding_dim or model.input_dim
        self.learning_rate = learning_rate
        self.max_nodes = max_nodes

        self.edge_logits = nn.Parameter(
            torch.zeros(max_nodes, max_nodes, device=device)
        )

        self.node_embeddings = nn.Parameter(
            torch.randn(max_nodes, self.embedding_dim, device=device) * 0.1
        )

        self.temperature = 0.2

        self.optimizer = torch.optim.Adam(
            [self.edge_logits, self.node_embeddings],
            lr=learning_rate
        )

        self.training = True

    def is_connected(self, adjacency_matrix):
        num_nodes = adjacency_matrix.shape[0]
        visited = [False] * num_nodes

        def dfs(node):
            visited[node] = True
            for neighbor in range(num_nodes):
                if adjacency_matrix[node, neighbor] > 0.5 and not visited[neighbor]:
                    dfs(neighbor)

        dfs(0)

        return all(visited)

    def ensure_connectivity(self, edges, num_nodes):
        binary_edges = (edges > 0.5).float()

        if self.is_connected(binary_edges):
            return edges

        components = []
        visited = [False] * num_nodes

        def find_component(node, component):
            visited[node] = True
            component.append(node)
            for neighbor in range(num_nodes):
                if binary_edges[node, neighbor] > 0.5 and not visited[neighbor]:
                    find_component(neighbor, component)

        for i in range(num_nodes):
            if not visited[i]:
                component = []
                find_component(i, component)
                components.append(component)

        modified_edges = edges.clone()
        for i in range(len(components) - 1):
            node1 = components[i][0]
            node2 = components[i + 1][0]

            modified_edges[node1, node2] = 1.0
            modified_edges[node2, node1] = 1.0

        return modified_edges

    def sample_graph(self, hard=False, num_nodes=None):
        if num_nodes is None:
            num_nodes = self.max_nodes

        edge_weights = torch.sigmoid((self.edge_logits[:num_nodes, :num_nodes] +
                                      self.edge_logits[:num_nodes, :num_nodes].transpose(0, 1)) / 2)

        if self.training:
            eps = 1e-10
            uniform = torch.rand_like(edge_weights)
            gumbel_noise = -torch.log(-torch.log(uniform + eps) + eps)
            continuous_edges = torch.sigmoid(
                (edge_weights.log() - (1 - edge_weights).log() + gumbel_noise) / self.temperature)
        else:
            continuous_edges = edge_weights

        continuous_edges = self.ensure_connectivity(continuous_edges, num_nodes)

        if hard:
            discrete_edges = (continuous_edges > 0.5).float()
            edges = (discrete_edges - continuous_edges).detach() + continuous_edges
        else:
            edges = continuous_edges

        node_features = self.node_embeddings[:num_nodes]

        return edges, node_features

    def compute_reward(self, edges, node_features, target_class=0):
        num_nodes = node_features.shape[0]

        if self.training:
            edge_index = []
            edge_weight = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        edge_index.append([i, j])
                        edge_weight.append(edges[i, j].item())

            if edge_index:
                edge_index = torch.tensor(edge_index, dtype=torch.long, device=self.device).t()
                edge_weight = torch.tensor(edge_weight, dtype=torch.float, device=self.device)
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
                edge_weight = torch.zeros(0, dtype=torch.float, device=self.device)

            self.model.eval()
            with torch.no_grad():
                output = self.model(node_features, edge_index)

            probs = F.softmax(output, dim=1)
            target_prob = probs[0, target_class].item()
        else:
            binary_edges = (edges > 0.5).float()
            edge_list = torch.nonzero(binary_edges).tolist()

            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

            self.model.eval()
            with torch.no_grad():
                output = self.model(node_features, edge_index)

            probs = F.softmax(output, dim=1)
            target_prob = probs[0, target_class].item()

        size_penalty = -0.02 * num_nodes

        return target_prob + size_penalty

    def compute_similarity(self, node_features, edges, class_avg_embedding):
        num_nodes = node_features.shape[0]
        binary_edges = (edges > 0.5).float()
        edge_list = torch.nonzero(binary_edges).tolist()

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            graph_embedding = self.model.get_graph_embedding(node_features, edge_index)

        similarity = F.cosine_similarity(graph_embedding, class_avg_embedding.unsqueeze(0), dim=1)

        return similarity.item()

    def generate_explanation(self, target_class=0, class_avg_embedding=None, num_nodes=None, max_steps=20000, mu=1.0):
        if num_nodes is None:
            num_nodes = self.max_nodes

        self.edge_logits.data.zero_()
        self.node_embeddings.data.normal_(0, 0.1)

        rewards = []
        similarities = []

        self.training = True
        for step in range(max_steps):
            self.optimizer.zero_grad()

            edges, node_features = self.sample_graph(hard=False, num_nodes=num_nodes)

            reward = self.compute_reward(edges, node_features, target_class)

            similarity = 0
            if class_avg_embedding is not None:
                similarity = self.compute_similarity(node_features, edges, class_avg_embedding)

            objective = reward + mu * similarity

            l1_reg = 0.01 * torch.mean(torch.abs(self.edge_logits[:num_nodes, :num_nodes]))
            l2_reg = 0.005 * torch.mean(torch.square(self.edge_logits[:num_nodes, :num_nodes]))

            degree = torch.sum(edges, dim=1)
            degree_reg = 0.1 * torch.mean(torch.abs(degree - 2))

            loss = -objective + l1_reg + l2_reg + degree_reg

            loss.backward()
            self.optimizer.step()

            rewards.append(reward)
            similarities.append(similarity)

        self.training = False
        edges, node_features = self.sample_graph(hard=True, num_nodes=num_nodes)

        final_graph = Graph()

        for i in range(num_nodes):
            node = Node(i, i % 5)
            node.features = node_features[i].detach()
            final_graph.add_node(node)

        binary_edges = (edges > 0.5).detach().cpu().numpy()
        for i in range(num_nodes):
            for j in range(num_nodes):
                if binary_edges[i, j] > 0.5 and i != j:
                    final_graph.add_edge(i, j)

        final_reward = self.compute_reward(edges, node_features, target_class)

        return final_graph, final_reward

    def generate_multiple_explanations(self, n_explanations=5, target_class=0, class_avg_embedding=None, max_nodes=7,
                                       min_nodes=3):
        explanations = []
        start_time = time.time()

        for i in range(n_explanations):
            num_nodes = random.randint(min_nodes, max_nodes)

            graph, reward = self.generate_explanation(
                target_class=target_class,
                class_avg_embedding=class_avg_embedding,
                num_nodes=num_nodes
            )

            explanations.append((graph, reward))

        train_time = time.time() - start_time

        explanations.sort(key=lambda x: x[1], reverse=True)

        return explanations, train_time

    def evaluate_recognizability(self, explanations, target_class=0):
        recognizable_count = 0

        for graph, _ in explanations:
            num_nodes = len(graph.nodes)

            node_features = []
            for node_id in graph.nodes:
                if hasattr(graph.nodes[node_id], 'features'):
                    node_features.append(graph.nodes[node_id].features)
                else:
                    feat = torch.zeros(self.embedding_dim, device=self.device)
                    feat[graph.nodes[node_id].label % self.embedding_dim] = 1.0
                    node_features.append(feat)

            x = torch.stack(node_features)

            edge_list = []
            node_id_to_idx = {nid: i for i, nid in enumerate(graph.nodes.keys())}

            for src, tgt in graph.edges:
                src_idx = node_id_to_idx[src]
                tgt_idx = node_id_to_idx[tgt]
                edge_list.append([src_idx, tgt_idx])

            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

            self.model.eval()
            with torch.no_grad():
                output = self.model(x, edge_index)
                pred = output.argmax(dim=1).item()

            if pred == target_class:
                recognizable_count += 1

        recognizability = recognizable_count / len(explanations) if explanations else 0

        return recognizability

    def is_pattern_match(self, pattern_graph, test_graph, similarity_threshold=0.5):
        if len(pattern_graph.nodes) > len(test_graph.nodes):
            return False, {}

        pattern_label_counts = Counter()
        for node_id, node in pattern_graph.nodes.items():
            pattern_label_counts[node.label] += 1

        test_label_counts = Counter()
        for node_id, node in test_graph.nodes.items():
            test_label_counts[node.label] += 1

        for label, count in pattern_label_counts.items():
            if test_label_counts.get(label, 0) < count:
                return False, {}

        node_candidates = {}
        for p_id, p_node in pattern_graph.nodes.items():
            candidates = []
            for t_id, t_node in test_graph.nodes.items():
                if t_node.label == p_node.label:
                    if hasattr(p_node, 'features'):
                        p_features = p_node.features.detach().cpu()
                    else:
                        p_features = torch.zeros(self.embedding_dim).cpu()

                    if hasattr(t_node, 'features'):
                        t_features = t_node.features.detach().cpu()
                    else:
                        t_features = torch.zeros(self.embedding_dim).cpu()

                    p_features = p_features.view(-1)
                    t_features = t_features.view(-1)
                    similarity = F.cosine_similarity(p_features.unsqueeze(0), t_features.unsqueeze(0), dim=1).item()

                    if similarity >= similarity_threshold:
                        candidates.append((t_id, similarity))

            if not candidates:
                return False, {}

            node_candidates[p_id] = candidates

        def backtrack(pattern_idx, mapping, used_test_nodes):
            if pattern_idx == len(pattern_graph.nodes):
                return mapping

            pattern_ids = list(pattern_graph.nodes.keys())
            p_id = pattern_ids[pattern_idx]

            for t_id, _ in node_candidates[p_id]:
                if t_id in used_test_nodes:
                    continue

                edge_compatible = True

                for neighbor_p_id in pattern_graph.nodes:
                    if neighbor_p_id not in mapping:
                        continue

                    if (p_id, neighbor_p_id) in pattern_graph.edges:
                        neighbor_t_id = mapping[neighbor_p_id]
                        if (t_id, neighbor_t_id) not in test_graph.edges:
                            edge_compatible = False
                            break

                    if (neighbor_p_id, p_id) in pattern_graph.edges:
                        neighbor_t_id = mapping[neighbor_p_id]
                        if (neighbor_t_id, t_id) not in test_graph.edges:
                            edge_compatible = False
                            break

                if edge_compatible:
                    new_mapping = mapping.copy()
                    new_mapping[p_id] = t_id
                    new_used = used_test_nodes.union({t_id})

                    result = backtrack(pattern_idx + 1, new_mapping, new_used)
                    if result:
                        return result

            return None

        mapping = backtrack(0, {}, set())

        return mapping is not None, mapping if mapping else {}

    def evaluate_reliability(self, explanations, test_samples, target_class=0, similarity_threshold=0.5,
                             sample_size=100):
        covered_samples = 0
        matching_details = {}

        explanation_matches = [0] * len(explanations)

        target_test_samples = [(i, test_graph, test_label)
                               for i, (test_graph, test_label) in enumerate(test_samples)
                               if test_label == target_class]

        total_available = len(target_test_samples)
        sample_size = min(sample_size, total_available)

        if sample_size < total_available:
            sampled_indices = random.sample(range(total_available), sample_size)
            sampled_test_samples = [target_test_samples[i] for i in sampled_indices]
        else:
            sampled_test_samples = target_test_samples

        for idx, test_graph, _ in tqdm(sampled_test_samples, desc="Evaluating reliability"):
            sample_matched = False

            for j, (pattern, _) in enumerate(explanations):
                is_match, mapping = self.is_pattern_match(pattern, test_graph, similarity_threshold)
                if is_match:
                    sample_matched = True
                    explanation_matches[j] += 1
                    if idx not in matching_details:
                        matching_details[idx] = []
                    matching_details[idx].append((j, mapping))

            if sample_matched:
                covered_samples += 1

        reliability = covered_samples / len(sampled_test_samples) if sampled_test_samples else 0

        for j, matches in enumerate(explanation_matches):
            coverage = matches / len(sampled_test_samples) if sampled_test_samples else 0

        return reliability, matching_details


def load_test_samples(samples_path):
    with open(samples_path, 'rb') as f:
        samples = pickle.load(f)

    test_graphs = []
    for center_id, graph in samples:
        g = Graph()

        for node_id, node_data in graph.nodes.items():
            node = Node(node_id, node_data.label, node_data.attributes)
            if hasattr(node_data, 'features'):
                node.features = node_data.features
            g.add_node(node)

        for source, target in graph.edges:
            g.add_edge(source, target)

        label = graph.nodes[center_id].attributes.get('gnn_prediction', 0)

        test_graphs.append((g, label))

    return test_graphs


def main():
    parser = argparse.ArgumentParser(description='GNNInterpreter for GNN model explanation')
    parser.add_argument('--model_path', type=str, default="models/loan/GCN_model.pt",
                        help='Path to saved model state dict')
    parser.add_argument('--samples_path', type=str, default="models/loan/loan_samples.pkl",
                        help='Path to test samples')
    parser.add_argument('--processed_data_path', type=str, default=None,
                        help='Path to processed data with model configuration')
    parser.add_argument('--target_class', type=int, default=0,
                        help='Target class to explain (0 or 1)')
    parser.add_argument('--num_explanations', type=int, default=50,
                        help='Number of explanations to generate')
    parser.add_argument('--min_nodes', type=int, default=5,
                        help='Minimum nodes in generated patterns')
    parser.add_argument('--max_nodes', type=int, default=10,
                        help='Maximum nodes in generated patterns')
    parser.add_argument('--similarity_threshold', type=float, default=0.3,
                        help='Threshold for node similarity in matching')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for training')
    parser.add_argument('--max_batches', type=int, default=3,
                        help='Maximum batches per epoch')
    parser.add_argument('--output', type=str, default='gnninterpreter_results.txt',
                        help='Output file for results')
    parser.add_argument('--model_type', type=str, default='GCN', choices=['GCN', 'GAT', 'GIN'],
                        help='GNN model type')
    parser.add_argument('--dataset', type=str, default='loan',
                        choices=['loan', 'insurance', 'trans'],
                        help='Dataset name')
    parser.add_argument('--input_dim', type=int, default=32,
                        help='Input dimension for the model')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for the model')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers')

    args = parser.parse_args()

    if args.model_path == "models/loan/GCN_model.pt":
        args.model_path = f"models/{args.dataset}/{args.model_type}_model.pt"

    if args.processed_data_path is None:
        args.processed_data_path = f"models/{args.dataset}/{args.model_type}_processed_data.pkl"

    if args.samples_path == "models/loan/loan_samples.pkl":
        args.samples_path = f"models/{args.dataset}/{args.dataset}_samples.pkl"

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    try:
        with open(args.processed_data_path, 'rb') as f:
            processed_data = pickle.load(f)
            first_graph = next(iter(processed_data.values()))
            input_dim = first_graph['features'].size(1)
    except (FileNotFoundError, AttributeError, KeyError) as e:
        input_dim = args.input_dim

    model = GNNModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.5,
        gnn_type=args.model_type,
        readout='mean',
        num_classes=2,
        device=device
    )

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    test_samples = load_test_samples(args.samples_path)

    explainer = GNNInterpreter(
        model,
        device=device,
        embedding_dim=input_dim,
        max_nodes=args.max_nodes
    )

    target_class = args.target_class
    class_avg_embeddings = {}

    target_samples = [(g, l) for g, l in test_samples if l == target_class]

    if target_samples:
        embeddings = []
        for graph, _ in target_samples:
            node_features = []
            for node_id in graph.nodes:
                if hasattr(graph.nodes[node_id], 'features'):
                    node_features.append(graph.nodes[node_id].features)
                else:
                    feat = torch.zeros(input_dim, device=device)
                    feat[graph.nodes[node_id].label % input_dim] = 1.0
                    node_features.append(feat)

            if node_features:
                x = torch.stack(node_features)
            else:
                x = torch.zeros((0, input_dim), device=device)

            edge_list = []
            node_id_to_idx = {nid: i for i, nid in enumerate(graph.nodes.keys())}

            for src, tgt in graph.edges:
                src_idx = node_id_to_idx[src]
                tgt_idx = node_id_to_idx[tgt]
                edge_list.append([src_idx, tgt_idx])

            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)

            model.eval()
            with torch.no_grad():
                if not hasattr(model, 'get_graph_embedding'):
                    def get_graph_embedding(x, edge_index, batch=None):
                        h = x
                        for conv in model.convs:
                            h = conv(h, edge_index)
                            h = F.relu(h)
                            h = F.dropout(h, p=model.dropout, training=False)

                        if batch is None:
                            batch = torch.zeros(h.size(0), dtype=torch.long, device=device)

                        if model.readout == 'mean':
                            return global_mean_pool(h, batch)
                        elif model.readout == 'max':
                            return global_max_pool(h, batch)
                        else:
                            return global_mean_pool(h, batch)

                    model.get_graph_embedding = get_graph_embedding

                graph_embedding = model.get_graph_embedding(x, edge_index)
                embeddings.append(graph_embedding)

        if embeddings:
            embeddings = torch.cat(embeddings, dim=0)
            class_avg_embeddings[target_class] = torch.mean(embeddings, dim=0)
        else:
            class_avg_embeddings[target_class] = None
    else:
        class_avg_embeddings[target_class] = None

    explanations, train_time = explainer.generate_multiple_explanations(
        n_explanations=args.num_explanations,
        target_class=args.target_class,
        class_avg_embedding=class_avg_embeddings.get(args.target_class),
        max_nodes=args.max_nodes,
        min_nodes=args.min_nodes
    )

    for i, (graph, reward) in enumerate(explanations):
        for node_id, node in graph.nodes.items():
            pass
        for src, tgt in graph.edges:
            pass

    recognizability = explainer.evaluate_recognizability(explanations, args.target_class)

    reliability, match_details = explainer.evaluate_reliability(
        explanations,
        test_samples,
        args.target_class,
        args.similarity_threshold
    )

if __name__ == "__main__":
    main()