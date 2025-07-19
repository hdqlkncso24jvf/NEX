import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_max_pool
import numpy as np
import pickle
import time
from tqdm import tqdm
import argparse
import random
import copy
import matplotlib.pyplot as plt
import networkx as nx


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

        logits = self.classifier(combined_features)

        return logits


class InduCEExplainer:
    def __init__(self, model, device='cuda', gnn_layers=3, hidden_dim=64, eta=0.1, gamma=0.4):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.gnn_layers = gnn_layers
        self.hidden_dim = hidden_dim
        self.eta = eta
        self.gamma = gamma

        self.policy_network = None

    def initialize_policy_network(self, input_feature_dim):
        self.policy_network = PolicyNetwork(
            input_feature_dim=input_feature_dim,
            hidden_dim=self.hidden_dim,
            gnn_layers=self.gnn_layers,
            device=self.device
        )

    def predict(self, x, edge_index, center_idx):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x, edge_index, None, center_idx)
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1).item()
            confidence = probs[0, pred].item()
        return pred, confidence, probs[0, 1].item()

    def get_node_features(self, x, edge_index, node_idx, t=0):
        node_features = x[node_idx].clone()

        degrees = torch.zeros(x.size(0), device=self.device)
        unique_nodes, counts = torch.unique(edge_index[0], return_counts=True)
        degrees[unique_nodes] = counts.float()
        degree_feature = degrees[node_idx].unsqueeze(-1)

        with torch.no_grad():
            outputs = self.model(x, edge_index, None, torch.tensor(node_idx, device=self.device))
            probs = F.softmax(outputs, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            entropy_feature = entropy.unsqueeze(-1)

        pred_class = torch.argmax(outputs, dim=1).item()
        num_classes = outputs.size(1)
        class_feature = F.one_hot(torch.tensor([pred_class]), num_classes=num_classes).float().to(self.device).squeeze(
            0)

        combined_features = torch.cat([node_features, degree_feature, entropy_feature, class_feature], dim=0)

        return combined_features

    def get_n_hop_neighborhood(self, G, center_node, n_hops=2):
        neighborhood = set([center_node])
        frontier = set([center_node])

        for _ in range(n_hops):
            new_frontier = set()
            for node in frontier:
                for neighbor in G.neighbors(node):
                    if neighbor not in neighborhood:
                        new_frontier.add(neighbor)
                        neighborhood.add(neighbor)
            frontier = new_frontier

        neighborhood_list = [center_node] + [n for n in neighborhood if n != center_node]
        return neighborhood_list

    def compute_edge_reward(self, x, edge_index, center_idx, current_edge, is_addition, orig_fraud_prob):
        if is_addition:
            if not isinstance(current_edge, torch.Tensor):
                current_edge = torch.tensor(current_edge, dtype=torch.long, device=self.device)
            new_edge = torch.cat([current_edge.view(-1, 1), current_edge.flip(0).view(-1, 1)], dim=1).t()
            temp_edge_index = torch.cat([edge_index, new_edge], dim=1)
        else:
            mask = ~(((edge_index[0] == current_edge[0]) & (edge_index[1] == current_edge[1])) |
                     ((edge_index[0] == current_edge[1]) & (edge_index[1] == current_edge[0])))
            temp_edge_index = edge_index[:, mask]

        _, _, new_fraud_prob = self.predict(x, temp_edge_index, center_idx)

        reward = -(new_fraud_prob - orig_fraud_prob)

        return reward

    def generate_counterfactual(self, x, edge_index, center_idx, max_iterations=200,
                                edge_budget=5, feature_perturbation_range=0.2):
        self.model.eval()
        start_time = time.time()

        cf_x = x.clone().to(self.device)
        cf_edge_index = edge_index.clone().to(self.device)

        orig_pred, orig_conf, orig_fraud_prob = self.predict(x, edge_index, center_idx)
        if orig_pred != 1:
            return False, edge_index, x, {
                'success': False,
                'message': 'Original prediction is not fraud',
                'edges_removed': 0,
                'edges_added': 0,
                'nodes_modified': 0,
                'time': 0
            }

        edge_budget = 5
        G = nx.Graph()
        num_nodes = x.size(0)
        for i in range(num_nodes):
            G.add_node(i)

        edge_list = edge_index.t().tolist()
        unique_edges = set()
        for src, dst in edge_list:
            if (src, dst) not in unique_edges and (dst, src) not in unique_edges:
                unique_edges.add((src, dst))
                G.add_edge(src, dst)

        removed_edges = set()
        added_edges = set()
        modified_nodes = set()

        n_hops = 2
        neighborhood = self.get_n_hop_neighborhood(G, center_idx.item(), n_hops=n_hops)

        edges_modified = 0
        current_pred = orig_pred
        current_fraud_prob = orig_fraud_prob

        possible_deletions = [(u, v) for u, v in unique_edges if u in neighborhood or v in neighborhood]

        possible_additions = []
        for node in neighborhood:
            for neighbor in neighborhood:
                if node != neighbor and not G.has_edge(node, neighbor):
                    possible_additions.append((node, neighbor))

        if len(possible_additions) > 100:
            random.shuffle(possible_additions)
            possible_additions = possible_additions[:100]

        while edges_modified < edge_budget and current_pred == 1:
            edge_rewards = {}

            for edge in possible_deletions:
                reward = self.compute_edge_reward(cf_x, cf_edge_index, center_idx, edge, False, current_fraud_prob)
                edge_rewards[('deletion', edge)] = reward

            for edge in possible_additions:
                reward = self.compute_edge_reward(cf_x, cf_edge_index, center_idx, edge, True, current_fraud_prob)
                edge_rewards[('addition', edge)] = reward

            if not edge_rewards:
                break

            best_action, best_edge = max(edge_rewards.items(), key=lambda x: x[1])[0]

            if best_action == 'deletion':
                mask = ~(((cf_edge_index[0] == best_edge[0]) & (cf_edge_index[1] == best_edge[1])) |
                         ((cf_edge_index[0] == best_edge[1]) & (cf_edge_index[1] == best_edge[0])))
                cf_edge_index = cf_edge_index[:, mask]

                removed_edges.add(best_edge)
                possible_deletions.remove(best_edge)

                possible_additions.append(best_edge)
            else:
                new_edge = torch.tensor([[best_edge[0], best_edge[1]],
                                         [best_edge[1], best_edge[0]]], dtype=torch.long, device=self.device).t()
                cf_edge_index = torch.cat([cf_edge_index, new_edge], dim=1)

                added_edges.add(best_edge)
                possible_additions.remove(best_edge)

                possible_deletions.append(best_edge)

            current_pred, _, current_fraud_prob = self.predict(cf_x, cf_edge_index, center_idx)

            edges_modified += 1

            if current_pred == 0:
                break

        if current_pred == 1:
            perturbations = {}
            for idx, node_idx in enumerate(neighborhood):
                perturbations[node_idx] = torch.zeros_like(cf_x[node_idx], requires_grad=True, device=self.device)

            optimizers = {}
            schedulers = {}
            learning_rate = 0.01
            for node_idx in perturbations:
                node_lr = learning_rate * (2.0 if node_idx == center_idx.item() else 1.0)
                optimizers[node_idx] = torch.optim.Adam([perturbations[node_idx]], lr=node_lr)
                schedulers[node_idx] = torch.optim.lr_scheduler.LambdaLR(
                    optimizers[node_idx],
                    lr_lambda=lambda epoch: min(1.0 + epoch / (max_iterations / 4), 3.0)
                )

            adaptation_params = {
                'aggressiveness': 0.5,
                'stagnation_counter': 0,
                'best_fraud_prob': 1.0,
                'consecutive_no_improvement': 0,
                'exploration_phase': False
            }

            for iteration in range(max_iterations):
                perturbed_x = cf_x.clone()
                for node_idx, perturb in perturbations.items():
                    perturbed_x[node_idx] = cf_x[node_idx] + perturb

                self.model.eval()
                outputs = self.model(perturbed_x, cf_edge_index, None, center_idx)
                probs = F.softmax(outputs, dim=1)
                fraud_prob = probs[0, 1].item()

                if fraud_prob < adaptation_params['best_fraud_prob']:
                    adaptation_params['best_fraud_prob'] = fraud_prob
                    adaptation_params['consecutive_no_improvement'] = 0

                    current_pred = 0 if fraud_prob < 0.5 else 1
                    if current_pred == 0:
                        for node_idx in perturbations:
                            if torch.any(perturbations[node_idx] != 0):
                                modified_nodes.add(node_idx)
                        break
                else:
                    adaptation_params['consecutive_no_improvement'] += 1

                self._update_adaptation_strategy(
                    adaptation_params,
                    iteration,
                    max_iterations,
                    perturbations,
                    fraud_prob,
                    center_idx.item()
                )

                loss = probs[0, 1]

                reg_coef = max(0.01, 0.1 - 0.1 * adaptation_params['aggressiveness'])
                l2_reg = sum(torch.sum(p ** 2) for p in perturbations.values())
                loss = loss + reg_coef * l2_reg

                loss.backward()

                for node_idx in optimizers:
                    if adaptation_params['exploration_phase']:
                        scale = 2.0 if node_idx == center_idx.item() else 1.0
                        perturbations[node_idx].grad *= scale

                    optimizers[node_idx].step()
                    optimizers[node_idx].zero_grad()
                    schedulers[node_idx].step()

                if iteration % 10 == 0 and 0.4 < fraud_prob < 0.6:
                    with torch.no_grad():
                        center_perturbation = perturbations[center_idx.item()]
                        if center_perturbation.grad is not None:
                            direction = -torch.sign(center_perturbation.grad)
                            center_perturbation.data += direction * learning_rate * 5.0

            if current_pred == 1:
                cf_x = cf_x.clone()
                for node_idx, perturb in perturbations.items():
                    if torch.any(perturb != 0):
                        cf_x[node_idx] = cf_x[node_idx] + perturb
                        modified_nodes.add(node_idx)

                current_pred, _, current_fraud_prob = self.predict(cf_x, cf_edge_index, center_idx)

        time_taken = time.time() - start_time
        stats = {
            'success': current_pred == 0,
            'original_prediction': orig_pred,
            'final_prediction': current_pred,
            'edges_removed': len(removed_edges),
            'edges_added': len(added_edges),
            'edges_modified': len(removed_edges) + len(added_edges),
            'nodes_modified': len(modified_nodes),
            'time': time_taken,
            'removed_edges': removed_edges,
            'added_edges': added_edges,
            'modified_nodes': modified_nodes,
            'original_fraud_prob': orig_fraud_prob,
            'final_fraud_prob': current_fraud_prob
        }

        return stats['success'], cf_edge_index, cf_x, stats

    def _update_adaptation_strategy(self, params, iteration, max_iterations, perturbations, current_fraud_prob,
                                    center_node):
        if params['consecutive_no_improvement'] > 10:
            params['aggressiveness'] = min(1.0, params['aggressiveness'] * 1.2)
            params['stagnation_counter'] += 1
            params['consecutive_no_improvement'] = 0

            if params['stagnation_counter'] >= 3:
                params['exploration_phase'] = True

                with torch.no_grad():
                    scale = 0.5 * params['aggressiveness']
                    noise = torch.randn_like(perturbations[center_node]) * scale
                    perturbations[center_node].data += noise

                    for node_idx in list(perturbations.keys())[:5]:
                        if node_idx != center_node:
                            small_noise = torch.randn_like(perturbations[node_idx]) * scale * 0.5
                            perturbations[node_idx].data += small_noise

        progress = iteration / max_iterations
        if progress > 0.5 and not params['exploration_phase'] and current_fraud_prob > 0.6:
            params['aggressiveness'] = min(1.0, params['aggressiveness'] + 0.1)

        elif progress > 0.8 and current_fraud_prob > 0.55:
            params['aggressiveness'] = 1.0
            params['exploration_phase'] = True


class PolicyNetwork(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim=64, gnn_layers=3, device='cuda'):
        super(PolicyNetwork, self).__init__()
        self.device = device

        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(input_feature_dim, hidden_dim))

        for _ in range(gnn_layers - 1):
            self.gat_layers.append(GATConv(hidden_dim, hidden_dim))

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.to(device)

    def forward(self, x, edge_index, actions):
        h = x
        for gat in self.gat_layers:
            h = gat(h, edge_index)
            h = F.relu(h)

        action_scores = []
        for action_type, (src, dst) in actions:
            edge_embedding = torch.cat([h[src], h[dst]], dim=0)

            action_feature = torch.tensor([1.0 if action_type == 'addition' else 0.0],
                                          device=self.device)

            combined = torch.cat([edge_embedding, action_feature], dim=0)

            score = self.mlp(combined)
            action_scores.append(score.item())

        return torch.tensor(action_scores, device=self.device)


def load_model_and_data(dataset_name, model_type, device='cuda'):
    model_path = f"models/{dataset_name}/{model_type}_model.pt"

    data_path = f"models/{dataset_name}/{model_type}_all_samples_features.pkl"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path, 'rb') as f:
        all_processed_samples = pickle.load(f)

    first_sample = next(iter(all_processed_samples.values()))
    input_dim = first_sample['features'].size(1)

    model = GNNModel(
        input_dim=input_dim,
        hidden_dim=64,
        num_layers=3,
        dropout=0.5,
        gnn_type=model_type,
        readout='mean',
        num_classes=2,
        device=device
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, all_processed_samples


def run_explanation_experiment(dataset_name, model_type, num_samples=50, device='cuda'):
    model, all_samples = load_model_and_data(dataset_name, model_type, device)

    explainer = InduCEExplainer(model, device)

    sample_keys = list(all_samples.keys())
    random.shuffle(sample_keys)
    sample_keys = sample_keys[:min(num_samples, len(sample_keys))]

    results = []
    fraud_samples = 0

    for sample_key in tqdm(sample_keys, desc="Generating counterfactuals"):
        sample = all_samples[sample_key]

        x = sample['features'].to(device)
        edge_index = sample['edge_index'].to(device)
        center_idx = torch.tensor(sample['center_idx'], device=device)

        with torch.no_grad():
            outputs = model(x, edge_index, None, center_idx)
            pred = torch.argmax(outputs, dim=1).item()

        if pred == 1:
            fraud_samples += 1

            success, cf_edge_index, cf_x, stats = explainer.generate_counterfactual(
                x, edge_index, center_idx,
                max_iterations=150,
                edge_budget=5
            )

            results.append(stats)

    successful_results = [r for r in results if r['success']]

    if fraud_samples == 0:
        return None

    fidelity = len(successful_results) / fraud_samples if fraud_samples > 0 else 0

    if not successful_results:
        return {
            'dataset': dataset_name,
            'model': model_type,
            'fidelity': 0,
            'avg_sparsity': 0,
            'avg_time': 0,
            'avg_edges_removed': 0,
            'avg_edges_added': 0,
            'avg_nodes_modified': 0,
            'num_samples_predicted_fraud': fraud_samples,
            'num_successful': 0
        }

    avg_edges_removed = sum(r['edges_removed'] for r in successful_results) / len(successful_results)
    avg_edges_added = sum(r['edges_added'] for r in successful_results) / len(successful_results)
    avg_nodes_modified = sum(r['nodes_modified'] for r in successful_results) / len(successful_results)
    avg_time = sum(r['time'] for r in successful_results) / len(successful_results)

    sparsity_values = []
    for result in successful_results:
        sample_key = sample_keys[results.index(result)]
        sample = all_samples[sample_key]
        total_edges = sample['edge_index'].size(1) // 2

        sparsity = (result['edges_removed'] + result['edges_added']) / total_edges if total_edges > 0 else 0
        sparsity_values.append(sparsity)

    avg_sparsity = sum(sparsity_values) / len(sparsity_values)

    return {
        'dataset': dataset_name,
        'model': model_type,
        'fidelity': fidelity,
        'avg_sparsity': avg_sparsity,
        'avg_time': avg_time,
        'avg_edges_removed': avg_edges_removed,
        'avg_edges_added': avg_edges_added,
        'avg_nodes_modified': avg_nodes_modified,
        'num_samples_predicted_fraud': fraud_samples,
        'num_successful': len(successful_results)
    }


def visualize_counterfactual(sample, cf_edge_index, cf_x, stats, dataset_name, model_type, sample_id):
    viz_dir = f"counterfactual_viz/{dataset_name}/{model_type}"
    os.makedirs(viz_dir, exist_ok=True)

    x = sample['features']
    edge_index = sample['edge_index']
    center_idx = sample['center_idx']

    G_orig = nx.Graph()
    G_cf = nx.Graph()

    for i in range(x.size(0)):
        G_orig.add_node(i)
        G_cf.add_node(i)

    edge_list_orig = edge_index.t().cpu().numpy()
    unique_edges_orig = set()
    for i in range(edge_list_orig.shape[0]):
        src, dst = edge_list_orig[i]
        if (src, dst) not in unique_edges_orig and (dst, src) not in unique_edges_orig:
            unique_edges_orig.add((src, dst))
            G_orig.add_edge(src, dst)

    if cf_edge_index.numel() > 0:
        edge_list_cf = cf_edge_index.t().cpu().numpy()
        unique_edges_cf = set()
        for i in range(edge_list_cf.shape[0]):
            src, dst = edge_list_cf[i]
            if (src, dst) not in unique_edges_cf and (dst, src) not in unique_edges_cf:
                unique_edges_cf.add((src, dst))
                G_cf.add_edge(src, dst)

    removed_edges = stats['removed_edges']

    added_edges = stats['added_edges']

    modified_nodes = stats['modified_nodes']

    plt.figure(figsize=(20, 10))

    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G_orig, seed=42)

    nx.draw_networkx_nodes(G_orig, pos,
                           nodelist=[i for i in G_orig.nodes() if i != center_idx],
                           node_color='lightblue',
                           node_size=300)

    nx.draw_networkx_nodes(G_orig, pos,
                           nodelist=[center_idx],
                           node_color='red',
                           node_size=500)

    nx.draw_networkx_edges(G_orig, pos, width=1.0, alpha=0.7)

    nx.draw_networkx_labels(G_orig, pos)

    plt.title(f"Original Graph (Prediction: Fraud, P={stats['original_fraud_prob']:.2f})")
    plt.axis('off')

    plt.subplot(1, 2, 2)

    nx.draw_networkx_nodes(G_cf, pos,
                           nodelist=[i for i in G_cf.nodes() if i != center_idx and i not in modified_nodes],
                           node_color='lightblue',
                           node_size=300)

    nx.draw_networkx_nodes(G_cf, pos,
                           nodelist=[center_idx],
                           node_color='green' if center_idx not in modified_nodes else 'yellow',
                           node_size=500)

    nx.draw_networkx_nodes(G_cf, pos,
                           nodelist=[i for i in modified_nodes if i != center_idx],
                           node_color='yellow',
                           node_size=400)

    regular_edges = [(u, v) for u, v in G_cf.edges()
                     if (u, v) not in added_edges and (v, u) not in added_edges]
    nx.draw_networkx_edges(G_cf, pos, edgelist=regular_edges, width=1.0, alpha=0.7)

    added_edges_list = [(u, v) for u, v in G_cf.edges()
                        if (u, v) in added_edges or (v, u) in added_edges]
    nx.draw_networkx_edges(G_cf, pos, edgelist=added_edges_list,
                           width=2.0, edge_color='green', style='dashed')

    nx.draw_networkx_labels(G_cf, pos)

    plt.title(f"Counterfactual Graph (Prediction: Non-Fraud, P={stats['final_fraud_prob']:.2f})\n"
              f"{stats['edges_removed']} edges removed, {stats['edges_added']} edges added, "
              f"{stats['nodes_modified']} nodes modified")
    plt.axis('off')

    plt.savefig(f"{viz_dir}/counterfactual_{sample_id}.png", bbox_inches='tight')
    plt.close()
