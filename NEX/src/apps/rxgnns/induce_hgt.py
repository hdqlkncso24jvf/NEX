import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
import math
import numpy as np
import pickle
import time
from tqdm import tqdm
import argparse
import random
import copy
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight, gain=0.1)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        output = self.W_o(attention_output)
        return output
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        tensuu = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            tensuu = tensuu.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(tensuu, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        return output


class HGTLayer(nn.Module):
    def __init__(self, node_types, edge_types, in_dim, out_dim, num_heads=4, dropout=0.1):
        super(HGTLayer, self).__init__()
        
        self.node_types = node_types
        self.edge_types = edge_types
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.d_k = out_dim // num_heads
        
        if out_dim % num_heads != 0:
            out_dim = (out_dim // num_heads) * num_heads
            self.out_dim = out_dim
            self.d_k = out_dim // num_heads
        
        self.k_linears = nn.ModuleDict()
        self.q_linears = nn.ModuleDict()
        self.v_linears = nn.ModuleDict()
        self.a_linears = nn.ModuleDict()
        
        for src_type in node_types:
            self.k_linears[src_type] = nn.Linear(in_dim, out_dim, bias=False)
            self.v_linears[src_type] = nn.Linear(in_dim, out_dim, bias=False)
            
        for dst_type in node_types:
            self.q_linears[dst_type] = nn.Linear(in_dim, out_dim, bias=False)
            
        for edge_type in edge_types:
            src_type, rel_type, dst_type = edge_type
            edge_key = f"{src_type}_{rel_type}_{dst_type}"
            self.a_linears[edge_key] = nn.Linear(out_dim, num_heads, bias=False)
        
        self.message_linears = nn.ModuleDict()
        for dst_type in node_types:
            self.message_linears[dst_type] = nn.Linear(out_dim, out_dim)
            
        self.agg_linears = nn.ModuleDict()
        for node_type in node_types:
            self.agg_linears[node_type] = nn.Linear(out_dim, out_dim)
            
        self.residual_linears = nn.ModuleDict()
        self.layer_norms = nn.ModuleDict()
        for node_type in node_types:
            self.residual_linears[node_type] = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            self.layer_norms[node_type] = nn.LayerNorm(out_dim)
        
        self.dropout_layer = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x_dict, edge_index_dict):
        k_dict = {}
        q_dict = {}
        v_dict = {}
        
        for node_type in self.node_types:
            if node_type in x_dict and x_dict[node_type].size(0) > 0:
                x = x_dict[node_type]
                k_dict[node_type] = self.k_linears[node_type](x)
                q_dict[node_type] = self.q_linears[node_type](x)
                v_dict[node_type] = self.v_linears[node_type](x)
            else:
                k_dict[node_type] = torch.zeros((0, self.out_dim), device=next(self.parameters()).device)
                q_dict[node_type] = torch.zeros((0, self.out_dim), device=next(self.parameters()).device)
                v_dict[node_type] = torch.zeros((0, self.out_dim), device=next(self.parameters()).device)
        
        new_x_dict = {}
        
        for dst_type in self.node_types:
            if dst_type not in q_dict or q_dict[dst_type].size(0) == 0:
                new_x_dict[dst_type] = torch.zeros((0, self.out_dim), device=next(self.parameters()).device)
                continue
                
            dst_q = q_dict[dst_type]
            num_dst_nodes = dst_q.size(0)
            
            gesammelte_nachrichten = torch.zeros_like(dst_q)
            total_attention_weights = torch.zeros(num_dst_nodes, device=dst_q.device)
            
            for edge_type in self.edge_types:
                src_type, rel_type, target_type = edge_type
                
                if (target_type != dst_type or 
                    edge_type not in edge_index_dict or 
                    edge_index_dict[edge_type].size(1) == 0 or
                    src_type not in k_dict or k_dict[src_type].size(0) == 0):
                    continue
                
                edge_index = edge_index_dict[edge_type]
                src_k = k_dict[src_type]
                src_v = v_dict[src_type]
                
                if (edge_index[0].max() >= src_k.size(0) or 
                    edge_index[1].max() >= dst_q.size(0)):
                    continue
                
                edge_src_k = src_k[edge_index[0]]
                edge_src_v = src_v[edge_index[0]]
                edge_dst_q = dst_q[edge_index[1]]
                
                edge_src_k = edge_src_k.view(-1, self.num_heads, self.d_k)
                edge_dst_q = edge_dst_q.view(-1, self.num_heads, self.d_k)
                
                attention_scores = torch.sum(edge_src_k * edge_dst_q, dim=-1)
                attention_scores = attention_scores / math.sqrt(self.d_k)
                
                edge_key = f"{src_type}_{rel_type}_{target_type}"
                if edge_key in self.a_linears:
                    edge_attention = self.a_linears[edge_key](edge_src_k.view(-1, self.out_dim))
                    attention_scores = attention_scores + edge_attention
                
                attention_weights = F.softmax(attention_scores, dim=-1)
                
                edge_src_v = edge_src_v.view(-1, self.num_heads, self.d_k)
                attended_values = attention_weights.unsqueeze(-1) * edge_src_v
                attended_values = attended_values.view(-1, self.out_dim)
                
                edge_messages = torch.zeros_like(dst_q)
                edge_messages.index_add_(0, edge_index[1], attended_values)
                
                edge_weight_sum = torch.zeros(num_dst_nodes, device=dst_q.device)
                edge_weight_sum.index_add_(0, edge_index[1], attention_weights.sum(dim=-1))
                
                gesammelte_nachrichten += edge_messages
                total_attention_weights += edge_weight_sum
            
            total_attention_weights = torch.clamp(total_attention_weights, min=1e-8)
            gesammelte_nachrichten = gesammelte_nachrichten / total_attention_weights.unsqueeze(-1)
            
            if dst_type in self.message_linears:
                gesammelte_nachrichten = self.message_linears[dst_type](gesammelte_nachrichten)
            
            if dst_type in self.agg_linears:
                gesammelte_nachrichten = self.agg_linears[dst_type](gesammelte_nachrichten)
            
            new_x_dict[dst_type] = gesammelte_nachrichten
        
        output_dict = {}
        for node_type in self.node_types:
            if node_type in x_dict and x_dict[node_type].size(0) > 0:
                residual = self.residual_linears[node_type](x_dict[node_type])
                output = new_x_dict[node_type] + residual
                
                output = self.layer_norms[node_type](output)
                
                output = self.dropout_layer(output)
                
                output_dict[node_type] = output
            else:
                output_dict[node_type] = torch.zeros((0, self.out_dim), device=next(self.parameters()).device)
        
        return output_dict


class HGTModel(nn.Module):
    def __init__(self, node_types, edge_types, input_dims, hidden_dim=64, num_layers=3, 
                 num_heads=4, dropout=0.5, num_classes=2, primary_node_type=None, device='cuda'):
        super(HGTModel, self).__init__()
        
        self.device = device
        self.node_types = node_types
        self.edge_types = edge_types
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.primary_node_type = primary_node_type or node_types[0]
        
        if hidden_dim % num_heads != 0:
            hidden_dim = (hidden_dim // num_heads) * num_heads
            self.hidden_dim = hidden_dim
        
        self.input_projections = nn.ModuleDict()
        for node_type in node_types:
            proj = nn.Linear(input_dims[node_type], hidden_dim)
            nn.init.xavier_uniform_(proj.weight, gain=0.1)
            nn.init.constant_(proj.bias, 0)
            self.input_projections[node_type] = proj
        
        self.hgt_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = HGTLayer(
                node_types=node_types,
                edge_types=edge_types,
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            self.hgt_layers.append(layer)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.to(device)
    
    def forward(self, x_dict, edge_index_dict, center_idx=None):
        h_dict = {}
        for node_type in self.node_types:
            if node_type in x_dict and x_dict[node_type].size(0) > 0:
                x = x_dict[node_type]
                
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"Warning: NaN/Inf in input {node_type}, replacing with zeros")
                    x = torch.zeros_like(x)
                
                x_norm = torch.norm(x, dim=1, keepdim=True)
                x_norm = torch.clamp(x_norm, min=1e-8)
                x = x / x_norm
                
                h = self.input_projections[node_type](x)
                
                if torch.isnan(h).any() or torch.isinf(h).any():
                    print(f"Warning: NaN/Inf after projection in {node_type}")
                    h = torch.zeros_like(h)
                
                h = F.layer_norm(h, h.shape[1:])
                h_dict[node_type] = h
            else:
                h_dict[node_type] = torch.zeros((0, self.hidden_dim), device=self.device)
        
        processed_edge_index_dict = {}
        for edge_type in self.edge_types:
            if edge_type in edge_index_dict and edge_index_dict[edge_type].size(1) > 0:
                edge_index = edge_index_dict[edge_type]
                
                src_type, rel_type, dst_type = edge_type
                max_src = h_dict[src_type].size(0)
                max_dst = h_dict[dst_type].size(0)
                
                if (max_src > 0 and max_dst > 0 and 
                    (edge_index[0] < max_src).all() and (edge_index[1] < max_dst).all()):
                    processed_edge_index_dict[edge_type] = edge_index
                else:
                    processed_edge_index_dict[edge_type] = torch.zeros((2, 0), dtype=torch.long, device=self.device)
            else:
                processed_edge_index_dict[edge_type] = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        
        for layer_idx, layer in enumerate(self.hgt_layers):
            try:
                h_dict = layer(h_dict, processed_edge_index_dict)
                
                for node_type in self.node_types:
                    if (node_type in h_dict and h_dict[node_type].size(0) > 0 and 
                        (torch.isnan(h_dict[node_type]).any() or torch.isinf(h_dict[node_type]).any())):
                        print(f"Warning: NaN/Inf in {node_type} after HGT layer {layer_idx}")
                        h_dict[node_type] = torch.zeros_like(h_dict[node_type])
                
            except Exception as e:
                print(f"Error in HGT layer {layer_idx}: {e}")
                break
        
        center_features = None
        if (center_idx is not None and self.primary_node_type in h_dict and 
            h_dict[self.primary_node_type].size(0) > 0 and center_idx < h_dict[self.primary_node_type].size(0)):
            center_features = h_dict[self.primary_node_type][center_idx]
            
            if torch.isnan(center_features).any() or torch.isinf(center_features).any():
                print("Warning: NaN/Inf in center features")
                center_features = torch.zeros(self.hidden_dim, device=self.device)
        
        if self.primary_node_type in h_dict and h_dict[self.primary_node_type].size(0) > 0:
            global_features = torch.mean(h_dict[self.primary_node_type], dim=0)
            
            if torch.isnan(global_features).any() or torch.isinf(global_features).any():
                print("Warning: NaN/Inf in global features")
                global_features = torch.zeros(self.hidden_dim, device=self.device)
        else:
            global_features = torch.zeros(self.hidden_dim, device=self.device)
        
        if center_features is not None:
            combined_features = torch.cat([center_features, global_features]).unsqueeze(0)
        else:
            combined_features = torch.cat([global_features, global_features]).unsqueeze(0)
        
        if torch.isnan(combined_features).any() or torch.isinf(combined_features).any():
            print("Warning: NaN/Inf in final features")
            combined_features = torch.zeros_like(combined_features)
        
        output = self.classifier(combined_features)
        return output


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

    def predict(self, x_dict, edge_index_dict, center_idx):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_dict, edge_index_dict, center_idx)
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1).item()
            confidence = probs[0, pred].item()
        return pred, confidence, probs[0, 1].item()

    def get_n_hop_neighborhood(self, edge_index_dict, center_node, primary_node_type, n_hops=2):
        G = nx.Graph()
        
        for edge_type, edge_index in edge_index_dict.items():
            src_type, rel_type, dst_type = edge_type
            
            if src_type == primary_node_type or dst_type == primary_node_type:
                edge_list = edge_index.t().cpu().numpy()
                for src, dst in edge_list:
                    if src_type == primary_node_type:
                        global_src = src
                    else:
                        global_src = f"{src_type}_{src}"
                    
                    if dst_type == primary_node_type:
                        global_dst = dst
                    else:
                        global_dst = f"{dst_type}_{dst}"
                    
                    G.add_edge(global_src, global_dst)
        
        linju = set([center_node])
        frontier = set([center_node])
        
        for _ in range(n_hops):
            new_frontier = set()
            for node in frontier:
                if node in G:
                    for neighbor in G.neighbors(node):
                        if neighbor not in linju:
                            new_frontier.add(neighbor)
                            linju.add(neighbor)
            frontier = new_frontier
        
        primary_neighborhood = [n for n in linju if isinstance(n, int)]
        return primary_neighborhood

    def compute_edge_reward(self, x_dict, edge_index_dict, center_idx, primary_node_type, 
                            current_edge, edge_type, is_addition, orig_fraud_prob):
        new_edge_index_dict = copy.deepcopy(edge_index_dict)
        
        if is_addition:
            if not isinstance(current_edge, torch.Tensor):
                current_edge = torch.tensor(current_edge, dtype=torch.long, device=self.device)
            new_edge = current_edge.view(2, 1)
            if edge_type in new_edge_index_dict:
                new_edge_index_dict[edge_type] = torch.cat([new_edge_index_dict[edge_type], new_edge], dim=1)
            else:
                new_edge_index_dict[edge_type] = new_edge
        else:
            if edge_type in new_edge_index_dict:
                edge_index = new_edge_index_dict[edge_type]
                mask = ~((edge_index[0] == current_edge[0]) & (edge_index[1] == current_edge[1]))
                new_edge_index_dict[edge_type] = edge_index[:, mask]
        
        _, _, new_fraud_prob = self.predict(x_dict, new_edge_index_dict, center_idx)
        
        recompense = -(new_fraud_prob - orig_fraud_prob)
        
        return recompense

    def generate_counterfactual(self, x_dict, edge_index_dict, center_idx, primary_node_type,
                                max_iterations=200, edge_budget=5, feature_perturbation_range=0.2):
        self.model.eval()
        start_time = time.time()
        
        cf_x_dict = {k: v.clone().to(self.device) for k, v in x_dict.items()}
        cf_edge_index_dict = {k: v.clone().to(self.device) for k, v in edge_index_dict.items()}
        
        orig_pred, orig_conf, orig_fraud_prob = self.predict(x_dict, edge_index_dict, center_idx)
        
        removed_edges = set()
        added_edges = set()
        modified_nodes = set()
        
        n_hops = 2
        linju = self.get_n_hop_neighborhood(edge_index_dict, center_idx, primary_node_type, n_hops=n_hops)
        
        target_pred = 1 - orig_pred
        current_pred = orig_pred
        current_fraud_prob = orig_fraud_prob
        
        edges_modified = 0
        
        possible_deletions = {}
        possible_additions = {}
        
        for edge_type in edge_index_dict.keys():
            src_type, rel_type, dst_type = edge_type
            
            if src_type == primary_node_type or dst_type == primary_node_type:
                edge_index = edge_index_dict[edge_type]
                edge_list = edge_index.t().cpu().numpy()
                
                possible_deletions[edge_type] = []
                for src, dst in edge_list:
                    if (src_type == primary_node_type and src in linju) or \
                       (dst_type == primary_node_type and dst in linju):
                        possible_deletions[edge_type].append((src, dst))
                
                possible_additions[edge_type] = []
                if src_type == primary_node_type and dst_type == primary_node_type:
                    for src in linju[:30]:
                        for dst in linju[:30]:
                            if src != dst and (src, dst) not in possible_deletions[edge_type] and \
                               (dst, src) not in possible_deletions[edge_type]:
                                possible_additions[edge_type].append((src, dst))
        
        while edges_modified < edge_budget * 2 and current_pred != target_pred:
            edge_rewards = {}
            
            for edge_type, edge_list in possible_deletions.items():
                for edge in edge_list:
                    recompense = self.compute_edge_reward(
                        cf_x_dict, cf_edge_index_dict, center_idx, primary_node_type,
                        edge, edge_type, False, current_fraud_prob
                    )
                    if target_pred == 0:
                        edge_rewards[('deletion', edge_type, edge)] = recompense
                    else:
                        edge_rewards[('deletion', edge_type, edge)] = -recompense
            
            for edge_type, edge_list in possible_additions.items():
                for edge in edge_list[:20]:
                    recompense = self.compute_edge_reward(
                        cf_x_dict, cf_edge_index_dict, center_idx, primary_node_type,
                        edge, edge_type, True, current_fraud_prob
                    )
                    if target_pred == 0:
                        edge_rewards[('addition', edge_type, edge)] = recompense
                    else:
                        edge_rewards[('addition', edge_type, edge)] = -recompense
            
            if not edge_rewards:
                break
                
            best_action, best_edge_type, best_edge = max(edge_rewards.items(), key=lambda x: x[1])[0]
            
            if best_action == 'deletion':
                edge_index = cf_edge_index_dict[best_edge_type]
                mask = ~((edge_index[0] == best_edge[0]) & (edge_index[1] == best_edge[1]))
                cf_edge_index_dict[best_edge_type] = edge_index[:, mask]
                
                removed_edges.add((best_edge_type, best_edge))
                possible_deletions[best_edge_type].remove(best_edge)
                
                if best_edge not in possible_additions[best_edge_type]:
                    possible_additions[best_edge_type].append(best_edge)
            else:
                new_edge = torch.tensor([[best_edge[0]], [best_edge[1]]], dtype=torch.long, device=self.device)
                cf_edge_index_dict[best_edge_type] = torch.cat([cf_edge_index_dict[best_edge_type], new_edge], dim=1)
                
                added_edges.add((best_edge_type, best_edge))
                possible_additions[best_edge_type].remove(best_edge)
                
                if best_edge not in possible_deletions[best_edge_type]:
                    possible_deletions[best_edge_type].append(best_edge)
            
            current_pred, _, current_fraud_prob = self.predict(cf_x_dict, cf_edge_index_dict, center_idx)
            edges_modified += 1
            
            if current_pred == target_pred:
                break
        
        if current_pred != target_pred and primary_node_type in cf_x_dict:
            perturbations = {}
            primary_x = cf_x_dict[primary_node_type]
            
            valid_neighborhood = [n for n in linju if n < primary_x.size(0)]
            
            for node_idx in valid_neighborhood:
                initial_perturbation = torch.randn_like(primary_x[node_idx], device=self.device) * 0.5
                perturbations[node_idx] = initial_perturbation.requires_grad_(True)
            
            optimizers = {}
            learning_rate = 0.05
            for node_idx in perturbations:
                node_lr = learning_rate * (3.0 if node_idx == center_idx else 1.5)
                optimizers[node_idx] = torch.optim.Adam([perturbations[node_idx]], lr=node_lr)
            
            adaptation_params = {
                'aggressiveness': 1.0,
                'stagnation_counter': 0,
                'best_distance': float('inf'),
                'consecutive_no_improvement': 0,
                'exploration_phase': True
            }
            
            for iteration in range(max_iterations):
                perturbed_x_dict = copy.deepcopy(cf_x_dict)
                for node_idx, perturb in perturbations.items():
                    perturbed_x_dict[primary_node_type][node_idx] = cf_x_dict[primary_node_type][node_idx] + perturb
                
                self.model.eval()
                outputs = self.model(perturbed_x_dict, cf_edge_index_dict, center_idx)
                probs = F.softmax(outputs, dim=1)
                fraud_prob = probs[0, 1].item()
                
                if target_pred == 0:
                    distance_to_target = fraud_prob
                    current_pred = 0 if fraud_prob < 0.5 else 1
                else:
                    distance_to_target = 1.0 - fraud_prob
                    current_pred = 1 if fraud_prob >= 0.5 else 0
                
                if distance_to_target < adaptation_params['best_distance']:
                    adaptation_params['best_distance'] = distance_to_target
                    adaptation_params['consecutive_no_improvement'] = 0
                    
                    if current_pred == target_pred:
                        for node_idx in perturbations:
                            if torch.any(perturbations[node_idx] != 0):
                                modified_nodes.add(node_idx)
                        break
                else:
                    adaptation_params['consecutive_no_improvement'] += 1
                
                if target_pred == 0:
                    loss = probs[0, 1]
                else:
                    loss = -probs[0, 1]
                
                reg_coef = 0.001
                l2_reg = sum(torch.sum(p**2) for p in perturbations.values())
                loss = loss + reg_coef * l2_reg
                
                loss.backward()
                
                for node_idx in optimizers:
                    scale = 5.0 if node_idx == center_idx else 2.0
                    if perturbations[node_idx].grad is not None:
                        perturbations[node_idx].grad *= scale
                    
                    optimizers[node_idx].step()
                    optimizers[node_idx].zero_grad()
                
                if iteration % 20 == 0 and adaptation_params['consecutive_no_improvement'] > 5:
                    with torch.no_grad():
                        for node_idx in perturbations:
                            noise_scale = 1.0 if node_idx == center_idx else 0.5
                            noise = torch.randn_like(perturbations[node_idx]) * noise_scale
                            perturbations[node_idx].data += noise
                    
                    adaptation_params['consecutive_no_improvement'] = 0
                
                if iteration % 10 == 0 and 0.3 < distance_to_target < 0.7:
                    with torch.no_grad():
                        center_perturbation = perturbations[center_idx]
                        if target_pred == 0:
                            center_perturbation.data -= torch.abs(center_perturbation.data) * 0.5
                        else:
                            center_perturbation.data += torch.abs(center_perturbation.data) * 0.5 + 0.5
            
            for node_idx, perturb in perturbations.items():
                if torch.any(perturb != 0):
                    cf_x_dict[primary_node_type][node_idx] = cf_x_dict[primary_node_type][node_idx] + perturb
                    modified_nodes.add(node_idx)
            
            current_pred, _, current_fraud_prob = self.predict(cf_x_dict, cf_edge_index_dict, center_idx)
        
        time_taken = time.time() - start_time
        success = (current_pred == target_pred)
        
        tonggye = {
            'success': success,
            'original_prediction': orig_pred,
            'final_prediction': current_pred,
            'target_prediction': target_pred,
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
        
        return success, cf_edge_index_dict, cf_x_dict, tonggye

    def _update_adaptation_strategy(self, params, iteration, max_iterations, perturbations, current_fraud_prob, center_node):
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


def load_model_and_data(dataset_name, model_type, device='cuda'):
    model_path = f"models/{dataset_name}/HGT_model.pt"
    
    data_path = f"models/{dataset_name}/HGT_all_hetero_samples_features.pkl"
    
    dataset_info_path = f"models/{dataset_name}/HGT_dataset_info.pkl"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(dataset_info_path):
        raise FileNotFoundError(f"Dataset info file not found: {dataset_info_path}")
    
    with open(dataset_info_path, 'rb') as f:
        dataset_info = pickle.load(f)
    
    with open(data_path, 'rb') as f:
        all_processed_samples = pickle.load(f)
    
    first_sample = next(iter(all_processed_samples.values()))
    primary_node_type = dataset_info['primary_node_type']
    input_dim = first_sample['x_dict'][primary_node_type].size(1)
    
    input_dims = {}
    for node_type in dataset_info['node_types']:
        input_dims[node_type] = input_dim
    
    model = HGTModel(
        node_types=dataset_info['node_types'],
        edge_types=dataset_info['edge_types'],
        input_dims=input_dims,
        hidden_dim=64,
        num_layers=3,
        num_heads=4,
        dropout=0.5,
        num_classes=2,
        primary_node_type=dataset_info['primary_node_type'],
        device=device
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, all_processed_samples, dataset_info