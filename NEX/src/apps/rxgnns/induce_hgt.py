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

# Import HGT components from the provided code
class MultiHeadAttention(nn.Module):
   """Multi-head attention mechanism"""
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
       
       # Linear transformations and reshape
       Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
       K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
       V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
       
       # Attention
       attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
       
       # Concatenate heads
       attention_output = attention_output.transpose(1, 2).contiguous().view(
           batch_size, -1, self.d_model)
       
       # Final linear layer
       output = self.W_o(attention_output)
       return output
   
   def scaled_dot_product_attention(self, Q, K, V, mask=None):
       scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
       
       if mask is not None:
           scores = scores.masked_fill(mask == 0, -1e9)
       
       attention_weights = F.softmax(scores, dim=-1)
       attention_weights = self.dropout(attention_weights)
       
       output = torch.matmul(attention_weights, V)
       return output


class HGTLayer(nn.Module):
   """HGT layer implementation that works with any node/edge types"""
   def __init__(self, node_types, edge_types, in_dim, out_dim, num_heads=4, dropout=0.1):
       super(HGTLayer, self).__init__()
       
       self.node_types = node_types
       self.edge_types = edge_types
       self.in_dim = in_dim
       self.out_dim = out_dim
       self.num_heads = num_heads
       self.dropout = dropout
       self.d_k = out_dim // num_heads
       
       # Ensure dimension can be divided by number of heads
       if out_dim % num_heads != 0:
           out_dim = (out_dim // num_heads) * num_heads
           self.out_dim = out_dim
           self.d_k = out_dim // num_heads
       
       # Heterogeneous attention mechanism: create parameters for each node/edge type combination
       self.k_linears = nn.ModuleDict()
       self.q_linears = nn.ModuleDict()
       self.v_linears = nn.ModuleDict()
       self.a_linears = nn.ModuleDict()
       
       # Create K,V transformations for each source node type
       for src_type in node_types:
           self.k_linears[src_type] = nn.Linear(in_dim, out_dim, bias=False)
           self.v_linears[src_type] = nn.Linear(in_dim, out_dim, bias=False)
           
       # Create Q transformations for each target node type
       for dst_type in node_types:
           self.q_linears[dst_type] = nn.Linear(in_dim, out_dim, bias=False)
           
       # Create attention transformations for each edge type
       for edge_type in edge_types:
           src_type, rel_type, dst_type = edge_type
           edge_key = f"{src_type}_{rel_type}_{dst_type}"
           self.a_linears[edge_key] = nn.Linear(out_dim, num_heads, bias=False)
       
       # Heterogeneous message passing: create message transformations for each target node type
       self.message_linears = nn.ModuleDict()
       for dst_type in node_types:
           self.message_linears[dst_type] = nn.Linear(out_dim, out_dim)
           
       # Target-specific aggregation: create aggregation parameters for each node type
       self.agg_linears = nn.ModuleDict()
       for node_type in node_types:
           self.agg_linears[node_type] = nn.Linear(out_dim, out_dim)
           
       # Residual connections and normalization
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
       # 1. Heterogeneous attention mechanism: compute K, Q, V
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
       
       # 2. Heterogeneous message passing: aggregate messages for each target node type
       new_x_dict = {}
       
       for dst_type in self.node_types:
           if dst_type not in q_dict or q_dict[dst_type].size(0) == 0:
               new_x_dict[dst_type] = torch.zeros((0, self.out_dim), device=next(self.parameters()).device)
               continue
               
           dst_q = q_dict[dst_type]  # [N_dst, out_dim]
           num_dst_nodes = dst_q.size(0)
           
           # Collect all messages pointing to dst_type
           aggregated_messages = torch.zeros_like(dst_q)  # [N_dst, out_dim]
           total_attention_weights = torch.zeros(num_dst_nodes, device=dst_q.device)  # [N_dst]
           
           for edge_type in self.edge_types:
               src_type, rel_type, target_type = edge_type
               
               if (target_type != dst_type or 
                   edge_type not in edge_index_dict or 
                   edge_index_dict[edge_type].size(1) == 0 or
                   src_type not in k_dict or k_dict[src_type].size(0) == 0):
                   continue
               
               edge_index = edge_index_dict[edge_type]  # [2, E]
               src_k = k_dict[src_type]  # [N_src, out_dim]
               src_v = v_dict[src_type]  # [N_src, out_dim]
               
               # Validate edge index validity
               if (edge_index[0].max() >= src_k.size(0) or 
                   edge_index[1].max() >= dst_q.size(0)):
                   continue
               
               # Get node features on edges
               edge_src_k = src_k[edge_index[0]]  # [E, out_dim]
               edge_src_v = src_v[edge_index[0]]  # [E, out_dim]
               edge_dst_q = dst_q[edge_index[1]]  # [E, out_dim]
               
               # Compute attention scores
               # Reshape K, Q to multi-head format
               edge_src_k = edge_src_k.view(-1, self.num_heads, self.d_k)  # [E, H, d_k]
               edge_dst_q = edge_dst_q.view(-1, self.num_heads, self.d_k)  # [E, H, d_k]
               
               # Compute attention scores (scaled dot-product)
               attention_scores = torch.sum(edge_src_k * edge_dst_q, dim=-1)  # [E, H]
               attention_scores = attention_scores / math.sqrt(self.d_k)
               
               # Edge type specific attention modulation
               edge_key = f"{src_type}_{rel_type}_{target_type}"
               if edge_key in self.a_linears:
                   # Use K as input to compute edge type attention weights
                   edge_attention = self.a_linears[edge_key](edge_src_k.view(-1, self.out_dim))  # [E, H]
                   attention_scores = attention_scores + edge_attention
               
               # Softmax normalization (over all incoming edges for each target node)
               attention_weights = F.softmax(attention_scores, dim=-1)  # [E, H]
               
               # Apply attention weights to V
               edge_src_v = edge_src_v.view(-1, self.num_heads, self.d_k)  # [E, H, d_k]
               attended_values = attention_weights.unsqueeze(-1) * edge_src_v  # [E, H, d_k]
               attended_values = attended_values.view(-1, self.out_dim)  # [E, out_dim]
               
               # Aggregate to target nodes
               edge_messages = torch.zeros_like(dst_q)  # [N_dst, out_dim]
               edge_messages.index_add_(0, edge_index[1], attended_values)
               
               # Compute attention weight sum for normalization
               edge_weight_sum = torch.zeros(num_dst_nodes, device=dst_q.device)
               edge_weight_sum.index_add_(0, edge_index[1], attention_weights.sum(dim=-1))
               
               aggregated_messages += edge_messages
               total_attention_weights += edge_weight_sum
           
           # Normalize aggregated messages
           # Avoid division by zero
           total_attention_weights = torch.clamp(total_attention_weights, min=1e-8)
           aggregated_messages = aggregated_messages / total_attention_weights.unsqueeze(-1)
           
           # 3. Target-specific aggregation: apply message transformation
           if dst_type in self.message_linears:
               aggregated_messages = self.message_linears[dst_type](aggregated_messages)
           
           # Apply final aggregation transformation
           if dst_type in self.agg_linears:
               aggregated_messages = self.agg_linears[dst_type](aggregated_messages)
           
           new_x_dict[dst_type] = aggregated_messages
       
       # 4. Residual connections and layer normalization
       output_dict = {}
       for node_type in self.node_types:
           if node_type in x_dict and x_dict[node_type].size(0) > 0:
               # Residual connection
               residual = self.residual_linears[node_type](x_dict[node_type])
               output = new_x_dict[node_type] + residual
               
               # Layer normalization
               output = self.layer_norms[node_type](output)
               
               # Dropout
               output = self.dropout_layer(output)
               
               output_dict[node_type] = output
           else:
               output_dict[node_type] = torch.zeros((0, self.out_dim), device=next(self.parameters()).device)
       
       return output_dict


class HGTModel(nn.Module):
   """HGT model implementation that works with any dataset structure"""
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
       
       # Ensure hidden_dim can be divided by num_heads
       if hidden_dim % num_heads != 0:
           hidden_dim = (hidden_dim // num_heads) * num_heads
           self.hidden_dim = hidden_dim
       
       # Input projection layers: map different input dimensions to unified hidden dimension
       self.input_projections = nn.ModuleDict()
       for node_type in node_types:
           proj = nn.Linear(input_dims[node_type], hidden_dim)
           nn.init.xavier_uniform_(proj.weight, gain=0.1)
           nn.init.constant_(proj.bias, 0)
           self.input_projections[node_type] = proj
       
       # Multi-layer HGT
       self.hgt_layers = nn.ModuleList()
       for i in range(num_layers):
           # All layers use hidden_dim -> hidden_dim
           layer = HGTLayer(
               node_types=node_types,
               edge_types=edge_types,
               in_dim=hidden_dim,
               out_dim=hidden_dim,
               num_heads=num_heads,
               dropout=dropout
           )
           self.hgt_layers.append(layer)
       
       # Classifier: use center node features and global features
       self.classifier = nn.Sequential(
           nn.Linear(hidden_dim * 2, hidden_dim),
           nn.ReLU(),
           nn.Dropout(dropout),
           nn.Linear(hidden_dim, num_classes)
       )
       
       self.to(device)
   
   def forward(self, x_dict, edge_index_dict, center_idx=None):
       # Data preprocessing and input projection
       h_dict = {}
       for node_type in self.node_types:
           if node_type in x_dict and x_dict[node_type].size(0) > 0:
               x = x_dict[node_type]
               
               # Check and handle anomalous values
               if torch.isnan(x).any() or torch.isinf(x).any():
                   print(f"Warning: NaN/Inf in input {node_type}, replacing with zeros")
                   x = torch.zeros_like(x)
               
               # Feature normalization
               x_norm = torch.norm(x, dim=1, keepdim=True)
               x_norm = torch.clamp(x_norm, min=1e-8)
               x = x / x_norm
               
               # Input projection
               h = self.input_projections[node_type](x)
               
               # Check projected features
               if torch.isnan(h).any() or torch.isinf(h).any():
                   print(f"Warning: NaN/Inf after projection in {node_type}")
                   h = torch.zeros_like(h)
               
               h = F.layer_norm(h, h.shape[1:])
               h_dict[node_type] = h
           else:
               h_dict[node_type] = torch.zeros((0, self.hidden_dim), device=self.device)
       
       # Process edge indices
       processed_edge_index_dict = {}
       for edge_type in self.edge_types:
           if edge_type in edge_index_dict and edge_index_dict[edge_type].size(1) > 0:
               edge_index = edge_index_dict[edge_type]
               
               # Validate edge indices
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
       
       # Multi-layer HGT forward propagation
       for layer_idx, layer in enumerate(self.hgt_layers):
           try:
               h_dict = layer(h_dict, processed_edge_index_dict)
               
               # Check each layer output
               for node_type in self.node_types:
                   if (node_type in h_dict and h_dict[node_type].size(0) > 0 and 
                       (torch.isnan(h_dict[node_type]).any() or torch.isinf(h_dict[node_type]).any())):
                       print(f"Warning: NaN/Inf in {node_type} after HGT layer {layer_idx}")
                       h_dict[node_type] = torch.zeros_like(h_dict[node_type])
               
           except Exception as e:
               print(f"Error in HGT layer {layer_idx}: {e}")
               break
       
       # Extract center node features and global features
       center_features = None
       if (center_idx is not None and self.primary_node_type in h_dict and 
           h_dict[self.primary_node_type].size(0) > 0 and center_idx < h_dict[self.primary_node_type].size(0)):
           center_features = h_dict[self.primary_node_type][center_idx]
           
           if torch.isnan(center_features).any() or torch.isinf(center_features).any():
               print("Warning: NaN/Inf in center features")
               center_features = torch.zeros(self.hidden_dim, device=self.device)
       
       # Global features
       if self.primary_node_type in h_dict and h_dict[self.primary_node_type].size(0) > 0:
           global_features = torch.mean(h_dict[self.primary_node_type], dim=0)
           
           if torch.isnan(global_features).any() or torch.isinf(global_features).any():
               print("Warning: NaN/Inf in global features")
               global_features = torch.zeros(self.hidden_dim, device=self.device)
       else:
           global_features = torch.zeros(self.hidden_dim, device=self.device)
       
       # Combine features for classification
       if center_features is not None:
           combined_features = torch.cat([center_features, global_features]).unsqueeze(0)
       else:
           combined_features = torch.cat([global_features, global_features]).unsqueeze(0)
       
       # Final check
       if torch.isnan(combined_features).any() or torch.isinf(combined_features).any():
           print("Warning: NaN/Inf in final features")
           combined_features = torch.zeros_like(combined_features)
       
       # Classification
       output = self.classifier(combined_features)
       return output


class InduCEExplainer:
    """
    Inductive Counterfactual GNN Explainer based on the InduCE approach for HGT
    This implementation includes both edge modifications and feature perturbations
    """
    def __init__(self, model, device='cuda', gnn_layers=3, hidden_dim=64, eta=0.1, gamma=0.4):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # InduCE parameters
        self.gnn_layers = gnn_layers
        self.hidden_dim = hidden_dim
        self.eta = eta  # Entropy regularization parameter
        self.gamma = gamma  # Discount factor for rewards
        
        # Create policy network for counterfactual generation
        self.policy_network = None

    def predict(self, x_dict, edge_index_dict, center_idx):
        """Get prediction for a heterogeneous graph"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_dict, edge_index_dict, center_idx)
            probs = F.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1).item()
            confidence = probs[0, pred].item()
        return pred, confidence, probs[0, 1].item()  # Return prediction, confidence, and fraud probability

    def get_n_hop_neighborhood(self, edge_index_dict, center_node, primary_node_type, n_hops=2):
        """Get n-hop neighborhood of center node in heterogeneous graph"""
        # Build a NetworkX graph from edge indices
        G = nx.Graph()
        
        # Add edges from all edge types that involve the primary node type
        for edge_type, edge_index in edge_index_dict.items():
            src_type, rel_type, dst_type = edge_type
            
            # Only consider edges involving the primary node type
            if src_type == primary_node_type or dst_type == primary_node_type:
                edge_list = edge_index.t().cpu().numpy()
                for src, dst in edge_list:
                    # Map to global node IDs considering node types
                    if src_type == primary_node_type:
                        global_src = src
                    else:
                        global_src = f"{src_type}_{src}"
                    
                    if dst_type == primary_node_type:
                        global_dst = dst
                    else:
                        global_dst = f"{dst_type}_{dst}"
                    
                    G.add_edge(global_src, global_dst)
        
        # Get n-hop neighbors starting from center node
        neighborhood = set([center_node])
        frontier = set([center_node])
        
        for _ in range(n_hops):
            new_frontier = set()
            for node in frontier:
                if node in G:
                    for neighbor in G.neighbors(node):
                        if neighbor not in neighborhood:
                            new_frontier.add(neighbor)
                            neighborhood.add(neighbor)
            frontier = new_frontier
        
        # Filter to only include primary node type nodes
        primary_neighborhood = [n for n in neighborhood if isinstance(n, int)]
        return primary_neighborhood

    def compute_edge_reward(self, x_dict, edge_index_dict, center_idx, primary_node_type, 
                            current_edge, edge_type, is_addition, orig_fraud_prob):
        """
        Compute the reward of perturbing a specific edge in heterogeneous graph
        """
        # Create a modified edge index dict
        new_edge_index_dict = copy.deepcopy(edge_index_dict)
        
        if is_addition:
            # Add edge if it's an addition action
            if not isinstance(current_edge, torch.Tensor):
                current_edge = torch.tensor(current_edge, dtype=torch.long, device=self.device)
            new_edge = current_edge.view(2, 1)
            if edge_type in new_edge_index_dict:
                new_edge_index_dict[edge_type] = torch.cat([new_edge_index_dict[edge_type], new_edge], dim=1)
            else:
                new_edge_index_dict[edge_type] = new_edge
        else:
            # Remove edge if it's a deletion action
            if edge_type in new_edge_index_dict:
                edge_index = new_edge_index_dict[edge_type]
                mask = ~((edge_index[0] == current_edge[0]) & (edge_index[1] == current_edge[1]))
                new_edge_index_dict[edge_type] = edge_index[:, mask]
        
        # Get prediction with modified edge index
        _, _, new_fraud_prob = self.predict(x_dict, new_edge_index_dict, center_idx)
        
        # Compute reward based on change in fraud probability
        reward = -(new_fraud_prob - orig_fraud_prob)  # Negative because we want to decrease fraud probability
        
        return reward

    def generate_counterfactual(self, x_dict, edge_index_dict, center_idx, primary_node_type,
                            max_iterations=200, edge_budget=5, feature_perturbation_range=0.2):
        """
        Generate counterfactual explanation using InduCE approach for heterogeneous graphs
        """
        # Initialize
        self.model.eval()
        start_time = time.time()
        
        # Copy original tensors for modification
        cf_x_dict = {k: v.clone().to(self.device) for k, v in x_dict.items()}
        cf_edge_index_dict = {k: v.clone().to(self.device) for k, v in edge_index_dict.items()}
        
        # Get original prediction - now accept any prediction to attempt counterfactual
        orig_pred, orig_conf, orig_fraud_prob = self.predict(x_dict, edge_index_dict, center_idx)
        
        # Track modifications
        removed_edges = set()
        added_edges = set()
        modified_nodes = set()
        
        # Get the n-hop neighborhood of the center node
        n_hops = 2
        neighborhood = self.get_n_hop_neighborhood(edge_index_dict, center_idx, primary_node_type, n_hops=n_hops)
        
        # Determine target: if originally fraud(1), change to non-fraud(0); if non-fraud(0), change to fraud(1)
        target_pred = 1 - orig_pred
        current_pred = orig_pred
        current_fraud_prob = orig_fraud_prob
        
        # Phase 1: Edge modification (more aggressive)
        edges_modified = 0
        
        # Initialize possible edge perturbations for each edge type
        possible_deletions = {}
        possible_additions = {}
        
        for edge_type in edge_index_dict.keys():
            src_type, rel_type, dst_type = edge_type
            
            if src_type == primary_node_type or dst_type == primary_node_type:
                edge_index = edge_index_dict[edge_type]
                edge_list = edge_index.t().cpu().numpy()
                
                # Possible deletions: existing edges in neighborhood
                possible_deletions[edge_type] = []
                for src, dst in edge_list:
                    if (src_type == primary_node_type and src in neighborhood) or \
                    (dst_type == primary_node_type and dst in neighborhood):
                        possible_deletions[edge_type].append((src, dst))
                
                # Possible additions: new edges within neighborhood (more aggressive)
                possible_additions[edge_type] = []
                if src_type == primary_node_type and dst_type == primary_node_type:
                    for src in neighborhood[:30]:  # Increased from 20
                        for dst in neighborhood[:30]:
                            if src != dst and (src, dst) not in possible_deletions[edge_type] and \
                            (dst, src) not in possible_deletions[edge_type]:
                                possible_additions[edge_type].append((src, dst))
        
        # More aggressive edge modification
        while edges_modified < edge_budget * 2 and current_pred != target_pred:  # Increased budget
            edge_rewards = {}
            
            # Calculate rewards for edge deletions
            for edge_type, edge_list in possible_deletions.items():
                for edge in edge_list:
                    reward = self.compute_edge_reward(
                        cf_x_dict, cf_edge_index_dict, center_idx, primary_node_type,
                        edge, edge_type, False, current_fraud_prob
                    )
                    # Modify reward calculation based on target
                    if target_pred == 0:  # Want to decrease fraud probability
                        edge_rewards[('deletion', edge_type, edge)] = reward
                    else:  # Want to increase fraud probability
                        edge_rewards[('deletion', edge_type, edge)] = -reward
            
            # Calculate rewards for edge additions
            for edge_type, edge_list in possible_additions.items():
                for edge in edge_list[:20]:  # Increased from 10
                    reward = self.compute_edge_reward(
                        cf_x_dict, cf_edge_index_dict, center_idx, primary_node_type,
                        edge, edge_type, True, current_fraud_prob
                    )
                    # Modify reward calculation based on target
                    if target_pred == 0:  # Want to decrease fraud probability
                        edge_rewards[('addition', edge_type, edge)] = reward
                    else:  # Want to increase fraud probability
                        edge_rewards[('addition', edge_type, edge)] = -reward
            
            if not edge_rewards:
                break
                
            best_action, best_edge_type, best_edge = max(edge_rewards.items(), key=lambda x: x[1])[0]
            
            # Apply the best edge perturbation
            if best_action == 'deletion':
                edge_index = cf_edge_index_dict[best_edge_type]
                mask = ~((edge_index[0] == best_edge[0]) & (edge_index[1] == best_edge[1]))
                cf_edge_index_dict[best_edge_type] = edge_index[:, mask]
                
                removed_edges.add((best_edge_type, best_edge))
                possible_deletions[best_edge_type].remove(best_edge)
                
                if best_edge not in possible_additions[best_edge_type]:
                    possible_additions[best_edge_type].append(best_edge)
            else:  # addition
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
        
        # Phase 2: Much more aggressive node feature perturbation
        if current_pred != target_pred and primary_node_type in cf_x_dict:
            perturbations = {}
            primary_x = cf_x_dict[primary_node_type]
            
            valid_neighborhood = [n for n in neighborhood if n < primary_x.size(0)]
            
            for node_idx in valid_neighborhood:
                # Start with larger initial perturbations
                initial_perturbation = torch.randn_like(primary_x[node_idx], device=self.device) * 0.5
                perturbations[node_idx] = initial_perturbation.requires_grad_(True)
            
            # Much more aggressive optimization
            optimizers = {}
            learning_rate = 0.05  # Increased from 0.01
            for node_idx in perturbations:
                node_lr = learning_rate * (3.0 if node_idx == center_idx else 1.5)  # Much higher LR
                optimizers[node_idx] = torch.optim.Adam([perturbations[node_idx]], lr=node_lr)
            
            adaptation_params = {
                'aggressiveness': 1.0,  # Start with maximum aggressiveness
                'stagnation_counter': 0,
                'best_distance': float('inf'),
                'consecutive_no_improvement': 0,
                'exploration_phase': True  # Start in exploration mode
            }
            
            for iteration in range(max_iterations):
                # Apply current perturbations
                perturbed_x_dict = copy.deepcopy(cf_x_dict)
                for node_idx, perturb in perturbations.items():
                    perturbed_x_dict[primary_node_type][node_idx] = cf_x_dict[primary_node_type][node_idx] + perturb
                
                # Forward pass
                self.model.eval()
                outputs = self.model(perturbed_x_dict, cf_edge_index_dict, center_idx)
                probs = F.softmax(outputs, dim=1)
                fraud_prob = probs[0, 1].item()
                
                # Calculate distance to target
                if target_pred == 0:
                    distance_to_target = fraud_prob  # Want to minimize fraud probability
                    current_pred = 0 if fraud_prob < 0.5 else 1
                else:
                    distance_to_target = 1.0 - fraud_prob  # Want to maximize fraud probability
                    current_pred = 1 if fraud_prob >= 0.5 else 0
                
                # Update best results
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
                
                # Calculate loss based on target
                if target_pred == 0:
                    loss = probs[0, 1]  # Minimize fraud probability
                else:
                    loss = -probs[0, 1]  # Maximize fraud probability (minimize negative)
                
                # Much lighter regularization
                reg_coef = 0.001  # Reduced from 0.01-0.1
                l2_reg = sum(torch.sum(p**2) for p in perturbations.values())
                loss = loss + reg_coef * l2_reg
                
                # Backpropagation
                loss.backward()
                
                # Aggressive gradient updates
                for node_idx in optimizers:
                    # Apply strong gradient scaling
                    scale = 5.0 if node_idx == center_idx else 2.0
                    if perturbations[node_idx].grad is not None:
                        perturbations[node_idx].grad *= scale
                    
                    optimizers[node_idx].step()
                    optimizers[node_idx].zero_grad()
                
                # Periodic aggressive exploration
                if iteration % 20 == 0 and adaptation_params['consecutive_no_improvement'] > 5:
                    with torch.no_grad():
                        # Add large random perturbations to break out of local optima
                        for node_idx in perturbations:
                            noise_scale = 1.0 if node_idx == center_idx else 0.5
                            noise = torch.randn_like(perturbations[node_idx]) * noise_scale
                            perturbations[node_idx].data += noise
                    
                    adaptation_params['consecutive_no_improvement'] = 0
                
                # Direct feature manipulation for center node if close to target
                if iteration % 10 == 0 and 0.3 < distance_to_target < 0.7:
                    with torch.no_grad():
                        center_perturbation = perturbations[center_idx]
                        if target_pred == 0:
                            # Add strong negative perturbation to reduce fraud probability
                            center_perturbation.data -= torch.abs(center_perturbation.data) * 0.5
                        else:
                            # Add strong positive perturbation to increase fraud probability
                            center_perturbation.data += torch.abs(center_perturbation.data) * 0.5 + 0.5
            
            # Apply final perturbations
            for node_idx, perturb in perturbations.items():
                if torch.any(perturb != 0):
                    cf_x_dict[primary_node_type][node_idx] = cf_x_dict[primary_node_type][node_idx] + perturb
                    modified_nodes.add(node_idx)
            
            # Get final prediction
            current_pred, _, current_fraud_prob = self.predict(cf_x_dict, cf_edge_index_dict, center_idx)
        
        # Calculate statistics
        time_taken = time.time() - start_time
        success = (current_pred == target_pred)
        
        stats = {
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
        
        return success, cf_edge_index_dict, cf_x_dict, stats

    def _update_adaptation_strategy(self, params, iteration, max_iterations, perturbations, current_fraud_prob, center_node):
        """
        Update adaptive strategy parameters
        """
        # Adjust aggressiveness based on iteration progress and stagnation
        if params['consecutive_no_improvement'] > 10:
            # Multiple consecutive iterations with no improvement, increase aggressiveness
            params['aggressiveness'] = min(1.0, params['aggressiveness'] * 1.2)
            params['stagnation_counter'] += 1
            params['consecutive_no_improvement'] = 0
            
            # Enter exploration phase after multiple stagnations
            if params['stagnation_counter'] >= 3:
                params['exploration_phase'] = True
                
                # Reset some perturbations to break out of local optima
                with torch.no_grad():
                    # Randomly perturb center node features
                    scale = 0.5 * params['aggressiveness']
                    noise = torch.randn_like(perturbations[center_node]) * scale
                    perturbations[center_node].data += noise
                    
                    # Add small perturbations to some other nodes
                    for node_idx in list(perturbations.keys())[:5]:
                        if node_idx != center_node:
                            small_noise = torch.randn_like(perturbations[node_idx]) * scale * 0.5
                            perturbations[node_idx].data += small_noise
        
        # Adjust global exploration/exploitation balance based on iteration progress
        progress = iteration / max_iterations
        if progress > 0.5 and not params['exploration_phase'] and current_fraud_prob > 0.6:
            # If past halfway but fraud probability still high, increase aggressiveness
            params['aggressiveness'] = min(1.0, params['aggressiveness'] + 0.1)
            
        elif progress > 0.8 and current_fraud_prob > 0.55:
            # Near end of iterations but still not reaching goal, significantly increase aggressiveness
            params['aggressiveness'] = 1.0
            params['exploration_phase'] = True


def load_model_and_data(dataset_name, model_type, device='cuda'):
   """Load the trained HGT model and processed heterogeneous data"""
   # Model path - fixed to use HGT naming
   model_path = f"models/{dataset_name}/HGT_model.pt"
   
   # Load all_samples_features which has more complete info
   data_path = f"models/{dataset_name}/HGT_all_hetero_samples_features.pkl"
   
   # Load dataset info
   dataset_info_path = f"models/{dataset_name}/HGT_dataset_info.pkl"
   
   # Check if files exist
   if not os.path.exists(model_path):
       raise FileNotFoundError(f"Model file not found: {model_path}")
   if not os.path.exists(data_path):
       raise FileNotFoundError(f"Data file not found: {data_path}")
   if not os.path.exists(dataset_info_path):
       raise FileNotFoundError(f"Dataset info file not found: {dataset_info_path}")
   
   # Load dataset info
   with open(dataset_info_path, 'rb') as f:
       dataset_info = pickle.load(f)
   
   # Load processed data
   with open(data_path, 'rb') as f:
       all_processed_samples = pickle.load(f)
   
   # Get input dimension from the first sample
   first_sample = next(iter(all_processed_samples.values()))
   primary_node_type = dataset_info['primary_node_type']
   input_dim = first_sample['x_dict'][primary_node_type].size(1)
   
   # Create input dimensions dict
   input_dims = {}
   for node_type in dataset_info['node_types']:
       input_dims[node_type] = input_dim
   
   # Initialize HGT model architecture
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
   
   # Load trained weights
   model.load_state_dict(torch.load(model_path, map_location=device))
   model.eval()
   
   return model, all_processed_samples, dataset_info
