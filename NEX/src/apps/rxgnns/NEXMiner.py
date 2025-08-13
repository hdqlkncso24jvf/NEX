import os
import pickle
import random
import threading
import time
from collections import defaultdict, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import argparse
import math

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available, using fallback ranking")

from CFE import CounterfactualExplainer
from graph_matcher import Matcher, Pattern, Node, Graph, AttributePredicate, \
  AttributeComparisonPredicate, RxGNNs

MODEL_DIR = "models"
RULES_DIR = "rules"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RULES_DIR, exist_ok=True)

from torch_geometric.nn import GCNConv, GATConv, GINConv, global_mean_pool, global_max_pool
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from sklearn.metrics import precision_score, recall_score, f1_score

class FeatureExtractor:
  def __init__(self, embedding_dim=16, device='cuda'):
      self.embedding_dim = embedding_dim
      self.device = device
      self.attribute_embeddings = {}
      self.attribute_values = defaultdict(set)
      self.embedding_tables = {}
      
  def fit(self, graphs):
      for graph in graphs:
          for node_id, node in graph.nodes.items():
              if node.label not in self.attribute_values['node_label']:
                  self.attribute_values['node_label'].add(node.label)
                  
              for attr_name, attr_value in node.attributes.items():
                  if isinstance(attr_value, (int, float)):
                      self.attribute_values[attr_name].add(attr_value)
                  elif isinstance(attr_value, str):
                      self.attribute_values[attr_name].add(attr_value)
                  else:
                      self.attribute_values[attr_name].add(str(attr_value))
      
      for attr_name, values in self.attribute_values.items():
          num_values = len(values) + 1
          self.embedding_tables[attr_name] = nn.Embedding(
              num_embeddings=num_values,
              embedding_dim=self.embedding_dim
          ).to(self.device)
          
          self.attribute_embeddings[attr_name] = {
              value: idx + 1 for idx, value in enumerate(values)
          }
      
  def transform_graph(self, graph, center_id=None, label=None):
      node_features = []
      node_id_to_idx = {}
      center_idx = None
      
      for idx, (node_id, node) in enumerate(graph.nodes.items()):
          node_id_to_idx[node_id] = idx
          
          is_center = (node_id == center_id)
          if is_center:
              center_idx = idx
          
          features = self.extract_node_features(node)
          
          if is_center:
              center_indicator = torch.ones(self.embedding_dim, device=self.device)
          else:
              center_indicator = torch.zeros(self.embedding_dim, device=self.device)
          
          features = torch.cat([features, center_indicator])
          node_features.append(features)
      
      x = torch.stack(node_features)
      
      edge_list = []
      for source_id, target_id in graph.edges:
          source_idx = node_id_to_idx[source_id]
          target_idx = node_id_to_idx[target_id]
          edge_list.append([source_idx, target_idx])
          edge_list.append([target_idx, source_idx])
      
      if edge_list:
          edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t().contiguous()
      else:
          edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
      
      return x, edge_index, center_idx
  
  def extract_node_features(self, node):
      embeddings = []
      
      label_idx = 0
      if 'node_label' in self.attribute_embeddings and node.label in self.attribute_embeddings['node_label']:
          label_idx = self.attribute_embeddings['node_label'][node.label]
      label_embedding = self.embedding_tables['node_label'](
          torch.tensor(label_idx, device=self.device)
      )
      embeddings.append(label_embedding)
      
      all_attr_names = list(self.attribute_embeddings.keys())
      for attr_name in all_attr_names:
          if attr_name == 'node_label':
              continue
              
          attr_idx = 0
          if attr_name in node.attributes:
              attr_value = node.attributes[attr_name]
              if isinstance(attr_value, (int, float, str)):
                  if attr_value in self.attribute_embeddings[attr_name]:
                      attr_idx = self.attribute_embeddings[attr_name][attr_value]
              else:
                  str_value = str(attr_value)
                  if str_value in self.attribute_embeddings[attr_name]:
                      attr_idx = self.attribute_embeddings[attr_name][str_value]
          
          attr_embedding = self.embedding_tables[attr_name](
              torch.tensor(attr_idx, device=self.device)
          )
          embeddings.append(attr_embedding)
      
      if embeddings:
          return torch.cat(embeddings)
      else:
          return torch.zeros(self.embedding_dim, device=self.device)


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
          self.convs.append(GATConv(
              input_dim, hidden_dim,
              heads=1,
              concat=False,
              dropout=0.0,
              negative_slope=0.2,
              add_self_loops=True
          ))
      elif gnn_type == 'GIN':
          self.convs.append(GINConv(
              nn=nn.Linear(input_dim, hidden_dim),
              train_eps=False,
              aggr='mean'
          ))
      else:
          raise ValueError(f"Unbekannter GNN-Typ: {gnn_type}")
      
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
          raise ValueError(f"Unbekannte Readout-Funktion: {self.readout}")
      
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
      
      return x, readout_features, center_features


class HeteroFeatureExtractor:
  def __init__(self, embedding_dim=16, edge_dim=8, device='cuda'):
      self.embedding_dim = embedding_dim
      self.edge_dim = edge_dim
      self.device = device
      self.attribute_embeddings = {}
      self.attribute_values = defaultdict(set)
      self.embedding_tables = {}
      self.node_types = []
      self.edge_types = []
      self.primary_node_type = None
      self.label_key = None
      
  def infer_dataset_structure(self, hetero_data_list):
      sample_hetero_data_list = hetero_data_list[:min(10, len(hetero_data_list))]
      
      node_type_counts = defaultdict(int)
      edge_type_sets = set()
      label_keys = set()
      
      for center_id, hetero_data in sample_hetero_data_list:
          if 'node_types' in hetero_data:
              for node_type, nodes in hetero_data['node_types'].items():
                  if nodes:
                      node_type_counts[node_type] += len(nodes)
          
          if 'edge_labels' in hetero_data:
              for edge_id, edge_name in hetero_data['edge_labels'].items():
                  edge_type_sets.add(edge_name.lower())
          
          if 'node_labels' in hetero_data:
              for label_key in hetero_data['node_labels'].keys():
                  label_keys.add(label_key)
      
      self.node_types = sorted(node_type_counts.keys())
      
      if label_keys:
          self.label_key = list(label_keys)[0]
          self.primary_node_type = self.label_key
      else:
          self.primary_node_type = min(node_type_counts.keys(), key=node_type_counts.get)
          self.label_key = self.primary_node_type
      
      self.edge_types = []
      edge_type_list = sorted(edge_type_sets)
      
      for edge_type_name in edge_type_list:
          for target_type in self.node_types:
              if target_type != self.primary_node_type:
                  self.edge_types.append((self.primary_node_type, edge_type_name, target_type))
      
      if not self.edge_types:
          for target_type in self.node_types:
              if target_type != self.primary_node_type:
                  self.edge_types.append((self.primary_node_type, 'connects_to', target_type))
      
      return self.node_types, self.edge_types, self.primary_node_type, self.label_key

  def fit(self, hetero_data_list):
      self.infer_dataset_structure(hetero_data_list)
      
      all_graphs = []
      for center_id, hetero_data in hetero_data_list:
          graph = self.reconstruct_graph_from_hetero_data(hetero_data)
          all_graphs.append(graph)
      
      for graph in all_graphs:
          for node_id, node in graph.nodes.items():
              if node.label not in self.attribute_values['node_label']:
                  self.attribute_values['node_label'].add(node.label)
                  
              for attr_name, attr_value in node.attributes.items():
                  if isinstance(attr_value, (int, float)):
                      self.attribute_values[attr_name].add(attr_value)
                  elif isinstance(attr_value, str):
                      self.attribute_values[attr_name].add(attr_value)
                  else:
                      self.attribute_values[attr_name].add(str(attr_value))
      
      for attr_name, values in self.attribute_values.items():
          num_values = len(values) + 1
          embedding_table = nn.Embedding(
              num_embeddings=num_values,
              embedding_dim=self.embedding_dim
          ).to(self.device)
          
          nn.init.xavier_uniform_(embedding_table.weight)
          with torch.no_grad():
              embedding_table.weight.clamp_(-2.0, 2.0)
          
          self.embedding_tables[attr_name] = embedding_table
          
          self.attribute_embeddings[attr_name] = {
              value: idx + 1 for idx, value in enumerate(values)
          }

  def reconstruct_graph_from_hetero_data(self, hetero_data):
      graph = Graph()
      
      if 'node_types' not in hetero_data:
          return graph
      
      for node_type, node_list in hetero_data['node_types'].items():
          node_label = self.node_types.index(node_type) if node_type in self.node_types else 0
          
          for node_id in node_list:
              attrs = {}
              
              if ('node_labels' in hetero_data and 
                  node_type in hetero_data['node_labels'] and 
                  node_id in hetero_data['node_labels'][node_type]):
                  attrs['gnn_prediction'] = hetero_data['node_labels'][node_type][node_id]
              
              graph.add_node(Node(node_id, node_label, attrs))
      
      if 'edges_with_labels' in hetero_data:
          edge_label_to_type = {}
          if 'edge_labels' in hetero_data:
              edge_label_to_type = hetero_data['edge_labels']
          
          for edge_data in hetero_data['edges_with_labels']:
              if len(edge_data) >= 2:
                  src_id, tgt_id = edge_data[0], edge_data[1]
                  
                  if len(edge_data) >= 3:
                      edge_label = edge_data[2]
                      edge_type = edge_label_to_type.get(edge_label, 'unknown')
                      
                      if src_id in graph.nodes and graph.nodes[src_id].label == self.node_types.index(self.primary_node_type):
                          graph.nodes[src_id].attributes['edge_type'] = edge_type
                      elif tgt_id in graph.nodes and graph.nodes[tgt_id].label == self.node_types.index(self.primary_node_type):
                          graph.nodes[tgt_id].attributes['edge_type'] = edge_type
                  
                  graph.add_edge(src_id, tgt_id)
      
      return graph

  def transform_hetero_data(self, hetero_data, center_id=None, label=None):
      data = HeteroData()
      
      feature_dim = self.embedding_dim * len(self.attribute_embeddings) + self.embedding_dim
      
      node_features_dict = {}
      center_idx_dict = {}
      
      for node_type in self.node_types:
          if node_type not in hetero_data['node_types']:
              node_features_dict[node_type] = torch.zeros((0, feature_dim), device=self.device)
              center_idx_dict[node_type] = None
              continue
          
          node_list = hetero_data['node_types'][node_type]
          node_features = []
          center_idx_for_type = None
          
          if node_list:
              for idx, node_id in enumerate(node_list):
                  node_label = self.node_types.index(node_type)
                  node_attrs = {}
                  
                  if ('node_labels' in hetero_data and 
                      node_type in hetero_data['node_labels'] and 
                      node_id in hetero_data['node_labels'][node_type]):
                      node_attrs['gnn_prediction'] = hetero_data['node_labels'][node_type][node_id]
                  
                  if 'edges_with_labels' in hetero_data and 'edge_labels' in hetero_data:
                      for edge_data in hetero_data['edges_with_labels']:
                          if len(edge_data) >= 3 and (edge_data[0] == node_id or edge_data[1] == node_id):
                              edge_label = edge_data[2]
                              edge_type = hetero_data['edge_labels'].get(edge_label, 'unknown')
                              node_attrs['edge_type'] = edge_type
                              break
                  
                  node = Node(node_id, node_label, node_attrs)
                  
                  is_center = (node_id == center_id)
                  if is_center:
                      center_idx_for_type = idx
                  
                  features = self.extract_node_features(node)
                  
                  if is_center:
                      center_indicator = torch.ones(self.embedding_dim, device=self.device)
                  else:
                      center_indicator = torch.zeros(self.embedding_dim, device=self.device)
                  
                  features = torch.cat([features, center_indicator])
                  node_features.append(features)
          
          if node_features:
              node_features_dict[node_type] = torch.stack(node_features)
          else:
              dummy_features = torch.zeros(feature_dim, device=self.device)
              node_features_dict[node_type] = dummy_features.unsqueeze(0)
              if center_id and node_type == self.primary_node_type:
                  center_idx_for_type = 0
          
          center_idx_dict[node_type] = center_idx_for_type
      
      for node_type in self.node_types:
          data[node_type].x = node_features_dict[node_type]
      
      id_to_idx_mappings = {}
      for node_type in self.node_types:
          if node_type in hetero_data['node_types']:
              node_list = hetero_data['node_types'][node_type]
              id_to_idx_mappings[node_type] = {node_id: idx for idx, node_id in enumerate(node_list)}
          else:
              id_to_idx_mappings[node_type] = {}
      
      edge_type_to_name = {}
      if 'edge_labels' in hetero_data:
          edge_type_to_name = {v.lower(): v.lower() for v in hetero_data['edge_labels'].values()}
      
      if not edge_type_to_name:
          edge_type_to_name = {'connects_to': 'connects_to'}
      
      for edge_type_tuple in self.edge_types:
          src_type, relation, tgt_type = edge_type_tuple
          data[src_type, relation, tgt_type].edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
          data[src_type, relation, tgt_type].edge_attr = torch.zeros((0, self.edge_dim), device=self.device)
      
      if 'edges_with_labels' in hetero_data and hetero_data['edges_with_labels']:
          edges_by_type = defaultdict(list)
          edge_attrs_by_type = defaultdict(list)
          
          for edge_data in hetero_data['edges_with_labels']:
              if len(edge_data) >= 2:
                  src_id, tgt_id = edge_data[0], edge_data[1]
                  
                  edge_type_name = 'connects_to'
                  if len(edge_data) >= 3 and 'edge_labels' in hetero_data:
                      edge_label = edge_data[2]
                      edge_type_name = hetero_data['edge_labels'].get(edge_label, 'connects_to').lower()
                  
                  src_type = None
                  tgt_type = None
                  
                  for node_type, node_list in hetero_data['node_types'].items():
                      if src_id in node_list:
                          src_type = node_type
                      if tgt_id in node_list:
                          tgt_type = node_type
                  
                  if src_type and tgt_type and src_type in id_to_idx_mappings and tgt_type in id_to_idx_mappings:
                      if src_id in id_to_idx_mappings[src_type] and tgt_id in id_to_idx_mappings[tgt_type]:
                          src_idx = id_to_idx_mappings[src_type][src_id]
                          tgt_idx = id_to_idx_mappings[tgt_type][tgt_id]
                          
                          edge_key = None
                          for edge_tuple in self.edge_types:
                              if ((edge_tuple[0] == src_type and edge_tuple[2] == tgt_type) or
                                  (edge_tuple[0] == tgt_type and edge_tuple[2] == src_type)):
                                  if edge_type_name in edge_tuple[1] or edge_tuple[1] == 'connects_to':
                                      edge_key = edge_tuple
                                      break
                          
                          if not edge_key:
                              edge_key = (src_type, edge_type_name, tgt_type)
                          
                          if edge_key[0] == src_type:
                              edges_by_type[edge_key].append([src_idx, tgt_idx])
                          else:
                              edges_by_type[edge_key].append([tgt_idx, src_idx])
                          
                          edge_attr = torch.randn(self.edge_dim, device=self.device)
                          edge_attrs_by_type[edge_key].append(edge_attr)
          
          for edge_key, edge_list in edges_by_type.items():
              if edge_list:
                  edge_tensor = torch.tensor(edge_list, dtype=torch.long, device=self.device).t().contiguous()
                  edge_attr_tensor = torch.stack(edge_attrs_by_type[edge_key])
                  
                  data[edge_key].edge_index = edge_tensor
                  data[edge_key].edge_attr = edge_attr_tensor
      
      main_center_idx = center_idx_dict.get(self.primary_node_type, 0)
      
      return data, main_center_idx

def extract_node_features(self, node):
      embeddings = []
      
      label_idx = 0
      if 'node_label' in self.attribute_embeddings and node.label in self.attribute_embeddings['node_label']:
          label_idx = self.attribute_embeddings['node_label'][node.label]
      label_embedding = self.embedding_tables['node_label'](
          torch.tensor(label_idx, device=self.device)
      )
      embeddings.append(label_embedding)
      
      all_attr_names = list(self.attribute_embeddings.keys())
      for attr_name in all_attr_names:
          if attr_name == 'node_label':
              continue
              
          attr_idx = 0
          if attr_name in node.attributes:
              attr_value = node.attributes[attr_name]
              if isinstance(attr_value, (int, float, str)):
                  if attr_value in self.attribute_embeddings[attr_name]:
                      attr_idx = self.attribute_embeddings[attr_name][attr_value]
              else:
                  str_value = str(attr_value)
                  if str_value in self.attribute_embeddings[attr_name]:
                      attr_idx = self.attribute_embeddings[attr_name][str_value]
          
          attr_embedding = self.embedding_tables[attr_name](
              torch.tensor(attr_idx, device=self.device)
          )
          embeddings.append(attr_embedding)
      
      if embeddings:
          return torch.cat(embeddings)
      else:
          return torch.zeros(self.embedding_dim, device=self.device)


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
      scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
      
      if mask is not None:
          scores = scores.masked_fill(mask == 0, -1e9)
      
      attention_weights = F.softmax(scores, dim=-1)
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
          
          aggregated_messages = torch.zeros_like(dst_q)
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
              
              aggregated_messages += edge_messages
              total_attention_weights += edge_weight_sum
          
          total_attention_weights = torch.clamp(total_attention_weights, min=1e-8)
          aggregated_messages = aggregated_messages / total_attention_weights.unsqueeze(-1)
          
          if dst_type in self.message_linears:
              aggregated_messages = self.message_linears[dst_type](aggregated_messages)
          
          if dst_type in self.agg_linears:
              aggregated_messages = self.agg_linears[dst_type](aggregated_messages)
          
          new_x_dict[dst_type] = aggregated_messages
      
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
                  x = torch.zeros_like(x)
              
              x_norm = torch.norm(x, dim=1, keepdim=True)
              x_norm = torch.clamp(x_norm, min=1e-8)
              x = x / x_norm
              
              h = self.input_projections[node_type](x)
              
              if torch.isnan(h).any() or torch.isinf(h).any():
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
                      h_dict[node_type] = torch.zeros_like(h_dict[node_type])
              
          except Exception as e:
              break
      
      center_features = None
      if (center_idx is not None and self.primary_node_type in h_dict and 
          h_dict[self.primary_node_type].size(0) > 0 and center_idx < h_dict[self.primary_node_type].size(0)):
          center_features = h_dict[self.primary_node_type][center_idx]
          
          if torch.isnan(center_features).any() or torch.isinf(center_features).any():
              center_features = torch.zeros(self.hidden_dim, device=self.device)
      
      if self.primary_node_type in h_dict and h_dict[self.primary_node_type].size(0) > 0:
          global_features = torch.mean(h_dict[self.primary_node_type], dim=0)
          
          if torch.isnan(global_features).any() or torch.isinf(global_features).any():
              global_features = torch.zeros(self.hidden_dim, device=self.device)
      else:
          global_features = torch.zeros(self.hidden_dim, device=self.device)
      
      if center_features is not None:
          combined_features = torch.cat([center_features, global_features]).unsqueeze(0)
      else:
          combined_features = torch.cat([global_features, global_features]).unsqueeze(0)
      
      if torch.isnan(combined_features).any() or torch.isinf(combined_features).any():
          combined_features = torch.zeros_like(combined_features)
      
      output = self.classifier(combined_features)
      return output, global_features, center_features


class DQN(nn.Module):
  def __init__(self, state_dim, action_dim, gnn_embedding_dim=64):
      super(DQN, self).__init__()
      self.gnn_embedding_dim = gnn_embedding_dim
      
      total_input_dim = state_dim + gnn_embedding_dim
      
      self.fc1 = nn.Linear(total_input_dim, 128)
      self.fc2 = nn.Linear(128, 64)
      self.fc3 = nn.Linear(64, action_dim + 1)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      return self.fc3(x)


class GNNModelLoader:
   def __init__(self, dataset_name, model_type='GCN', device='cuda'):
       self.dataset_name = dataset_name
       self.model_type = model_type
       self.device = device
       self.feature_extractor = None
       self.gnn_model = None
       self.hetero_feature_extractor = None
       self.hgt_model = None
       
   def load_model(self):
       if self.model_type == 'HGT':
           return self._load_hgt_model()
       else:
           return self._load_standard_gnn_model()
   
   def _load_standard_gnn_model(self):
       model_path = f"models/{self.dataset_name}/{self.model_type}_model.pt"
       data_path = f"models/{self.dataset_name}/{self.model_type}_processed_data.pkl"
       
       if not os.path.exists(model_path) or not os.path.exists(data_path):
           return False
       
       try:
           with open(data_path, 'rb') as f:
               processed_graphs = pickle.load(f)
           
           first_graph = next(iter(processed_graphs.values()))
           input_dim = first_graph['features'].size(1)
           
           self.gnn_model = GNNModel(
               input_dim=input_dim,
               hidden_dim=64,
               num_layers=3,
               dropout=0.5,
               gnn_type=self.model_type,
               readout='mean',
               num_classes=2,
               device=self.device
           )
           
           self.gnn_model.load_state_dict(torch.load(model_path, map_location=self.device))
           self.gnn_model.eval()
           
           return True
           
       except Exception as e:
           return False
   
   def _load_hgt_model(self):
       model_path = f"models/{self.dataset_name}/HGT_model.pt"
       info_path = f"models/{self.dataset_name}/HGT_dataset_info.pkl"
       
       if not os.path.exists(model_path) or not os.path.exists(info_path):
           return False
       
       try:
           with open(info_path, 'rb') as f:
               dataset_info = pickle.load(f)
           
           self.hetero_feature_extractor = HeteroFeatureExtractor(
               embedding_dim=16, edge_dim=8, device=self.device
           )
           self.hetero_feature_extractor.node_types = dataset_info['node_types']
           self.hetero_feature_extractor.edge_types = dataset_info['edge_types']
           self.hetero_feature_extractor.primary_node_type = dataset_info['primary_node_type']
           self.hetero_feature_extractor.label_key = dataset_info['label_key']
           
           input_dim = 16 * 5 + 16
           input_dims = {node_type: input_dim for node_type in dataset_info['node_types']}
           
           self.hgt_model = HGTModel(
               node_types=dataset_info['node_types'],
               edge_types=dataset_info['edge_types'],
               input_dims=input_dims,
               hidden_dim=64,
               num_layers=3,
               num_heads=4,
               dropout=0.5,
               num_classes=2,
               primary_node_type=dataset_info['primary_node_type'],
               device=self.device
           )
           
           self.hgt_model.load_state_dict(torch.load(model_path, map_location=self.device))
           self.hgt_model.eval()
           
           return True
           
       except Exception as e:
           return False
   
   def initialize_feature_extractor_from_graph(self, graph):
       if self.model_type == 'HGT':
           return self._initialize_hetero_extractor_from_graph(graph)
       else:
           return self._initialize_standard_extractor_from_graph(graph)
   
   def _initialize_standard_extractor_from_graph(self, graph):
       self.feature_extractor = FeatureExtractor(embedding_dim=16, device=self.device)
       
       all_attributes = defaultdict(set)
       
       for node_id, node in graph.nodes.items():
           all_attributes['node_label'].add(node.label)
           
           if hasattr(node, 'attributes') and node.attributes:
               for attr_name, attr_value in node.attributes.items():
                   if attr_value is not None:
                       if isinstance(attr_value, bool):
                           all_attributes[attr_name].add(True)
                           all_attributes[attr_name].add(False)
                       else:
                           all_attributes[attr_name].add(attr_value)
       
       self.feature_extractor.attribute_values = dict(all_attributes)
       self.feature_extractor.attribute_embeddings = {}
       self.feature_extractor.embedding_tables = {}
       
       for attr_name, values in all_attributes.items():
           num_values = len(values) + 1
           
           embedding_table = nn.Embedding(
               num_embeddings=num_values,
               embedding_dim=16
           ).to(self.device)
           
           nn.init.xavier_uniform_(embedding_table.weight)
           
           self.feature_extractor.embedding_tables[attr_name] = embedding_table
           
           self.feature_extractor.attribute_embeddings[attr_name] = {
               value: idx + 1 for idx, value in enumerate(values)
           }
       
       expected_input_dim = 16 * len(all_attributes)
       
       return expected_input_dim
   
   def _initialize_hetero_extractor_from_graph(self, graph):
       return 224
   
   def extract_pattern_embedding(self, pattern, data_graph=None):
       if self.feature_extractor is None and data_graph is not None:
           expected_dim = self.initialize_feature_extractor_from_graph(data_graph)
       
       if self.model_type == 'HGT':
           return self._extract_hgt_embedding(pattern)
       else:
           return self._extract_standard_gnn_embedding(pattern)
   
   def _extract_standard_gnn_embedding(self, pattern):
       try:
           graph = self._pattern_to_graph(pattern)
           
           with torch.no_grad():
               x, edge_index, center_idx = self.feature_extractor.transform_graph(
                   graph, center_id=pattern.pivot_id
               )
               
               if self.gnn_model is not None:
                   try:
                       model_input_dim = self.gnn_model.input_dim
                       current_dim = x.size(1)
                       
                       if current_dim != model_input_dim:
                           if current_dim < model_input_dim:
                               padding = torch.zeros(x.size(0), model_input_dim - current_dim, device=self.device)
                               x = torch.cat([x, padding], dim=1)
                           else:
                               x = x[:, :model_input_dim]
                       
                       _, readout_features, center_features = self.gnn_model(x, edge_index, None, center_idx)
                       if readout_features is not None:
                           embedding = readout_features.squeeze(0)
                           if embedding.size(0) > 64:
                               return embedding[:64]
                           elif embedding.size(0) < 64:
                               padding = torch.zeros(64 - embedding.size(0), device=self.device)
                               return torch.cat([embedding, padding])
                           else:
                               return embedding
                   except Exception as e:
                       pass
               
               if x.size(0) > 0:
                   pooled_features = torch.mean(x, dim=0)
                   if pooled_features.size(0) > 64:
                       return pooled_features[:64].to(self.device)
                   elif pooled_features.size(0) < 64:
                       padding = torch.zeros(64 - pooled_features.size(0), device=self.device)
                       return torch.cat([pooled_features, padding]).to(self.device)
                   else:
                       return pooled_features.to(self.device)
               else:
                   return torch.zeros(64, device=self.device)
                   
       except Exception as e:
           return torch.zeros(64, device=self.device)
   
   def _extract_hgt_embedding(self, pattern):
       try:
           hetero_data = self._pattern_to_hetero_data(pattern)
           
           with torch.no_grad():
               data, center_idx = self.hetero_feature_extractor.transform_hetero_data(
                   hetero_data, center_id=pattern.pivot_id
               )
               
               if self.hgt_model is not None:
                   try:
                       _, global_features, center_features = self.hgt_model(
                           data.x_dict, data.edge_index_dict, center_idx
                       )
                       if global_features is not None:
                           if global_features.size(0) > 64:
                               return global_features[:64]
                           elif global_features.size(0) < 64:
                               padding = torch.zeros(64 - global_features.size(0), device=self.device)
                               return torch.cat([global_features, padding])
                           else:
                               return global_features
                   except Exception as e:
                       pass
               
               primary_type = self.hetero_feature_extractor.primary_node_type
               if primary_type in data.x_dict and data.x_dict[primary_type].size(0) > 0:
                   pooled_features = torch.mean(data.x_dict[primary_type], dim=0)
                   if pooled_features.size(0) > 64:
                       return pooled_features[:64].to(self.device)
                   elif pooled_features.size(0) < 64:
                       padding = torch.zeros(64 - pooled_features.size(0), device=self.device)
                       return torch.cat([pooled_features, padding]).to(self.device)
                   else:
                       return pooled_features.to(self.device)
               else:
                   return torch.zeros(64, device=self.device)
                   
       except Exception as e:
           return torch.zeros(64, device=self.device)
   
   def _pattern_to_graph(self, pattern):
       graph = Graph()
       
       known_attributes = set()
       if self.feature_extractor and hasattr(self.feature_extractor, 'attribute_values'):
           known_attributes = set(self.feature_extractor.attribute_values.keys())
       
       node_type_prefixes = {0: 'u', 1: 'l', 2: 'c', 3: 's', 4: 'j'}
       
       for node_id, node in pattern.graph.nodes.items():
           attrs = {}
           if hasattr(node, 'attributes') and node.attributes:
               attrs.update(node.attributes)
           
           if node_id == pattern.pivot_id:
               attrs['is_pivot'] = True
               if 'gnn_prediction' not in attrs:
                   attrs['gnn_prediction'] = True
           
           for attr_name in known_attributes:
               if attr_name not in attrs:
                   attrs[attr_name] = self._get_default_value_for_attribute(attr_name, node.label)
           
           graph.add_node(Node(node_id, node.label, attrs))
       
       for src_id, tgt_id in pattern.graph.edges:
           graph.add_edge(src_id, tgt_id)
       
       return graph
   
   def _get_default_value_for_attribute(self, attr_name, node_label):
       if (self.feature_extractor and 
           hasattr(self.feature_extractor, 'attribute_values') and
           attr_name in self.feature_extractor.attribute_values):
           
           values = self.feature_extractor.attribute_values[attr_name]
           
           if not values:
               return None
           
           values_list = list(values)
           
           if all(isinstance(v, bool) for v in values_list):
               return False
           
           elif all(isinstance(v, (int, float)) for v in values_list):
               if attr_name.endswith('_level'):
                   return 3
               else:
                   return int(sum(values_list) / len(values_list))
           
           elif all(isinstance(v, str) for v in values_list):
               return sorted(values_list)[0]
           
           else:
               return values_list[0]
       
       if attr_name == 'node_label':
           return node_label
       elif attr_name == 'gnn_prediction':
           return False
       elif attr_name.endswith('_level'):
           return 3
       elif 'bool' in attr_name.lower() or attr_name in ['married_single', 'car_ownership']:
           return False
       else:
           return 0
   
   def _pattern_to_hetero_data(self, pattern):
       hetero_data = {
           'node_types': defaultdict(list),
           'node_labels': defaultdict(dict),
           'edges_with_labels': [],
           'edge_labels': {}
       }
       
       for node_id, node in pattern.graph.nodes.items():
           if hasattr(self.hetero_feature_extractor, 'node_types') and self.hetero_feature_extractor.node_types:
               if node.label < len(self.hetero_feature_extractor.node_types):
                   node_type = self.hetero_feature_extractor.node_types[node.label]
               else:
                   node_type = self.hetero_feature_extractor.node_types[0]
           else:
               type_mapping = {0: 'user', 1: 'loan', 2: 'city', 3: 'state', 4: 'job'}
               node_type = type_mapping.get(node.label, 'unknown')
           
           hetero_data['node_types'][node_type].append(node_id)
           hetero_data['node_labels'][node_type][node_id] = (node_id == pattern.pivot_id)
       
       edge_counter = 0
       for src_id, tgt_id in pattern.graph.edges:
           edge_label = f"edge_{edge_counter}"
           hetero_data['edges_with_labels'].append([src_id, tgt_id, edge_label])
           hetero_data['edge_labels'][edge_label] = 'connects_to'
           edge_counter += 1
       
       return hetero_data

class PatternComposer:
   def __init__(self, matcher, motifs, gnn_model_loader=None, model_path="models/pattern_composer.pt"):
       self.matcher = matcher
       self.motifs = motifs
       self.model_path = model_path
       self.gnn_model_loader = gnn_model_loader

       self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

       self.basic_state_dim = 4
       self.gnn_embedding_dim = 64
       self.state_dim = self.basic_state_dim + self.gnn_embedding_dim

       self.action_dim = 10

       self.dqn = DQN(self.basic_state_dim, self.action_dim, self.gnn_embedding_dim).to(self.device)
       
       if os.path.exists(self.model_path):
           self.dqn.load_state_dict(torch.load(self.model_path, map_location=self.device))
       else:
           self.train_dqn(max_episodes=100)

   def get_state(self, pattern):
       num_nodes = len(pattern.graph.nodes)
       num_edges = len(pattern.graph.edges)
       support = self.matcher.get_pattern_support(pattern)
       confidence = self.matcher.get_pattern_confidence(pattern)

       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

       basic_state = torch.tensor([num_nodes, num_edges, support, confidence], 
                               dtype=torch.float32, device=device)

       if self.gnn_model_loader is not None:
           try:
               gnn_embedding = self.gnn_model_loader.extract_pattern_embedding(pattern)
               
               if gnn_embedding.device != device:
                   gnn_embedding = gnn_embedding.to(device)
               
               if gnn_embedding.size(0) != self.gnn_embedding_dim:
                   if gnn_embedding.size(0) > self.gnn_embedding_dim:
                       gnn_embedding = gnn_embedding[:self.gnn_embedding_dim]
                   else:
                       padding = torch.zeros(self.gnn_embedding_dim - gnn_embedding.size(0), 
                                           device=device)
                       gnn_embedding = torch.cat([gnn_embedding, padding])
           except Exception as e:
               gnn_embedding = torch.zeros(self.gnn_embedding_dim, device=device)
       else:
           gnn_embedding = torch.zeros(self.gnn_embedding_dim, device=device)

       full_state = torch.cat([basic_state, gnn_embedding])
       
       return full_state

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

           try:
               state = self.get_state(pattern1)
           except Exception as e:
               continue

           if random.random() < epsilon:
               if random.random() < 0.3:
                   action_idx = self.action_dim
               else:
                   action_idx = random.randint(0, min(self.action_dim - 1, len(merge_candidates) - 1))
           else:
               with torch.no_grad():
                   q_values = self.dqn(state.unsqueeze(0))
                   valid_actions = min(self.action_dim, len(merge_candidates))
                   action_idx = torch.argmax(q_values[0, :valid_actions + 1]).item()

           if action_idx == self.action_dim or action_idx >= len(merge_candidates):
               reward = 0
               next_state = state
           else:
               try:
                   if action_idx < len(merge_candidates):
                       merge_nodes = merge_candidates[action_idx]
                       new_pattern = self.matcher.merge_patterns(pattern1, pattern2, [merge_nodes])

                       next_state = self.get_state(new_pattern)
                       reward = self.get_reward(pattern1, new_pattern)
                   else:
                       reward = 0
                       next_state = state
               except Exception as e:
                   continue

           memory.append((state, action_idx, reward, next_state))

           if len(memory) > 32:
               batch_size = min(32, len(memory))
               batch = random.sample(memory, batch_size)

               states, actions, rewards, next_states = zip(*batch)

               states = torch.stack(states)
               actions = torch.tensor(actions, dtype=torch.long, device=self.device)
               rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
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
           state_input = state.unsqueeze(0)
           if state_input.device != self.device:
               state_input = state_input.to(self.device)
               
           q_values = self.dqn(state_input)

           candidates_count = len(merge_candidates)

           valid_length = min(q_values.size(1), candidates_count + 1)
           valid_q_values = q_values[0, :valid_length]

           best_idx = torch.argmax(valid_q_values).item()

           if best_idx >= candidates_count:
               return None, True

           if best_idx < candidates_count:
               return merge_candidates[best_idx], False
           else:
               return merge_candidates[0], False


import os
import torch
import pickle
import numpy as np
import random
from typing import List, Tuple, Dict, Any, Optional
from collections import defaultdict

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not available, using fallback ranking")

from graph_matcher import AttributePredicate, AttributeComparisonPredicate, WLPredicate, RxGNNs


class PredicateSelector:
    def __init__(self, matcher=None, ppl_file="ppl.pickle", model_name="Qwen/Qwen2.5-1.5B", 
                 use_llm=True, fallback_strategy='support_based'):
        self.matcher = matcher
        self.ppl_file = ppl_file
        self.ppl_cache = {}
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.prompt_template = ""
        self.use_llm = use_llm and TRANSFORMERS_AVAILABLE
        self.fallback_strategy = fallback_strategy
        self.llm_failed = False
        
        self.predicate_stats = defaultdict(lambda: {'pos_count': 0, 'neg_count': 0, 'total_count': 0})
        
        self._load_ppl_cache()
        
        self._load_prompt_template()

    def _load_ppl_cache(self):
        try:
            if os.path.exists(self.ppl_file):
                with open(self.ppl_file, 'rb') as f:
                    self.ppl_cache = pickle.load(f)
                print(f"Loaded {len(self.ppl_cache)} cached perplexity values")
        except Exception as e:
            print(f"Failed to load PPL cache: {e}")
            self.ppl_cache = {}

    def _load_prompt_template(self):
        try:
            with open("prompt.txt", "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
        except Exception as e:
            print(f"Unable to read prompt.txt: {e}, using default template")
            self.prompt_template = """
Given the following rule condition in a loan approval context:
{rule}

Rate how likely this condition is to lead to loan denial on a scale from 1-100, where:
- 1 means very unlikely to cause denial
- 100 means very likely to cause denial

Consider factors like risk assessment, creditworthiness, and financial stability.
"""

    def _load_model(self):
        if not self.use_llm:
            return False
            
        if self.model is None or self.tokenizer is None:
            try:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Loading model {self.model_name} on device: {device}")

                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    padding_side='left'
                )
                
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32
                ).to(device)
                
                self.model.eval()
                print("Model loaded successfully")
                return True
                
            except Exception as e:
                print(f"Failed to load model: {e}")
                self.llm_failed = True
                self.use_llm = False
                return False
        return True

    def _predicate_to_natural_language(self, predicate, pattern):
        try:
            if isinstance(predicate, AttributePredicate):
                node_id = predicate.node_id
                node_type = self._get_node_type_description(node_id, pattern)
                
                op_map = {
                    "==": "equals",
                    ">": "is greater than",
                    "<": "is less than", 
                    ">=": "is at least",
                    "<=": "is at most",
                    "!=": "is not equal to"
                }
                
                operator = op_map.get(predicate.operator, predicate.operator)
                attr_desc = self._get_attribute_description(predicate.attribute)
                
                return f"The {node_type} {attr_desc} {operator} {predicate.value}"

            elif isinstance(predicate, AttributeComparisonPredicate):
                node1_type = self._get_node_type_description(predicate.node1_id, pattern)
                node2_type = self._get_node_type_description(predicate.node2_id, pattern)
                
                op_map = {
                    "==": "equals",
                    ">": "is greater than",
                    "<": "is less than",
                    ">=": "is at least", 
                    "<=": "is at most",
                    "!=": "differs from"
                }
                
                operator = op_map.get(predicate.operator, predicate.operator)
                attr1_desc = self._get_attribute_description(predicate.attr1)
                attr2_desc = self._get_attribute_description(predicate.attr2)
                
                return f"The {node1_type} {attr1_desc} {operator} the {node2_type} {attr2_desc}"

            elif isinstance(predicate, WLPredicate):
                node_type = self._get_node_type_description(predicate.node_id, pattern)
                return f"The {node_type} has similar structural patterns to rejected cases"

            else:
                return str(predicate)
                
        except Exception as e:
            print(f"Error converting predicate to natural language: {e}")
            return f"Predicate: {str(predicate)}"

    def _get_node_type_description(self, node_id, pattern):
        type_map = {
            0: "applicant",
            1: "loan application", 
            2: "city",
            3: "state",
            4: "profession"
        }
        
        try:
            if pattern and node_id in pattern.graph.nodes:
                label = pattern.graph.nodes[node_id].label
                return type_map.get(label, f"node_{node_id}")
            return f"node_{node_id}"
        except:
            return f"node_{node_id}"

    def _get_attribute_description(self, attribute):
        attr_map = {
            'income_level': 'income level',
            'age_level': 'age',
            'experience_level': 'work experience', 
            'job_years_level': 'years at current job',
            'house_years_level': 'years at current address',
            'married_single': 'marital status',
            'house_ownership': 'home ownership status',
            'car_ownership': 'car ownership status',
            'gnn_prediction': 'prediction score'
        }
        
        return attr_map.get(attribute, attribute)

    def _calculate_support_based_score(self, predicate, pattern):
        if self.matcher is None:
            return random.randint(1, 100)
            
        try:
            temp_rule = RxGNNs(pattern)
            temp_rule.add_precondition(predicate)
            
            result = self.matcher.evaluate_rule(temp_rule)
            support = result.get('support', 0)
            confidence = result.get('confidence', 0)
            
            score = max(1, 100 - (support * 10 + confidence * 50))
            return score
            
        except Exception as e:
            print(f"Support calculation failed: {e}")
            return random.randint(50, 100)

    def _calculate_frequency_based_score(self, predicate, pattern):
        pred_key = predicate.description() if hasattr(predicate, 'description') else str(predicate)
        
        stats = self.predicate_stats[pred_key]
        total = stats['total_count']
        
        if total == 0:
            return random.randint(1, 100)
            
        neg_ratio = stats['neg_count'] / total
        score = max(1, int(100 * (1 - neg_ratio)))
        
        return score

    def calculate_ppl(self, predicate, pattern):
        pred_key = predicate.description() if hasattr(predicate, 'description') else str(predicate)
        
        if pred_key in self.ppl_cache:
            return self.ppl_cache[pred_key]

        ppl_value = None
        
        if self.use_llm and not self.llm_failed:
            ppl_value = self._calculate_llm_ppl(predicate, pattern, pred_key)
        
        if ppl_value is None:
            if self.fallback_strategy == 'support_based':
                ppl_value = self._calculate_support_based_score(predicate, pattern)
            elif self.fallback_strategy == 'frequency_based':
                ppl_value = self._calculate_frequency_based_score(predicate, pattern)
            else:  # random
                ppl_value = random.randint(1, 100)
            
            print(f"Using fallback strategy '{self.fallback_strategy}' for predicate: {pred_key[:50]}...")

        self.ppl_cache[pred_key] = ppl_value
        return ppl_value

    def _calculate_llm_ppl(self, predicate, pattern, pred_key):
        try:
            if not self._load_model():
                return None

            natural_language = self._predicate_to_natural_language(predicate, pattern)
            prompt = self.prompt_template.format(rule=natural_language)

            device = next(self.model.parameters()).device
            
            encodings = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(device)

            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=input_ids
                )

                neg_log_likelihood = outputs.loss.item()
                ppl = np.exp(neg_log_likelihood)
                
                ppl = max(1, min(ppl, 1000))

            return ppl

        except Exception as e:
            self.llm_failed = True
            self.use_llm = False
            return None

    def rank_predicates(self, predicates, pattern):
        if not predicates:
            return []

        scored_predicates = []
        print(f"Ranking {len(predicates)} predicates...")

        for i, pred in enumerate(predicates):
            try:
                ppl = self.calculate_ppl(pred, pattern)
                scored_predicates.append((pred, ppl))
                
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(predicates)} predicates")
                    
            except Exception as e:
                print(f"Error processing predicate {i}: {e}")
                scored_predicates.append((pred, 50))

        ranked = sorted(scored_predicates, key=lambda x: x[1])
        
        print(f"Ranking complete. PPL range: {ranked[0][1]:.2f} - {ranked[-1][1]:.2f}")
        return ranked

    def update_predicate_stats(self, predicate, is_positive_case):
        pred_key = predicate.description() if hasattr(predicate, 'description') else str(predicate)
        
        self.predicate_stats[pred_key]['total_count'] += 1
        if is_positive_case:
            self.predicate_stats[pred_key]['pos_count'] += 1
        else:
            self.predicate_stats[pred_key]['neg_count'] += 1

    def save_ppl_cache(self, filename=None):
        if filename is None:
            filename = self.ppl_file

        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.ppl_cache, f)
            print(f"Saved {len(self.ppl_cache)} PPL values to cache")
        except Exception as e:
            print(f"Failed to save PPL cache: {e}")

    def get_stats(self):
        return {
            'cache_size': len(self.ppl_cache),
            'llm_available': self.use_llm and not self.llm_failed,
            'fallback_strategy': self.fallback_strategy,
            'predicate_stats_count': len(self.predicate_stats)
        }

    def __del__(self):
        try:
            self.save_ppl_cache()
        except:
            pass

class RuleDiscovery:
   def __init__(self, data_graph, motifs=None, support_threshold=5, confidence_threshold=0.5,
                max_verification_time=50, max_pattern_combinations=3, ppl_file="ppl.pickle",
                sample_ratio=0.0001, min_nodes=100, gnn_model_type='GCN', dataset_name='loan'):
       self.original_graph = data_graph

       self.motifs_input = motifs
       self.support_threshold = support_threshold
       self.confidence_threshold = confidence_threshold
       self.max_verification_time = max_verification_time
       self.max_pattern_combinations = max_pattern_combinations
       self.gnn_model_type = gnn_model_type
       self.dataset_name = dataset_name

       if sample_ratio < 1.0:
           self.data_graph = self.sample_graph(data_graph, sample_ratio, min_nodes)
       else:
           self.data_graph = data_graph

       self.matcher = Matcher(self.data_graph)

       self.motifs = motifs if motifs is not None else self.generate_random_motifs(k=5, max_nodes=3)

       self.gnn_model_loader = GNNModelLoader(dataset_name, gnn_model_type)
       gnn_loaded = self.gnn_model_loader.load_model()
       if not gnn_loaded:
           self.gnn_model_loader = None

       self.composer = PatternComposer(self.matcher, self.motifs, self.gnn_model_loader)
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

       return sampled_graph

   def _serialize_pattern(self, pattern):
       nodes_str = sorted([f"{nid}:{pattern.graph.nodes[nid].label}"
                           for nid in pattern.graph.nodes])
       edges_str = sorted([f"{src}->{tgt}" for src, tgt in pattern.graph.edges])
       return "|".join(nodes_str) + "#" + "|".join(edges_str)

   def evaluate_with_timeout(self, rule):
       from tqdm import tqdm
       import threading

       result = {'support': 0, 'confidence': 0}
       error_occurred = [False]
       completion_event = threading.Event()

       def evaluation_task():
           nonlocal result
           try:
               progress = tqdm(desc="Regelauswertung", leave=False, position=0)
               progress.set_description("Starte Regelauswertung...")

               result = self.matcher.evaluate_rule(rule)

               progress.set_description("Regelauswertung abgeschlossen")
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

       timeout_bar = tqdm(total=self.max_verification_time, desc="Warte auf Regelverifizierung", leave=False, position=1)

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

           combine_progress = tqdm(enumerate(self.motifs), desc=f"Thread {thread_id}: Musterkombination",
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
                   "Kombinationen": current_count,
                   "Zielmuster": i,
                   "Gefundene Regeln": len(self.discovered_rules)
               })

               try:
                   best_merge, should_stop = self.composer.select_best_merge_or_stop(pattern, other_pattern)

                   if should_stop:
                       combine_progress.set_postfix({"Status": "Stoppe Zusammenführung", "Kombinationen": current_count})
                       continue

                   if best_merge:
                       combine_progress.set_postfix({"Status": "Zusammenführung", "Kombinationen": current_count})
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
           work_items.append((i, f"Einzelnes-Motif-{i}"))

       for i in range(len(self.motifs)):
           for j in range(i + 1, len(self.motifs)):
               work_items.append((-1, f"Kombination-{i}-{j}"))

       progress_bar = tqdm(total=len(work_items), desc="Gesamtmining-Fortschritt")
       rules_progress = tqdm(total=self.max_rules, desc="Gefundene Regeln")

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
                   f.write(f"Regel {i + 1}: {rule.description()}\n")

   def load_rules(self, filename="rules/discovered_rules.pkl"):
       if os.path.exists(filename):
           with open(filename, 'rb') as f:
               self.discovered_rules = pickle.load(f)
           return True
       return False

   def generate_random_motifs(self, k, max_nodes):
       if k <= 0 or max_nodes <= 0:
           raise ValueError("k und max_nodes müssen positive Ganzzahlen sein")

       motifs = []
       seen_motifs = set()
       matcher = Matcher(self.data_graph)

       all_nodes = list(self.data_graph.nodes.keys())
       gnn_true_nodes = [nid for nid, node in self.data_graph.nodes.items()
                       if node.attributes.get('gnn_prediction', False)]

       node_type_prefixes = {0: 'u', 1: 'l', 2: 'c', 3: 's', 4: 'j'}

       attempts = 0
       max_attempts = k * 10

       while len(motifs) < k and attempts < max_attempts:
           attempts += 1

           if gnn_true_nodes and random.random() < 0.8:
               start_node_id = random.choice(gnn_true_nodes)
           else:
               start_node_id = random.choice(all_nodes)

           start_node_label = self.data_graph.nodes[start_node_id].label
           prefix = node_type_prefixes.get(start_node_label, 'n')
           pivot_id = f"{prefix}1"

           pattern = Pattern(pivot_id)
           visited_nodes = {start_node_id: pivot_id}
           pattern.add_node(Node(pivot_id, start_node_label, {}))

           current_node = start_node_id
           node_counter = {label: 2 for label in node_type_prefixes.keys()}

           for step in range(max_nodes - 1):
               neighbors = []
               for edge in self.data_graph.edges:
                   if edge[0] == current_node:
                       neighbors.append(edge[1])
                   elif edge[1] == current_node:
                       neighbors.append(edge[0])

               if not neighbors:
                   break

               next_node = random.choice(neighbors)
               next_node_label = self.data_graph.nodes[next_node].label

               if next_node in visited_nodes:
                   src_id = visited_nodes[current_node]
                   tgt_id = visited_nodes[next_node]
                   if (src_id, tgt_id) not in pattern.graph.edges:
                       pattern.add_edge(src_id, tgt_id)
               else:
                   prefix = node_type_prefixes.get(next_node_label, 'n')
                   new_node_id = f"{prefix}{node_counter[next_node_label]}"
                   node_counter[next_node_label] += 1

                   visited_nodes[next_node] = new_node_id
                   pattern.add_node(Node(new_node_id, next_node_label, {}))
                   
                   if (current_node, next_node) in self.data_graph.edges:
                       pattern.add_edge(visited_nodes[current_node], new_node_id)
                   else:
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

   def generate_predicates(self, pattern):
       from tqdm import tqdm

       predicates = []

       label_attributes = {}
       attribute_analysis = {}

       node_sample_size = min(2000, len(self.data_graph.nodes))
       sampled_nodes = list(self.data_graph.nodes.items())[:node_sample_size]
       
       for node_id, node in tqdm(sampled_nodes, desc="Analysiere Knotenattribute", leave=False):
           label = node.label
           if label not in label_attributes:
               label_attributes[label] = set()
               attribute_analysis[label] = {}

           for attr, value in node.attributes.items():
               label_attributes[label].add(attr)

               if attr not in attribute_analysis[label]:
                   attribute_analysis[label][attr] = {
                       'values': [],
                       'unique_values': set(),
                       'type_counts': defaultdict(int)
                   }

               if value is not None:
                   attribute_analysis[label][attr]['values'].append(value)
                   attribute_analysis[label][attr]['unique_values'].add(value)
                   attribute_analysis[label][attr]['type_counts'][type(value).__name__] += 1

       for node_id, node in tqdm(pattern.graph.nodes.items(), desc="Generiere Knotenprädikate", leave=False):
           label = node.label
           
           if label not in label_attributes:
               continue

           for attr in label_attributes[label]:
               if attr not in attribute_analysis[label]:
                   continue
                   
               analysis = attribute_analysis[label][attr]
               values = analysis['values']
               unique_values = analysis['unique_values']
               type_counts = analysis['type_counts']
               
               generated_predicates = self._generate_predicates_for_attribute(
                   node_id, attr, values, unique_values, type_counts
               )
               
               predicates.extend(generated_predicates)

       node_pairs = [(n1, n2) for n1 in pattern.graph.nodes.keys() 
                   for n2 in pattern.graph.nodes.keys() if n1 != n2]
       
       if len(node_pairs) > 10:
           node_pairs = node_pairs[:10]
       
       for node1_id, node2_id in tqdm(node_pairs, desc="Generiere Vergleichsprädikate", leave=False):
           node1_label = pattern.graph.nodes[node1_id].label
           node2_label = pattern.graph.nodes[node2_id].label

           if node1_label not in label_attributes or node2_label not in label_attributes:
               continue

           common_attrs = self._find_comparable_attributes(
               label_attributes, attribute_analysis, node1_label, node2_label
           )

           for attr1, attr2 in common_attrs:
               comparison_predicates = [
                   AttributeComparisonPredicate(node1_id, attr1, node2_id, attr2, '=='),
                   AttributeComparisonPredicate(node1_id, attr1, node2_id, attr2, '>'),
                   AttributeComparisonPredicate(node1_id, attr1, node2_id, attr2, '<')
               ]
               predicates.extend(comparison_predicates)
       
       return predicates

   def _generate_predicates_for_attribute(self, node_id, attr_name, values, unique_values, type_counts):
       predicates = []
       
       if not values:
           return predicates
       
       dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
       unique_count = len(unique_values)
       sample_size = len(values)
       
       if dominant_type == 'bool' or unique_values.issubset({True, False, 0, 1}):
           for value in unique_values:
               if value in [True, False, 0, 1]:
                   bool_value = bool(value)
                   predicates.append(AttributePredicate(node_id, attr_name, bool_value, '=='))
           
       elif unique_count <= 10 and unique_count < sample_size * 0.5:
           for value in unique_values:
               if value is not None:
                   predicates.append(AttributePredicate(node_id, attr_name, value, '=='))
       
       elif dominant_type in ['int', 'float'] or all(isinstance(v, (int, float)) for v in unique_values if v is not None):
           numeric_values = [v for v in values if isinstance(v, (int, float)) and v is not None]
           
           if not numeric_values:
               return predicates
               
           if (all(isinstance(v, int) for v in numeric_values) and 
               min(numeric_values) >= 1 and max(numeric_values) <= 10 and
               unique_count <= 10):
               min_val = min(numeric_values)
               max_val = max(numeric_values)
               
               if max_val > min_val:
                   low_threshold = min_val + (max_val - min_val) * 0.33
                   high_threshold = min_val + (max_val - min_val) * 0.67
                   
                   predicates.extend([
                       AttributePredicate(node_id, attr_name, int(low_threshold), '<='),
                       AttributePredicate(node_id, attr_name, int(low_threshold), '>='),
                       AttributePredicate(node_id, attr_name, int(high_threshold), '<='),
                       AttributePredicate(node_id, attr_name, int(high_threshold), '>=')
                   ])
           else:
               import statistics
               
               try:
                   mean_val = statistics.mean(numeric_values)
                   median_val = statistics.median(numeric_values)
                   
                   if len(numeric_values) >= 4:
                       sorted_values = sorted(numeric_values)
                       q1 = statistics.median(sorted_values[:len(sorted_values)//2])
                       q3 = statistics.median(sorted_values[(len(sorted_values)+1)//2:])
                       
                       thresholds = [q1, median_val, q3]
                   else:
                       thresholds = [median_val]
                   
                   for threshold in thresholds:
                       predicates.extend([
                           AttributePredicate(node_id, attr_name, threshold, '<='),
                           AttributePredicate(node_id, attr_name, threshold, '>=')
                       ])
                       
               except statistics.StatisticsError:
                   min_val = min(numeric_values)
                   max_val = max(numeric_values)
                   if max_val > min_val:
                       mid_val = (min_val + max_val) / 2
                       predicates.extend([
                           AttributePredicate(node_id, attr_name, mid_val, '<='),
                           AttributePredicate(node_id, attr_name, mid_val, '>=')
                       ])
       
       elif dominant_type == 'str':
           if unique_count <= 20:
               for value in unique_values:
                   if value is not None and isinstance(value, str):
                       predicates.append(AttributePredicate(node_id, attr_name, value, '=='))
       
       else:
           main_type_values = [v for v in unique_values 
                           if type(v).__name__ == dominant_type and v is not None]
           
           if main_type_values and len(main_type_values) <= 10:
               for value in main_type_values:
                   predicates.append(AttributePredicate(node_id, attr_name, value, '=='))
       
       return predicates

   def _find_comparable_attributes(self, label_attributes, attribute_analysis, label1, label2):
       comparable_attrs = []
       
       attrs1 = list(label_attributes[label1])[:5]
       attrs2 = list(label_attributes[label2])[:5]
       
       for attr1 in attrs1:
           for attr2 in attrs2:
               if attr1 == attr2:
                   if (attr1 in attribute_analysis[label1] and 
                       attr2 in attribute_analysis[label2]):
                       
                       analysis1 = attribute_analysis[label1][attr1]
                       analysis2 = attribute_analysis[label2][attr2]
                       
                       type1 = max(analysis1['type_counts'].items(), key=lambda x: x[1])[0]
                       type2 = max(analysis2['type_counts'].items(), key=lambda x: x[1])[0]
                       
                       if type1 in ['int', 'float'] and type2 in ['int', 'float']:
                           comparable_attrs.append((attr1, attr2))
                       elif type1 == 'str' and type2 == 'str':
                           comparable_attrs.append((attr1, attr2))
                       elif type1 == 'bool' and type2 == 'bool':
                           comparable_attrs.append((attr1, attr2))
       
       return comparable_attrs

   def level_wise_mining(self, pattern, thread_id):
       from tqdm import tqdm

       try:
           predicates = self.generate_predicates(pattern)

           if not predicates:
               return

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

           exploration_progress = tqdm(desc=f"Thread {thread_id}: Prädikatexploration", position=thread_id % 10)
           combo_count = 0

           while queue and len(self.discovered_rules) < self.max_rules:
               current_predicates = queue.popleft()
               combo_count += 1
               exploration_progress.update(1)
               exploration_progress.set_description(f"Thread {thread_id}: Exploriert {combo_count} Kombinationen")

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
           import traceback
           traceback.print_exc()

def main():
   parser = argparse.ArgumentParser(description='NEX: Graph Neural Network Rule Discovery')
   
   parser.add_argument('--dataset', type=str, required=True)
   parser.add_argument('--gnn_type', type=str, required=True)
   parser.add_argument('--support_threshold', type=int, required=True)
   parser.add_argument('--confidence_threshold', type=float, required=True)
   parser.add_argument('--max_rules', type=int, required=True)
   parser.add_argument('--num_threads', type=int, required=True)
   parser.add_argument('--sample_ratio', type=float, required=True)
   parser.add_argument('--max_verification_time', type=int, required=True)
   parser.add_argument('--output_dir', type=str, required=True)
   
   args = parser.parse_args()
   
   os.makedirs(args.output_dir, exist_ok=True)
   
   dataset_path = f'data/{args.dataset}_graph.pkl'
   if not os.path.exists(dataset_path):
       return 1
   
   try:
       with open(dataset_path, 'rb') as f:
           data_graph = pickle.load(f)
       
       rule_discovery = RuleDiscovery(
           data_graph=data_graph,
           motifs=None,
           support_threshold=args.support_threshold,
           confidence_threshold=args.confidence_threshold,
           max_verification_time=args.max_verification_time,
           max_pattern_combinations=20,
           sample_ratio=args.sample_ratio,
           min_nodes=100,
           gnn_model_type=args.gnn_type,
           dataset_name=args.dataset
       )
       
       rule_discovery.max_rules = args.max_rules
       
       discovered_rules = rule_discovery.discover(num_threads=args.num_threads)
       
       result_file = os.path.join(args.output_dir, f"{args.dataset}_{args.gnn_type}_rules.txt")
       with open(result_file, 'w', encoding='utf-8') as f:
           f.write(f"Dataset: {args.dataset}\n")
           f.write(f"GNN Model: {args.gnn_type}\n")
           f.write(f"Support Threshold: {args.support_threshold}\n")
           f.write(f"Confidence Threshold: {args.confidence_threshold}\n")
           f.write(f"Rules Found: {len(discovered_rules)}\n")
           f.write("=" * 50 + "\n")
           
           for i, rule in enumerate(discovered_rules):
               f.write(f"\nRule {i+1}:\n")
               f.write(f"  Pattern: {len(rule.pattern.graph.nodes)} nodes, {len(rule.pattern.graph.edges)} edges\n")
               f.write(f"  Preconditions: {len(rule.preconditions)} predicates\n")
               if hasattr(rule, 'description'):
                   f.write(f"  Description: {rule.description()}\n")
               f.write("-" * 30 + "\n")
       
       rule_discovery.predicate_selector.save_ppl_cache()
       
   except Exception as e:
       return 1
   
   return 0


if __name__ == "__main__":
  exit(main())