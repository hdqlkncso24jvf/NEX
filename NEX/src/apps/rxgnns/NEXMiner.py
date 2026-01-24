import os
import pickle
import random
import threading
import time
import math
from collections import defaultdict, deque
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import argparse
from copy import deepcopy
import hashlib
from typing import List, Set, Dict, Tuple, Optional
from queue import Queue, Empty

from graph_matcher import (
    Matcher, Pattern, Node, Graph,
    AttributePredicate, AttributeComparisonPredicate,
    WLPredicate, RxGNNs
)

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data

MODEL_DIR = "models"
RULES_DIR = "rules"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RULES_DIR, exist_ok=True)

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

                        if src_id in graph.nodes and graph.nodes[src_id].label == self.node_types.index(
                                self.primary_node_type):
                            graph.nodes[src_id].attributes['edge_type'] = edge_type
                        elif tgt_id in graph.nodes and graph.nodes[tgt_id].label == self.node_types.index(
                                self.primary_node_type):
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


AMAZON_CATEGORIES = {
    'Lingerie': 0,
    'Jewelry': 1,
    'Womens-Fashion': 2,
    'Mens-Fashion': 3,
    'Sports-outdoors': 4
}


def identify_dataset(dataset_name: str) -> str:
    dataset_lower = dataset_name.lower()
    if 'insurance' in dataset_lower:
        return 'insurance'
    elif 'loan' in dataset_lower:
        return 'loan'
    elif 'transaction' in dataset_lower or 'trans' in dataset_lower:
        return 'transaction'
    elif 'amazon' in dataset_lower:
        return 'amazon'
    else:
        return 'binary'


def load_and_process_graph(dataset_path: str, dataset_type: str, target_class: Optional[str] = None) -> Graph:
    if dataset_type == 'amazon':
        if target_class is None:
            raise ValueError(
                f"Amazon dataset must specify a target class! "
                f"Available classes: {', '.join(AMAZON_CATEGORIES.keys())}. "
                f"Use --target_class to specify, e.g., --target_class Jewelry"
            )
        if target_class not in AMAZON_CATEGORIES:
            raise ValueError(
                f"Invalid category: {target_class}. "
                f"Available classes: {', '.join(AMAZON_CATEGORIES.keys())}"
            )

    with open(dataset_path, 'rb') as f:
        original_graph = pickle.load(f)

    if dataset_type in ['insurance', 'loan', 'transaction', 'binary']:
        pass
    elif dataset_type == 'amazon':
        target_label = AMAZON_CATEGORIES[target_class]
        for node_id, node in original_graph.nodes.items():
            if node.label == 0:
                original_category = node.attributes.get('category', '')
                if original_category == target_class:
                    node.attributes['gnn_prediction'] = True
                else:
                    node.attributes['gnn_prediction'] = False
    return original_graph


def analyze_graph_structure(data_graph: Graph) -> Tuple[Dict, np.ndarray]:
    label_distribution = defaultdict(int)
    for node in data_graph.nodes.values():
        label_distribution[node.label] += 1

    labels = sorted(label_distribution.keys())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    n_labels = len(labels)
    edge_constraint_matrix = np.zeros((n_labels, n_labels), dtype=int)
    edge_type_counts = defaultdict(int)

    for src_id, tgt_id in data_graph.edges:
        src_label = data_graph.nodes[src_id].label
        tgt_label = data_graph.nodes[tgt_id].label
        src_idx = label_to_idx[src_label]
        tgt_idx = label_to_idx[tgt_label]
        edge_constraint_matrix[src_idx][tgt_idx] = 1
        edge_type = (src_label, tgt_label)
        edge_type_counts[edge_type] += 1

    return label_distribution, edge_constraint_matrix


def generate_motifs_bfs(data_graph: Graph,
                        label_distribution: Dict,
                        edge_constraint_matrix: np.ndarray,
                        max_nodes: int = 5) -> List[Pattern]:
    labels = sorted(label_distribution.keys())
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    pivot_label = 0
    all_motifs = []
    seen_motifs = set()

    for target_size in range(2, max_nodes + 1):
        motifs = _generate_motifs_of_size_bfs(
            pivot_label,
            target_size,
            labels,
            label_to_idx,
            edge_constraint_matrix
        )
        for motif in motifs:
            motif_sig = _get_motif_signature(motif)
            if motif_sig not in seen_motifs:
                seen_motifs.add(motif_sig)
                all_motifs.append(motif)
    return all_motifs


def _generate_motifs_of_size_bfs(pivot_label: int,
                                 target_size: int,
                                 labels: List[int],
                                 label_to_idx: Dict[int, int],
                                 edge_constraint_matrix: np.ndarray) -> List[Pattern]:
    motifs = []
    initial_state = {
        'nodes': [('x0', pivot_label)],
        'edges': [],
        'next_node_idx': 1
    }
    queue = [initial_state]

    while queue:
        state = queue.pop(0)
        current_size = len(state['nodes'])
        if current_size == target_size:
            motif = _state_to_pattern(state, pivot_label)
            motifs.append(motif)
            continue
        if len(motifs) > 50:
            break

        for new_label in labels:
            valid_sources = []
            for existing_node_id, existing_label in state['nodes']:
                src_idx = label_to_idx[existing_label]
                tgt_idx = label_to_idx[new_label]
                if edge_constraint_matrix[src_idx][tgt_idx] == 1:
                    valid_sources.append(existing_node_id)

            valid_targets = []
            for existing_node_id, existing_label in state['nodes']:
                src_idx = label_to_idx[new_label]
                tgt_idx = label_to_idx[existing_label]
                if edge_constraint_matrix[src_idx][tgt_idx] == 1:
                    valid_targets.append(existing_node_id)

            if not valid_sources and not valid_targets:
                continue

            for source_id in valid_sources[:3]:
                new_state = _extend_state_directed(state, new_label, [(source_id, 'new')])
                queue.append(new_state)

            for target_id in valid_targets[:3]:
                new_state = _extend_state_directed(state, new_label, [('new', target_id)])
                queue.append(new_state)
    return motifs


def _extend_state_directed(state: Dict, new_label: int, edges: List[Tuple[str, str]]) -> Dict:
    new_node_id = f"x{state['next_node_idx']}"
    new_nodes = state['nodes'] + [(new_node_id, new_label)]
    new_edges = state['edges'].copy()
    for src, tgt in edges:
        if src == 'new': src = new_node_id
        if tgt == 'new': tgt = new_node_id
        new_edges.append((src, tgt))
    return {
        'nodes': new_nodes,
        'edges': new_edges,
        'next_node_idx': state['next_node_idx'] + 1
    }


def _state_to_pattern(state: Dict, pivot_label: int) -> Pattern:
    pattern = Pattern()
    for node_id, label in state['nodes']:
        pattern.add_node(Node(node_id, label, {}))
    for src_id, tgt_id in state['edges']:
        pattern.add_edge(src_id, tgt_id)
    pattern.set_pivot('x0')
    return pattern


def _get_motif_signature(motif: Pattern) -> str:
    node_labels = []
    for node_id in sorted(motif.graph.nodes.keys()):
        node_labels.append(motif.graph.nodes[node_id].label)
    labels_sig = ",".join(map(str, sorted(node_labels)))
    edges_normalized = []
    for src, tgt in motif.graph.edges:
        src_label = motif.graph.nodes[src].label
        tgt_label = motif.graph.nodes[tgt].label
        edge_sig = (src_label, tgt_label)
        edges_normalized.append(edge_sig)
    edges_sig = "|".join(map(str, sorted(edges_normalized)))
    return f"{labels_sig}#{edges_sig}"


class PredicateGenerator:
    def __init__(self, data_graph, dataset_name: str = 'default', dataset_type: str = 'binary',
                 target_class: Optional[str] = None):
        self.data_graph = data_graph
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.target_class = target_class
        self._attr_stats = {}
        self._analyze_attributes()
        self._load_or_generate_ppl()

    def _analyze_attributes(self):
        attr_values = defaultdict(list)
        sampled_nodes = list(self.data_graph.nodes.values())
        if len(sampled_nodes) > 1000:
            sampled_nodes = random.sample(sampled_nodes, 1000)
        for node in sampled_nodes:
            for attr, value in node.attributes.items():
                if attr not in ['gnn_prediction', 'fraud', 'category'] and value is not None:
                    attr_values[attr].append(value)
        for attr, values in attr_values.items():
            if not values: continue
            unique_values = set(values)
            try:
                numeric_values = []
                for v in values:
                    try:
                        numeric_values.append(float(v))
                    except:
                        pass
                if len(numeric_values) > len(values) * 0.8:
                    sorted_nums = sorted(numeric_values)
                    n = len(sorted_nums)
                    self._attr_stats[attr] = {
                        'type': 'continuous',
                        'values': [sorted_nums[0], sorted_nums[n // 2], sorted_nums[-1]]
                    }
                else:
                    self._attr_stats[attr] = {
                        'type': 'categorical',
                        'values': list(unique_values)[:10]
                    }
            except:
                self._attr_stats[attr] = {
                    'type': 'categorical',
                    'values': list(unique_values)[:10]
                }

    def _load_or_generate_ppl(self):
        if self.dataset_type == 'amazon' and self.target_class:
            ppl_file = f"amazon_{self.target_class.lower()}_ppl.pkl"
        else:
            ppl_file = f"{self.dataset_name}_ppl.pkl"
        if os.path.exists(ppl_file):
            try:
                with open(ppl_file, 'rb') as f:
                    self.predicate_priority = pickle.load(f)
            except:
                self.predicate_priority = None
        else:
            self.predicate_priority = None

    def generate_all_predicates(self, pattern) -> List:
        all_predicates = []
        for node_id in pattern.graph.nodes:
            for attr, stats in self._attr_stats.items():
                if stats['type'] == 'continuous':
                    for val in stats['values']:
                        all_predicates.append(AttributePredicate(node_id, attr, val, '>='))
                        all_predicates.append(AttributePredicate(node_id, attr, val, '<'))
                else:
                    for val in stats['values'][:3]:
                        all_predicates.append(AttributePredicate(node_id, attr, val, '=='))
            all_predicates.append(WLPredicate(node_id, is_negated=True, gnn_attr='gnn_prediction'))
        if self.predicate_priority is not None:
            all_predicates = self._sort_by_ppl(all_predicates)
        else:
            random.shuffle(all_predicates)
        return all_predicates

    def _sort_by_ppl(self, predicates: List) -> List:
        pred_dict = {p.description(): p for p in predicates}
        sorted_predicates = []
        for pred_desc in self.predicate_priority:
            if pred_desc in pred_dict:
                sorted_predicates.append(pred_dict[pred_desc])
        ppl_set = set(self.predicate_priority)
        for pred in predicates:
            if pred.description() not in ppl_set:
                sorted_predicates.append(pred)
        return sorted_predicates


class GraphReadoutEncoder(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=32, device='cpu'):
        super(GraphReadoutEncoder, self).__init__()
        self.device = device
        self.conv1 = GCNConv(10, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.to(device)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return global_mean_pool(x, batch)

    def encode_pattern(self, pattern: Pattern):
        data = self._pattern_to_data(pattern)
        data = data.to(self.device)
        with torch.no_grad():
            embedding = self.forward(data.x, data.edge_index, data.batch)
        return embedding.squeeze().cpu().numpy()

    def _pattern_to_data(self, pattern: Pattern):
        node_list = list(pattern.graph.nodes.keys())
        node_to_idx = {n: i for i, n in enumerate(node_list)}
        x = []
        for node_id in node_list:
            node = pattern.graph.nodes[node_id]
            label_hash = hash(node.label) % 100 / 100.0
            degree = sum(1 for e in pattern.graph.edges if node_id in e)
            features = [label_hash, degree / max(len(node_list), 1)] + [0.0] * 8
            x.append(features)
        x = torch.tensor(x, dtype=torch.float)
        edge_index = []
        for src, tgt in pattern.graph.edges:
            edge_index.append([node_to_idx[src], node_to_idx[tgt]])
            edge_index.append([node_to_idx[tgt], node_to_idx[src]])
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        batch = torch.zeros(len(node_list), dtype=torch.long)
        return Data(x=x, edge_index=edge_index, batch=batch)


class DQNMerger:
    def __init__(self, encoder, data_graph, sample_ratio: float):
        self.encoder = encoder
        self.data_graph = data_graph
        self.sample_ratio = sample_ratio
        self.matcher = Matcher(data_graph)
        state_dim = 32
        max_pairs = 50
        self.action_dim = max_pairs + 1
        self.q_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_dim)
        ).to(encoder.device)
        self.target_network = deepcopy(self.q_network)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.replay_buffer = deque(maxlen=10000)
        self.epsilon = 0.2
        self.gamma = 0.95
        self.train_steps = 0
        self.model_file = os.path.join("models", "dqn_merger.pth")
        if not self._load_model():
            self._train_dqn()

    def _load_model(self) -> bool:
        if os.path.exists(self.model_file):
            try:
                checkpoint = torch.load(self.model_file)
                self.q_network.load_state_dict(checkpoint['q_network'])
                self.target_network.load_state_dict(checkpoint['target_network'])
                self.train_steps = checkpoint.get('train_steps', 0)
                return True
            except:
                return False
        return False

    def _train_dqn(self):
        training_episodes = 100
        negative_nodes = [
            nid for nid, n in self.data_graph.nodes.items()
            if not n.attributes.get('gnn_prediction', True)
        ]
        if len(negative_nodes) > 75:
            negative_nodes = random.sample(negative_nodes, 75)
        pbar = tqdm(range(training_episodes))
        for episode in pbar:
            try:
                pivot_id = random.choice(negative_nodes)
                pattern = self._extract_local_pattern(pivot_id)
                if pattern is None or len(pattern.graph.nodes) < 3: continue
                conf_before = self._get_pattern_confidence(pattern)
                merge_sequence, final_pattern = self._execute_merge_sequence(pattern, use_epsilon=True)
                if final_pattern is None: continue
                conf_after = self._get_pattern_confidence(final_pattern)
                reward = self._compute_reward(conf_before, conf_after, len(merge_sequence))
                state_before = self.encoder.encode_pattern(pattern)
                state_after = self.encoder.encode_pattern(final_pattern)
                for action_idx in merge_sequence:
                    self.replay_buffer.append((state_before, action_idx, reward, state_after, False))
                if len(self.replay_buffer) >= 64:
                    loss = self._update_network()
                if episode % 30 == 0 and episode > 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
            except:
                continue
        self.save_model()

    def _execute_merge_sequence(self, pattern: Pattern, use_epsilon: bool = False) -> Tuple[
        List[int], Optional[Pattern]]:
        current_pattern = pattern
        merge_sequence = []
        max_iterations = 5
        for iteration in range(max_iterations):
            mergeable_pairs = self._get_mergeable_pairs(current_pattern)
            if not mergeable_pairs: break
            if use_epsilon and random.random() < self.epsilon:
                if random.random() < 0.3:
                    action_idx = len(mergeable_pairs)
                    break
                else:
                    action_idx = random.randint(0, len(mergeable_pairs) - 1)
            else:
                state = self.encoder.encode_pattern(current_pattern)
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.encoder.device)
                    q_values = self.q_network(state_tensor).cpu().numpy()[0]
                valid_q = q_values[:len(mergeable_pairs) + 1]
                action_idx = int(np.argmax(valid_q))
            if action_idx >= len(mergeable_pairs): break
            merge_sequence.append(action_idx)
            n1, n2 = mergeable_pairs[action_idx]
            current_pattern = self._merge_two_nodes(current_pattern, n1, n2)
            if current_pattern is None: return merge_sequence, None
        return merge_sequence, current_pattern

    def _compute_reward(self, conf_before: float, conf_after: float, num_merges: int) -> float:
        delta_conf = conf_after - conf_before
        reward = delta_conf * 10.0 - num_merges * 0.1
        return reward

    def decide_merges(self, pattern: Pattern) -> List[Tuple[str, str]]:
        merge_sequence, final_pattern = self._execute_merge_sequence(pattern, use_epsilon=False)
        if not merge_sequence or final_pattern is None: return []
        current_pattern = pattern
        all_merge_pairs = []
        for action_idx in merge_sequence:
            mergeable_pairs = self._get_mergeable_pairs(current_pattern)
            if action_idx >= len(mergeable_pairs): break
            merge_pair = mergeable_pairs[action_idx]
            all_merge_pairs.append(merge_pair)
            current_pattern = self._merge_two_nodes(current_pattern, merge_pair[0], merge_pair[1])
            if current_pattern is None: break
        return all_merge_pairs

    def _extract_local_pattern(self, pivot_id) -> Optional[Pattern]:
        try:
            visited = {pivot_id}
            frontier = {pivot_id}
            for _ in range(2):
                next_frontier = set()
                for node in frontier:
                    for edge in self.data_graph.edges:
                        if edge[0] == node and edge[1] not in visited:
                            next_frontier.add(edge[1])
                        elif edge[1] == node and edge[0] not in visited:
                            next_frontier.add(edge[0])
                frontier = next_frontier
                visited.update(frontier)
                if not frontier: break
            if len(visited) > 8:
                visited = set(random.sample(list(visited), 8))
                visited.add(pivot_id)
            pattern = Pattern()
            node_id_map = {}
            for i, nid in enumerate(visited):
                if nid not in self.data_graph.nodes: continue
                node = self.data_graph.nodes[nid]
                var_id = f'x{i}'
                node_id_map[nid] = var_id
                pattern.add_node(Node(var_id, node.label, {}))
                if nid == pivot_id: pattern.set_pivot(var_id)
            for edge in self.data_graph.edges:
                if edge[0] in node_id_map and edge[1] in node_id_map:
                    pattern.add_edge(node_id_map[edge[0]], node_id_map[edge[1]])
            return pattern if len(pattern.graph.nodes) >= 2 else None
        except:
            return None

    def _get_pattern_confidence(self, pattern: Pattern) -> float:
        try:
            rule = RxGNNs(pattern)
            eval_result = self.matcher.evaluate_rule(rule, gnn_attr='gnn_prediction')
            return eval_result['confidence']
        except:
            return 0.0

    def _merge_two_nodes(self, pattern: Pattern, n1: str, n2: str) -> Optional[Pattern]:
        try:
            if n1 not in pattern.graph.nodes or n2 not in pattern.graph.nodes: return None
            merged = Pattern()
            for node_id, node in pattern.graph.nodes.items():
                if node_id == n2: continue
                merged.add_node(Node(node_id, node.label, node.attributes.copy()))
            merged.set_pivot(pattern.pivot_id)
            for src, tgt in pattern.graph.edges:
                src_new = n1 if src == n2 else src
                tgt_new = n1 if tgt == n2 else tgt
                if src_new != tgt_new and (src_new, tgt_new) not in merged.graph.edges:
                    merged.add_edge(src_new, tgt_new)
            return merged
        except:
            return None

    def _update_network(self):
        batch_size = 128
        if len(self.replay_buffer) < batch_size: batch_size = len(self.replay_buffer)
        batch = random.sample(self.replay_buffer, batch_size)
        states = torch.FloatTensor([s for s, _, _, _, _ in batch]).to(self.encoder.device)
        actions = torch.LongTensor([a for _, a, _, _, _ in batch]).to(self.encoder.device)
        rewards = torch.FloatTensor([r for _, _, r, _, _ in batch]).to(self.encoder.device)
        next_states = torch.FloatTensor([s for _, _, _, s, _ in batch]).to(self.encoder.device)
        dones = torch.FloatTensor([d for _, _, _, _, d in batch]).to(self.encoder.device)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q_values * (1 - dones)
        loss = F.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        self.train_steps += 1
        return loss.item()

    def _get_mergeable_pairs(self, pattern: Pattern) -> List[Tuple[str, str]]:
        pairs = []
        node_ids = list(pattern.graph.nodes.keys())
        for i, n1 in enumerate(node_ids):
            if n1 == pattern.pivot_id: continue
            for n2 in node_ids[i + 1:]:
                if n2 == pattern.pivot_id: continue
                node1 = pattern.graph.nodes[n1]
                node2 = pattern.graph.nodes[n2]
                if node1.label == node2.label:
                    pairs.append((n1, n2))
        return pairs

    def save_model(self):
        try:
            checkpoint = {
                'q_network': self.q_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'train_steps': self.train_steps
            }
            torch.save(checkpoint, self.model_file)
        except:
            pass


class MotifMerger:
    def __init__(self, dqn: DQNMerger):
        self.dqn = dqn

    def merge_with_dqn(self, pattern1: Pattern, pattern2: Pattern) -> Optional[Pattern]:
        try:
            merged = self._merge_pivots(pattern1, pattern2)
            if merged is None: return None
            merge_pairs = self.dqn.decide_merges(merged)
            for n1, n2 in merge_pairs:
                if merged.graph.nodes[n1].label != merged.graph.nodes[n2].label: continue
                merged = self._merge_two_nodes(merged, n1, n2)
                if merged is None: return None
            return merged
        except:
            return None

    def _merge_pivots(self, p1: Pattern, p2: Pattern) -> Optional[Pattern]:
        try:
            if p1.graph.nodes[p1.pivot_id].label != p2.graph.nodes[p2.pivot_id].label: return None
            merged = Pattern()
            for node_id, node in p1.graph.nodes.items():
                merged.add_node(Node(node_id, node.label, node.attributes.copy()))
            for edge in p1.graph.edges:
                merged.add_edge(edge[0], edge[1])
            merged.set_pivot(p1.pivot_id)
            for node_id, node in p2.graph.nodes.items():
                if node_id == p2.pivot_id: continue
                new_id = f"{node_id}_m"
                merged.add_node(Node(new_id, node.label, node.attributes.copy()))
            for src, tgt in p2.graph.edges:
                src_new = p1.pivot_id if src == p2.pivot_id else f"{src}_m"
                tgt_new = p1.pivot_id if tgt == p2.pivot_id else f"{tgt}_m"
                if (src_new, tgt_new) not in merged.graph.edges:
                    merged.add_edge(src_new, tgt_new)
            return merged
        except:
            return None

    def _merge_two_nodes(self, pattern: Pattern, n1: str, n2: str) -> Optional[Pattern]:
        try:
            if n1 not in pattern.graph.nodes or n2 not in pattern.graph.nodes: return None
            merged = Pattern()
            for node_id, node in pattern.graph.nodes.items():
                if node_id == n2: continue
                merged.add_node(Node(node_id, node.label, node.attributes.copy()))
            merged.set_pivot(pattern.pivot_id)
            for src, tgt in pattern.graph.edges:
                src_new = n1 if src == n2 else src
                tgt_new = n1 if tgt == n2 else tgt
                if src_new != tgt_new and (src_new, tgt_new) not in merged.graph.edges:
                    merged.add_edge(src_new, tgt_new)
            return merged
        except:
            return None


class CoverAlgorithm:
    def __init__(self, data_graph: Graph):
        self.data_graph = data_graph
        self.matcher = Matcher(data_graph)

    def compute_cover(self, rules: List[RxGNNs]) -> Dict:
        all_negatives = set(
            nid for nid, n in self.data_graph.nodes.items()
            if not n.attributes.get('gnn_prediction', True)
        )
        covered = set()
        selected_rules = []
        rule_coverage = {}
        pbar = tqdm(total=len(all_negatives))

        while len(covered) < len(all_negatives) and rules:
            best_rule_idx = None
            best_new_covered = set()
            for i, rule in enumerate(rules):
                if i in selected_rules: continue
                eval_result = self.matcher.evaluate_rule(rule, verbose=False, gnn_attr='gnn_prediction')
                rule_covered = set()
                for mapping in eval_result.get('satisfies_gnn_false', []):
                    pivot_id = rule.pattern.pivot_id
                    if pivot_id in mapping:
                        rule_covered.add(mapping[pivot_id])
                new_covered = rule_covered - covered
                if len(new_covered) > len(best_new_covered):
                    best_rule_idx = i
                    best_new_covered = new_covered
            if best_rule_idx is None or not best_new_covered: break
            selected_rules.append(best_rule_idx)
            rule_coverage[best_rule_idx] = best_new_covered
            covered.update(best_new_covered)
            pbar.update(len(best_new_covered))
        pbar.close()
        coverage_rate = len(covered) / len(all_negatives) if all_negatives else 0
        return {
            'selected_rules': selected_rules,
            'covered_nodes': covered,
            'coverage_rate': coverage_rate,
            'rule_coverage': rule_coverage
        }


class LevelWiseCache:
    def __init__(self):
        self.validated_rules = {}
        self.in_progress = set()
        self.lock = threading.Lock()

    def is_superset_validated(self, pattern: Pattern, predicates: Set) -> bool:
        with self.lock:
            pattern_sig = self._pattern_signature(pattern)
            if pattern_sig not in self.validated_rules: return False
            pred_descs = frozenset(p.description() for p in predicates)
            for validated_set in self.validated_rules[pattern_sig]:
                if validated_set.issubset(pred_descs) and validated_set != pred_descs:
                    return True
            return False

    def mark_validated(self, pattern: Pattern, predicates: Set):
        with self.lock:
            pattern_sig = self._pattern_signature(pattern)
            if pattern_sig not in self.validated_rules:
                self.validated_rules[pattern_sig] = set()
            pred_descs = frozenset(p.description() for p in predicates)
            self.validated_rules[pattern_sig].add(pred_descs)

    def is_in_progress(self, pattern: Pattern, predicates: Set) -> bool:
        with self.lock:
            sig = self._get_rule_signature(pattern, predicates)
            return sig in self.in_progress

    def mark_in_progress(self, pattern: Pattern, predicates: Set):
        with self.lock:
            sig = self._get_rule_signature(pattern, predicates)
            self.in_progress.add(sig)

    def mark_done(self, pattern: Pattern, predicates: Set):
        with self.lock:
            sig = self._get_rule_signature(pattern, predicates)
            self.in_progress.discard(sig)

    def _pattern_signature(self, pattern: Pattern) -> str:
        nodes = tuple(sorted([(nid, n.label) for nid, n in pattern.graph.nodes.items()]))
        edges = tuple(sorted(pattern.graph.edges))
        return str((nodes, edges))

    def _get_rule_signature(self, pattern: Pattern, predicates: Set) -> str:
        pattern_sig = self._pattern_signature(pattern)
        pred_sig = tuple(sorted(p.description() for p in predicates))
        return hashlib.md5(str((pattern_sig, pred_sig)).encode()).hexdigest()


class MiningTask:
    def __init__(self, pattern: Pattern, predicates: Set, thread_id: str, level: int):
        self.pattern = pattern
        self.predicates = predicates
        self.thread_id = thread_id
        self.level = level

    def description(self) -> str:
        pattern_desc = f"{len(self.pattern.graph.nodes)} nodes"
        pred_desc = f"{len(self.predicates)} predicates"
        return f"[{self.thread_id}] L{self.level}: {pattern_desc}, {pred_desc}"


class LevelWiseParallelMiner:
    def __init__(self, data_graph, valid_graph, motifs: List,
                 support_threshold=200, confidence_threshold=0.6,
                 max_predicates=4, max_motif_merges=3, max_threads=32,
                 sample_ratio=0.01, max_time=36000, dataset_name='default',
                 label_distribution=None, edge_constraint_matrix=None,
                 dataset_type='binary', target_class=None):
        self.data_graph = data_graph
        self.valid_graph = valid_graph
        self.motifs = motifs
        self.support_threshold = support_threshold
        self.confidence_threshold = confidence_threshold
        self.max_predicates = max_predicates
        self.max_motif_merges = max_motif_merges
        self.max_threads = max_threads
        self.sample_ratio = sample_ratio
        self.max_time = max_time
        self.dataset_name = dataset_name
        self.dataset_type = dataset_type
        self.target_class = target_class
        self.label_distribution = label_distribution
        self.edge_constraint_matrix = edge_constraint_matrix
        self.matcher = Matcher(data_graph)
        self.valid_matcher = Matcher(valid_graph)
        self.predicate_generator = PredicateGenerator(data_graph, dataset_name, dataset_type, target_class)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.encoder = GraphReadoutEncoder(device=device)
        self.dqn = DQNMerger(self.encoder, data_graph, sample_ratio)
        self.merger = MotifMerger(self.dqn)
        self.cache = LevelWiseCache()
        self.task_queue = Queue()
        self.active_threads = 0
        self.thread_lock = threading.Lock()
        self.stats = {'total_tasks': 0, 'valid_rules': 0, 'pruned_support': 0, 'pruned_confidence': 0,
                      'pruned_levelwise': 0, 'cache_hits': 0}
        self.stats_lock = threading.Lock()
        self.rule_counter = 0
        self.rule_lock = threading.Lock()
        self.valid_negatives = set(
            nid for nid, n in valid_graph.nodes.items() if not n.attributes.get('gnn_prediction', True))
        self.covered_negatives = set()
        self.covered_lock = threading.Lock()
        self.stop_flag = threading.Event()
        self.start_time = None
        self.pbar = None
        self.pbar_lock = threading.Lock()

    def mine(self):
        self.start_time = time.time()
        self.pbar = tqdm(total=0, position=0, bar_format='{desc}: {n} rules | {elapsed}')
        for i, motif in enumerate(self.motifs):
            task = MiningTask(motif, set(), f"T{i}", 0)
            self.task_queue.put(task)
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(self._worker) for _ in range(self.max_threads)]
            for future in futures: future.result()
        if self.pbar: self.pbar.close()
        self.dqn.save_model()

    def _worker(self):
        while not self.stop_flag.is_set():
            try:
                task = self.task_queue.get(timeout=1)
            except Empty:
                with self.thread_lock:
                    if self.active_threads == 0 and self.task_queue.empty(): break
                continue
            with self.thread_lock:
                self.active_threads += 1
            try:
                self._process_task(task)
            finally:
                with self.thread_lock:
                    self.active_threads -= 1
            if self._should_stop():
                self.stop_flag.set()
                break

    def _process_task(self, task):
        pattern = task.pattern
        predicates = task.predicates
        level = task.level
        with self.stats_lock:
            self.stats['total_tasks'] += 1
        if level > 10: return
        if self.cache.is_superset_validated(pattern, predicates):
            with self.stats_lock: self.stats['pruned_levelwise'] += 1
            return
        if self.cache.is_in_progress(pattern, predicates):
            with self.stats_lock: self.stats['cache_hits'] += 1
            return
        self.cache.mark_in_progress(pattern, predicates)
        try:
            rule = RxGNNs(pattern)
            if predicates: rule.add_preconditions(list(predicates))
            eval_result = self.matcher.evaluate_rule(rule, verbose=False, gnn_attr='gnn_prediction')
            raw_support = eval_result['support']
            scaled_support = int(raw_support / self.sample_ratio)
            confidence = eval_result['confidence']
            self.cache.mark_done(pattern, predicates)
            if scaled_support < self.support_threshold:
                with self.stats_lock: self.stats['pruned_support'] += 1
                return
            if confidence >= self.confidence_threshold:
                self._save_rule(rule, scaled_support, confidence)
                with self.stats_lock: self.stats['valid_rules'] += 1
                with self.pbar_lock: self.pbar.update(1)
                self.cache.mark_validated(pattern, predicates)
                return
            current_negative_count = scaled_support * confidence
            min_required_negatives = self.support_threshold * self.confidence_threshold
            if current_negative_count < min_required_negatives:
                with self.stats_lock: self.stats['pruned_confidence'] += 1
                return
            self._expand_rule(task)
        except:
            self.cache.mark_done(pattern, predicates)

    def _expand_rule(self, task):
        pattern = task.pattern
        predicates = task.predicates
        thread_id = task.thread_id
        level = task.level
        new_tasks = []
        if len(predicates) < self.max_predicates:
            all_predicates = self.predicate_generator.generate_all_predicates(pattern)
            existing_descs = {p.description() for p in predicates}
            new_predicates = [p for p in all_predicates if p.description() not in existing_descs]
            if len(new_predicates) > 10: new_predicates = random.sample(new_predicates, 10)
            for i, pred in enumerate(new_predicates):
                new_pred_set = predicates.copy()
                new_pred_set.add(pred)
                new_tasks.append(MiningTask(pattern, new_pred_set, f"{thread_id}.P{i}", level + 1))
        if level < self.max_motif_merges and len(pattern.graph.nodes) < 10:
            for i, motif in enumerate(self.motifs[:5]):
                merged = self.merger.merge_with_dqn(pattern, motif)
                if merged is not None:
                    try:
                        merged_support = self.matcher.get_pattern_support(merged)
                        scaled_support = int(merged_support / self.sample_ratio)
                        if scaled_support >= self.support_threshold:
                            new_tasks.append(MiningTask(merged, predicates.copy(), f"{thread_id}.M{i}", level + 1))
                    except:
                        pass
        for new_task in new_tasks: self.task_queue.put(new_task)

    def _save_rule(self, rule, support: int, confidence: float):
        with self.rule_lock:
            self.rule_counter += 1
            rule_id = self.rule_counter
        rule_file = os.path.join(RULES_DIR, f"RxGNNs_{rule_id}.pkl")
        try:
            rule_data = {'rule': rule, 'support': support, 'confidence': confidence,
                         'pattern_size': len(rule.pattern.graph.nodes), 'predicates_count': len(rule.preconditions)}
            with open(rule_file, 'wb') as f:
                pickle.dump(rule_data, f)
            self._update_coverage(rule)
        except:
            pass

    def _update_coverage(self, rule):
        try:
            eval_result = self.valid_matcher.evaluate_rule(rule, verbose=False, gnn_attr='gnn_prediction')
            for mapping in eval_result.get('satisfies_gnn_false', []):
                pivot_id = rule.pattern.pivot_id
                if pivot_id in mapping:
                    with self.covered_lock: self.covered_negatives.add(mapping[pivot_id])
        except:
            pass

    def _should_stop(self) -> bool:
        if time.time() - self.start_time > self.max_time: return True
        with self.covered_lock:
            if len(self.covered_negatives) >= len(self.valid_negatives): return True
        return False


def bfs_sample_graph(original_graph: Graph, sample_ratio: float = 0.01, min_nodes: int = 100) -> Graph:
    if sample_ratio >= 1.0: return original_graph
    original_count = len(original_graph.nodes)
    target_count = max(min_nodes, int(original_count * sample_ratio))
    if target_count >= original_count: return original_graph
    negative_nodes, positive_nodes = [], []
    for nid, node in original_graph.nodes.items():
        if node.label == 0:
            if node.attributes.get('gnn_prediction', True) == False:
                negative_nodes.append(nid)
            else:
                positive_nodes.append(nid)
    if not negative_nodes and not positive_nodes: return original_graph
    negative_ratio = len(negative_nodes) / (len(negative_nodes) + len(positive_nodes)) if (len(negative_nodes) + len(
        positive_nodes)) > 0 else 0.5
    target_negative = int(target_count * negative_ratio)
    target_positive = target_count - target_negative
    seeds = []
    if negative_nodes: seeds.extend(random.sample(negative_nodes, min(3, len(negative_nodes))))
    if positive_nodes: seeds.extend(random.sample(positive_nodes, min(3, len(positive_nodes))))
    sampled, queue = set(seeds), deque(seeds)
    sampled_negative = sum(1 for n in sampled if
                           original_graph.nodes[n].label == 0 and original_graph.nodes[n].attributes.get(
                               'gnn_prediction', True) == False)
    sampled_positive = sum(1 for n in sampled if
                           original_graph.nodes[n].label == 0 and original_graph.nodes[n].attributes.get(
                               'gnn_prediction', True) == True)
    pbar = tqdm(total=target_count, initial=len(sampled))
    while queue and len(sampled) < target_count:
        current = queue.popleft()
        neighbors = []
        for edge in original_graph.edges:
            if edge[0] == current and edge[1] not in sampled:
                neighbors.append(edge[1])
            elif edge[1] == current and edge[0] not in sampled:
                neighbors.append(edge[0])
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if len(sampled) >= target_count: break
            node = original_graph.nodes[neighbor]
            if node.label == 0:
                is_negative = node.attributes.get('gnn_prediction', True) == False
                should_add = (sampled_negative < target_negative and is_negative) or (
                            sampled_positive < target_positive and not is_negative) or (
                                         sampled_negative >= target_negative and sampled_positive >= target_positive)
            else:
                should_add = True
            if should_add:
                sampled.add(neighbor)
                queue.append(neighbor)
                if node.label == 0:
                    if node.attributes.get('gnn_prediction', True) == False:
                        sampled_negative += 1
                    else:
                        sampled_positive += 1
                pbar.update(1)
    pbar.close()
    sampled_graph = Graph()
    for nid in sampled:
        node = original_graph.nodes[nid]
        sampled_graph.add_node(Node(nid, node.label, node.attributes.copy()))
    for edge in original_graph.edges:
        if edge[0] in sampled and edge[1] in sampled: sampled_graph.add_edge(edge[0], edge[1])
    return sampled_graph


def verify_gnn_prediction(graph: Graph, stage: str = "", dataset_type: str = 'binary',
                          target_class: Optional[str] = None):
    gnn_false_count, gnn_true_count = 0, 0
    label_0_false, label_0_true = 0, 0
    for node in graph.nodes.values():
        gnn_pred = node.attributes.get('gnn_prediction', True)
        if gnn_pred == False:
            gnn_false_count += 1
            if node.label == 0: label_0_false += 1
        else:
            gnn_true_count += 1
            if node.label == 0: label_0_true += 1
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='insurance')
    parser.add_argument('--target_class', type=str, default=None)
    parser.add_argument('--support', type=int, default=200)
    parser.add_argument('--confidence', type=float, default=0.6)
    parser.add_argument('--max_predicates', type=int, default=4)
    parser.add_argument('--max_motif_merges', type=int, default=3)
    parser.add_argument('--sample_ratio', type=float, default=0.001)
    parser.add_argument('--valid_ratio', type=float, default=0.005)
    parser.add_argument('--max_threads', type=int, default=64)
    parser.add_argument('--max_time', type=int, default=36000)
    args = parser.parse_args()
    dataset_path = f'{args.dataset}_graph.pkl'
    if not os.path.exists(dataset_path): return 1
    try:
        dataset_type = identify_dataset(args.dataset)
        original_graph = load_and_process_graph(dataset_path, dataset_type, args.target_class)
        verify_gnn_prediction(original_graph, "Original", dataset_type, args.target_class)
        sampled_graph = bfs_sample_graph(original_graph, args.sample_ratio)
        verify_gnn_prediction(sampled_graph, "Sampled", dataset_type, args.target_class)
        valid_graph = bfs_sample_graph(original_graph, args.valid_ratio)
        verify_gnn_prediction(valid_graph, "Validation", dataset_type, args.target_class)
        label_distribution, edge_constraint_matrix = analyze_graph_structure(sampled_graph)
        motifs = generate_motifs_bfs(sampled_graph, label_distribution, edge_constraint_matrix, max_nodes=5)
        if not motifs: return 1
        miner = LevelWiseParallelMiner(sampled_graph, valid_graph, motifs, args.support, args.confidence,
                                       args.max_predicates, args.max_motif_merges, args.max_threads, args.sample_ratio,
                                       args.max_time, args.dataset, label_distribution, edge_constraint_matrix,
                                       dataset_type, args.target_class)
        miner.mine()
        all_rules = []
        for filename in os.listdir(RULES_DIR):
            if filename.startswith('RxGNNs_') and filename.endswith('.pkl'):
                rule_path = os.path.join(RULES_DIR, filename)
                with open(rule_path, 'rb') as f:
                    rule_data = pickle.load(f)
                    all_rules.append(rule_data['rule'])
        if all_rules:
            cover_algo = CoverAlgorithm(valid_graph)
            cover_result = cover_algo.compute_cover(all_rules)
            cover_file = os.path.join(RULES_DIR,
                                      f"{args.dataset}_{args.target_class}_cover.pkl" if dataset_type == 'amazon' and args.target_class else f"{args.dataset}_cover.pkl")
            with open(cover_file, 'wb') as f: pickle.dump(cover_result, f)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    exit(main())