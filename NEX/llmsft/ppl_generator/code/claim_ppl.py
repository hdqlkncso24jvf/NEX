import os
import pickle
import random
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional
from tqdm import tqdm
import numpy as np
from pathlib import Path

from graph_matcher import (
    Matcher, Pattern, Node, Graph,
    AttributePredicate, AttributeComparisonPredicate,
    WLPredicate, RxGNNs
)

ATTRIBUTE_METADATA = {
    "attribute_types": {
        "binary": ["group_fraud", "inpatient"],
        "ordinal_meaningful": [
            "reimb_level", "deduct_level", "age", "chronic_count",
            "ip_reimb_level", "op_reimb_level", "ip_deduct_level", "op_deduct_level"
        ],
        "categorical_encoded": ["gender", "state", "county"]
    },

    "attributes_by_label": {
        0: ["reimb_level", "deduct_level", "inpatient", "group_fraud"],
        1: ["age", "gender", "state", "county", "chronic_count",
            "ip_reimb_level", "ip_deduct_level", "op_reimb_level", "op_deduct_level"],
        2: [],
        3: []
    },

    "descriptions": {
        "group_fraud": {
            "neutral": "provider's historical compliance record",
            "evidence": "confirmed record of participation in organized fraud schemes documented by law enforcement"
        },
        "reimb_level": {
            "neutral": "reimbursement amount bracket",
            "evidence": "insurance company's reimbursement payment amount indicating potential overbilling"
        },
        "deduct_level": {
            "neutral": "patient's deductible level",
            "evidence": "patient's out-of-pocket payment revealing potential illegal deductible waivers"
        },
        "inpatient": {
            "neutral": "hospitalization status",
            "evidence": "overnight hospital admission triggering 3-10x higher payouts"
        },
        "chronic_count": {
            "neutral": "number of chronic conditions",
            "evidence": "count of documented chronic diseases that may indicate diagnosis fabrication"
        },
        "ip_reimb_level": {
            "neutral": "annual inpatient reimbursement level",
            "evidence": "total accumulated inpatient reimbursements revealing systematic fraud patterns"
        },
        "op_reimb_level": {
            "neutral": "annual outpatient reimbursement level",
            "evidence": "total accumulated outpatient reimbursements revealing churning patterns"
        },
        "ip_deduct_level": {
            "neutral": "annual inpatient deductible level",
            "evidence": "total accumulated inpatient out-of-pocket costs revealing systematic waivers"
        },
        "op_deduct_level": {
            "neutral": "annual outpatient deductible level",
            "evidence": "total accumulated outpatient out-of-pocket costs revealing systematic waivers"
        },
        "age": {
            "neutral": "patient's age in years",
            "evidence": "patient age for detecting medical impossibilities"
        },
        "gender": {
            "neutral": "patient's gender code",
            "protected": "biological sex classification with zero causal relationship to criminal behavior"
        },
        "state": {
            "neutral": "geographic state code",
            "protected": "U.S. state code where patient resides, cannot determine individual criminal intent"
        },
        "county": {
            "neutral": "geographic county code",
            "protected": "county code where patient resides, no logical connection to fraud propensity"
        }
    }
}



class GraphDataAnalyzer:
    def __init__(self, data_graph: Graph):
        self.data_graph = data_graph
        self.label_distribution = defaultdict(int)
        self.edge_constraints = defaultdict(set)
        self.attribute_info = {}
        self._analyze()

    def _analyze(self):
        for node in self.data_graph.nodes.values():
            self.label_distribution[node.label] += 1

        for src_id, tgt_id in self.data_graph.edges:
            src_label = self.data_graph.nodes[src_id].label
            tgt_label = self.data_graph.nodes[tgt_id].label
            self.edge_constraints[src_label].add(tgt_label)

        label_attributes = defaultdict(lambda: defaultdict(list))
        sample_size = min(20000, len(self.data_graph.nodes))
        sampled_nodes = random.sample(list(self.data_graph.nodes.values()), sample_size)

        for node in sampled_nodes:
            for attr, value in node.attributes.items():
                if attr not in ['gnn_prediction', 'fraud'] and value is not None:
                    label_attributes[node.label][attr].append(value)

        for label in sorted(label_attributes.keys()):
            self.attribute_info[label] = {}
            for attr, values in label_attributes[label].items():
                attr_info = self._analyze_attribute_strict(attr, values, label)
                self.attribute_info[label][attr] = attr_info

    def _analyze_attribute_strict(self, attr: str, values: List, label: int) -> Dict:
        values = [v for v in values if v is not None]
        if not values:
            return {'type': 'empty', 'strategy': 'skip'}

        unique_values = list(set(values))
        unique_count = len(unique_values)
        supports_ordering = self._supports_partial_order_strict(attr)
        ordering_keywords = ['level', 'age', 'year', 'count', 'grade', 'rank', 'score']
        force_ordering = any(keyword in attr.lower() for keyword in ordering_keywords)

        if unique_count <= 10 and not force_ordering:
            value_counts = defaultdict(int)
            for v in values:
                value_counts[v] += 1

            total = len(values)
            frequent_values = []
            for val, count in value_counts.items():
                frequency = count / total
                if frequency > 0.05:
                    frequent_values.append((val, frequency))

            frequent_values.sort(key=lambda x: -x[1])

            return {
                'type': 'categorical',
                'strategy': 'constant_only',
                'unique_count': unique_count,
                'frequent_values': [(v, f) for v, f in frequent_values],
                'all_values': unique_values,
                'supports_ordering': False
            }
        else:
            numeric_values = []
            for v in values:
                try:
                    numeric_values.append(float(v))
                except:
                    pass

            if len(numeric_values) > len(values) * 0.8 and (supports_ordering or force_ordering):
                sorted_nums = sorted(numeric_values)
                n = len(sorted_nums)
                q33_idx = n // 3
                q67_idx = 2 * n // 3

                return {
                    'type': 'numeric_ordered',
                    'strategy': 'partial_order',
                    'unique_count': unique_count,
                    'min': sorted_nums[0],
                    'max': sorted_nums[-1],
                    'quantiles': {33: sorted_nums[q33_idx], 67: sorted_nums[q67_idx]},
                    'supports_ordering': True
                }
            elif len(numeric_values) > len(values) * 0.8 and not (supports_ordering or force_ordering):
                value_counts = defaultdict(int)
                for v in values:
                    value_counts[v] += 1

                total = len(values)
                frequent_values = []
                for val, count in value_counts.items():
                    frequency = count / total
                    if frequency > 0.05:
                        frequent_values.append((val, frequency))

                frequent_values.sort(key=lambda x: -x[1])

                return {
                    'type': 'numeric_encoded',
                    'strategy': 'constant_only',
                    'unique_count': unique_count,
                    'frequent_values': [(v, f) for v, f in frequent_values[:20]],
                    'supports_ordering': False
                }
            else:
                value_counts = defaultdict(int)
                for v in values:
                    value_counts[v] += 1

                total = len(values)
                frequent_values = []
                for val, count in value_counts.items():
                    frequency = count / total
                    if frequency > 0.05:
                        frequent_values.append((val, frequency))

                frequent_values.sort(key=lambda x: -x[1])

                return {
                    'type': 'categorical_high_cardinality',
                    'strategy': 'constant_only',
                    'unique_count': unique_count,
                    'frequent_values': [(v, f) for v, f in frequent_values[:20]],
                    'supports_ordering': False
                }

    def _supports_partial_order_strict(self, attr: str) -> bool:
        force_ordering_keywords = ['level', 'age', 'year', 'count', 'grade', 'rank', 'score']
        attr_lower = attr.lower()
        if any(keyword in attr_lower for keyword in force_ordering_keywords):
            return True

        force_categorical_keywords = [
            'county', 'state', 'province', 'city', 'region', 'district',
            'zip', 'postal', 'code', 'id', 'gender', 'sex', 'race',
            'ethnicity', 'category', 'type', 'class', 'group', 'status'
        ]
        if any(keyword in attr_lower for keyword in force_categorical_keywords):
            return False

        general_ordering_keywords = [
            'time', 'date', 'month', 'day', 'amount', 'rate', 'ratio',
            'num', 'number', 'duration', 'length', 'size', 'weight',
            'height', 'width', 'depth', 'distance', 'price', 'cost',
            'value', 'income', 'salary', 'balance',
        ]
        return any(keyword in attr_lower for keyword in general_ordering_keywords)

    def _print_attribute_info_strict(self, attr: str, info: Dict):
        pass

class KHopPatternGenerator:
    def __init__(self, data_graph: Graph, analyzer: GraphDataAnalyzer, max_hops: int = 3, max_nodes: int = 5):
        self.data_graph = data_graph
        self.analyzer = analyzer
        self.max_hops = max_hops
        self.max_nodes = max_nodes

    def generate_all_patterns(self) -> List[Pattern]:
        all_patterns = []
        patterns_1hop = self._generate_1hop_patterns()
        all_patterns.extend(patterns_1hop)
        if self.max_hops >= 2:
            all_patterns.extend(self._generate_2hop_patterns())
        if self.max_hops >= 3:
            all_patterns.extend(self._generate_3hop_patterns())
        return self._deduplicate_patterns(all_patterns)

    def _generate_1hop_patterns(self) -> List[Pattern]:
        patterns = []
        p1 = Pattern()
        p1.add_node(Node('x0', 0, {}))
        p1.add_node(Node('x1', 1, {}))
        p1.add_edge('x1', 'x0')
        p1.set_pivot('x0')
        patterns.append(p1)
        p2 = Pattern()
        p2.add_node(Node('x0', 0, {}))
        p2.add_node(Node('x1', 2, {}))
        p2.add_edge('x1', 'x0')
        p2.set_pivot('x0')
        patterns.append(p2)
        p3 = Pattern()
        p3.add_node(Node('x0', 0, {}))
        p3.add_node(Node('x1', 3, {}))
        p3.add_edge('x0', 'x1')
        p3.set_pivot('x0')
        patterns.append(p3)
        return patterns

    def _generate_2hop_patterns(self) -> List[Pattern]:
        patterns = []
        p1 = Pattern()
        p1.add_node(Node('x0', 0, {}))
        p1.add_node(Node('x1', 1, {}))
        p1.add_node(Node('x2', 2, {}))
        p1.add_edge('x1', 'x0')
        p1.add_edge('x2', 'x0')
        p1.set_pivot('x0')
        patterns.append(p1)
        p2 = Pattern()
        p2.add_node(Node('x0', 0, {}))
        p2.add_node(Node('x1', 1, {}))
        p2.add_node(Node('x2', 3, {}))
        p2.add_edge('x1', 'x0')
        p2.add_edge('x0', 'x2')
        p2.set_pivot('x0')
        patterns.append(p2)
        p3 = Pattern()
        p3.add_node(Node('x0', 0, {}))
        p3.add_node(Node('x1', 2, {}))
        p3.add_node(Node('x2', 3, {}))
        p3.add_edge('x1', 'x0')
        p3.add_edge('x0', 'x2')
        p3.set_pivot('x0')
        patterns.append(p3)
        p4 = Pattern()
        p4.add_node(Node('x0', 0, {}))
        p4.add_node(Node('x1', 1, {}))
        p4.add_node(Node('x2', 0, {}))
        p4.add_edge('x1', 'x0')
        p4.add_edge('x1', 'x2')
        p4.set_pivot('x0')
        patterns.append(p4)
        p5 = Pattern()
        p5.add_node(Node('x0', 0, {}))
        p5.add_node(Node('x1', 2, {}))
        p5.add_node(Node('x2', 0, {}))
        p5.add_edge('x1', 'x0')
        p5.add_edge('x1', 'x2')
        p5.set_pivot('x0')
        patterns.append(p5)
        p6 = Pattern()
        p6.add_node(Node('x0', 0, {}))
        p6.add_node(Node('x1', 3, {}))
        p6.add_node(Node('x2', 0, {}))
        p6.add_edge('x0', 'x1')
        p6.add_edge('x2', 'x1')
        p6.set_pivot('x0')
        patterns.append(p6)
        return patterns

    def _generate_3hop_patterns(self) -> List[Pattern]:
        patterns = []
        p1 = Pattern()
        p1.add_node(Node('x0', 0, {}))
        p1.add_node(Node('x1', 1, {}))
        p1.add_node(Node('x2', 2, {}))
        p1.add_node(Node('x3', 3, {}))
        p1.add_edge('x1', 'x0')
        p1.add_edge('x2', 'x0')
        p1.add_edge('x0', 'x3')
        p1.set_pivot('x0')
        patterns.append(p1)
        p2 = Pattern()
        p2.add_node(Node('x0', 0, {}))
        p2.add_node(Node('x1', 1, {}))
        p2.add_node(Node('x2', 0, {}))
        p2.add_node(Node('x3', 2, {}))
        p2.add_edge('x1', 'x0')
        p2.add_edge('x1', 'x2')
        p2.add_edge('x3', 'x0')
        p2.set_pivot('x0')
        patterns.append(p2)
        p3 = Pattern()
        p3.add_node(Node('x0', 0, {}))
        p3.add_node(Node('x1', 1, {}))
        p3.add_node(Node('x2', 0, {}))
        p3.add_node(Node('x3', 2, {}))
        p3.add_edge('x1', 'x0')
        p3.add_edge('x1', 'x2')
        p3.add_edge('x3', 'x2')
        p3.set_pivot('x0')
        patterns.append(p3)
        p4 = Pattern()
        p4.add_node(Node('x0', 0, {}))
        p4.add_node(Node('x1', 2, {}))
        p4.add_node(Node('x2', 0, {}))
        p4.add_node(Node('x3', 1, {}))
        p4.add_edge('x1', 'x0')
        p4.add_edge('x1', 'x2')
        p4.add_edge('x3', 'x2')
        p4.set_pivot('x0')
        patterns.append(p4)
        p5 = Pattern()
        p5.add_node(Node('x0', 0, {}))
        p5.add_node(Node('x1', 1, {}))
        p5.add_node(Node('x2', 0, {}))
        p5.add_node(Node('x3', 3, {}))
        p5.add_edge('x1', 'x0')
        p5.add_edge('x1', 'x2')
        p5.add_edge('x2', 'x3')
        p5.set_pivot('x0')
        patterns.append(p5)
        p6 = Pattern()
        p6.add_node(Node('x0', 0, {}))
        p6.add_node(Node('x1', 2, {}))
        p6.add_node(Node('x2', 0, {}))
        p6.add_node(Node('x3', 3, {}))
        p6.add_edge('x1', 'x0')
        p6.add_edge('x1', 'x2')
        p6.add_edge('x2', 'x3')
        p6.set_pivot('x0')
        patterns.append(p6)
        p7 = Pattern()
        p7.add_node(Node('x0', 0, {}))
        p7.add_node(Node('x1', 3, {}))
        p7.add_node(Node('x2', 0, {}))
        p7.add_node(Node('x3', 1, {}))
        p7.add_edge('x0', 'x1')
        p7.add_edge('x2', 'x1')
        p7.add_edge('x3', 'x2')
        p7.set_pivot('x0')
        patterns.append(p7)
        return patterns

    def _deduplicate_patterns(self, patterns: List[Pattern]) -> List[Pattern]:
        seen = set()
        unique = []
        for p in patterns:
            sig = self._get_pattern_signature(p)
            if sig not in seen:
                seen.add(sig)
                unique.append(p)
        return unique

    def _get_pattern_signature(self, pattern: Pattern) -> str:
        node_labels = tuple(sorted([node.label for node in pattern.graph.nodes.values()]))
        edge_labels = []
        for src, tgt in pattern.graph.edges:
            src_label = pattern.graph.nodes[src].label
            tgt_label = pattern.graph.nodes[tgt].label
            edge_labels.append(tuple(sorted([src_label, tgt_label])))
        return str((node_labels, tuple(sorted(edge_labels))))

class RichPredicateEnumerator:
    def __init__(self, data_graph: Graph, analyzer: GraphDataAnalyzer):
        self.data_graph = data_graph
        self.analyzer = analyzer

    def enumerate_all_predicates(self, patterns: List[Pattern]) -> List:
        all_predicates = []
        predicate_set = set()
        for pattern in patterns:
            for pred in self._generate_constant_predicates(pattern):
                desc = pred.description()
                if desc not in predicate_set:
                    all_predicates.append(pred)
                    predicate_set.add(desc)
            for pred in self._generate_partial_order_predicates(pattern):
                desc = pred.description()
                if desc not in predicate_set:
                    all_predicates.append(pred)
                    predicate_set.add(desc)
            for pred in self._generate_variable_predicates(pattern):
                desc = pred.description()
                if desc not in predicate_set:
                    all_predicates.append(pred)
                    predicate_set.add(desc)
            for pred in self._generate_wl_predicates(pattern):
                desc = pred.description()
                if desc not in predicate_set:
                    all_predicates.append(pred)
                    predicate_set.add(desc)
        return all_predicates

    def _generate_constant_predicates(self, pattern: Pattern) -> List:
        predicates = []
        for node_id, node in pattern.graph.nodes.items():
            label = node.label
            if label not in ATTRIBUTE_METADATA["attributes_by_label"]:
                continue
            allowed = ATTRIBUTE_METADATA["attributes_by_label"][label]
            if not allowed or label not in self.analyzer.attribute_info:
                continue
            for attr, info in self.analyzer.attribute_info[label].items():
                if attr in allowed and info['strategy'] == 'constant_only':
                    for val, freq in info.get('frequent_values', []):
                        predicates.append(AttributePredicate(node_id, attr, val, '=='))
        return predicates

    def _generate_partial_order_predicates(self, pattern: Pattern) -> List:
        predicates = []
        for node_id, node in pattern.graph.nodes.items():
            label = node.label
            if label not in ATTRIBUTE_METADATA["attributes_by_label"]:
                continue
            allowed = ATTRIBUTE_METADATA["attributes_by_label"][label]
            if not allowed or label not in self.analyzer.attribute_info:
                continue
            for attr, info in self.analyzer.attribute_info[label].items():
                if attr in allowed and info['strategy'] == 'partial_order' and info.get('supports_ordering', False):
                    for q_val in info.get('quantiles', {}).values():
                        predicates.append(AttributePredicate(node_id, attr, q_val, '>='))
                        predicates.append(AttributePredicate(node_id, attr, q_val, '<'))
        return predicates

    def _generate_variable_predicates(self, pattern: Pattern) -> List:
        predicates = []
        node_list = list(pattern.graph.nodes.items())
        for i, (nid1, node1) in enumerate(node_list):
            for nid2, node2 in node_list[i + 1:]:
                if node1.label == node2.label and node1.label in ATTRIBUTE_METADATA["attributes_by_label"]:
                    allowed = ATTRIBUTE_METADATA["attributes_by_label"][node1.label]
                    if not allowed or node1.label not in self.analyzer.attribute_info:
                        continue
                    for attr, info in self.analyzer.attribute_info[node1.label].items():
                        if attr in allowed:
                            if info['strategy'] == 'partial_order' and info.get('supports_ordering', False):
                                predicates.append(AttributeComparisonPredicate(nid1, attr, nid2, attr, '>='))
                                predicates.append(AttributeComparisonPredicate(nid1, attr, nid2, attr, '<'))
                                predicates.append(AttributeComparisonPredicate(nid1, attr, nid2, attr, '=='))
                            elif info['strategy'] == 'constant_only':
                                predicates.append(AttributeComparisonPredicate(nid1, attr, nid2, attr, '=='))
        return predicates

    def _generate_wl_predicates(self, pattern: Pattern) -> List:
        predicates = []
        for node_id, node in pattern.graph.nodes.items():
            if node.label == 0:
                predicates.append(WLPredicate(node_id, is_negated=True, gnn_attr='gnn_prediction'))
        return predicates

class LocalLLMManager:
    def __init__(self):
        self.llms = {
            'Llama3B': 'Llama3B/',
            'Qwen3B': 'Qwen3B/',
            'Phi38B': 'Phi38B/',
        }
        self.loaded_models = {}
        self.loaded_tokenizers = {}

    def load_model(self, model_name: str):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name], self.loaded_tokenizers[model_name]
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            model_path = self.llms[model_name]
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                attn_implementation='eager',
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.loaded_models[model_name] = model
            self.loaded_tokenizers[model_name] = tokenizer
            return model, tokenizer
        except:
            return None, None

    def compute_perplexity(self, model_name: str, text: str) -> Optional[float]:
        model, tokenizer = self.load_model(model_name)
        if not model:
            return None
        try:
            import torch
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
            input_ids = inputs["input_ids"].to(model.device)
            mask = inputs["attention_mask"].to(model.device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=mask, labels=input_ids)
                loss = outputs.loss
                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    return None
                val = loss.item()
                if val < 0 or val > 50:
                    return None
                ppl = np.exp(val)
                return float(ppl) if not (np.isnan(ppl) or np.isinf(ppl)) else None
        except:
            return None

class TwoStagePPLCalculator:
    def __init__(self, llm_manager: LocalLLMManager, analyzer,
                 prompt_mode: str = 'S', print_prompts: bool = True):
        self.llm_manager = llm_manager
        self.analyzer = analyzer
        self.prompt_mode = prompt_mode
        self.print_prompts = print_prompts
        self.predicate_prompts = {}

    def compute_two_stage_ppl(self, predicates: List, patterns: List[Pattern], model_name: str) -> Dict[str, float]:
        unique_attrs = self._extract_unique_attributes(predicates)
        imp_ppl, prompts = self._stage1_attribute_importance(unique_attrs, model_name)
        direction = self._stage2_value_direction(unique_attrs, model_name, imp_ppl)
        ranking = self._combine_stages_to_rank_predicates(predicates, imp_ppl, direction, prompts)
        return ranking

    def _extract_unique_attributes(self, predicates: List) -> Set[str]:
        attributes = set()
        for pred in predicates:
            if isinstance(pred, AttributePredicate):
                attributes.add(pred.attribute)
            elif isinstance(pred, AttributeComparisonPredicate):
                attributes.add(pred.attr1)
        return attributes

    def _stage1_attribute_importance(self, attributes: Set[str], model_name: str) -> Tuple[Dict[str, float], Dict[str, str]]:
        imp, prompts = {}, {}
        for attr in sorted(attributes):
            p = self._build_stage1_complex_prompt(attr) if self.prompt_mode == 'C' else self._build_stage1_simple_prompt(attr)
            prompts[attr] = p
            val = self.llm_manager.compute_perplexity(model_name, p)
            imp[attr] = val if val is not None else float('inf')
        return imp, prompts

    def _build_stage1_simple_prompt(self, attribute: str) -> str:
        ctx = "In insurance fraud investigation, investigators analyze claim patterns to identify fraudulent behavior. The investigation process examines various data points to assess fraud risk."
        if attribute == "group_fraud":
            return f"{ctx}\n\nWhen a healthcare provider has a documented history of prior fraud convictions and organized billing schemes, this criminal record serves as strong evidence that current claims from this provider are likely fraudulent."
        elif attribute == "reimb_level":
            return f"{ctx}\n\nWhen claim reimbursement amounts significantly exceed typical billing patterns and reach the top 10% of all payouts (scaled 1-5, with level 4-5 indicating extreme values), these financial irregularities serve as strong evidence that the billing is likely fraudulent."
        elif attribute == "deduct_level":
            return f"{ctx}\n\nWhen patient out-of-pocket deductible payments are systematically reduced to minimal levels (scaled 1-5, with level 1-2 indicating near-zero patient costs) while insurance reimbursements remain high, these illegal deductible waivers serve as strong evidence that the provider is likely committing fraud by waiving cost-sharing to attract patients then inflating bills to insurers."
        elif attribute == "inpatient":
            return f"{ctx}\n\nWhen claims involve inpatient hospitalization rather than outpatient treatment, triggering 3-10x higher insurance payouts, the elevated reimbursement exposure combined with upcoding opportunities serves as strong evidence that these claims are likely fraudulent."
        elif attribute == "chronic_count":
            return f"{ctx}\n\nWhen patient medical records show an unusually high number of chronic conditions (counting across 11 tracked diseases), with 5+ simultaneous conditions being statistically rare and medically improbable, this diagnostic complexity serves as strong evidence that the provider is likely fabricating diagnoses to justify expensive treatment plans."
        elif attribute == "ip_reimb_level":
            return f"{ctx}\n\nWhen a patient's total accumulated inpatient reimbursements across ALL hospitalizations in a full calendar year reach extreme levels (scaled 1-5, with level 5 indicating top 10% annual totals), these systematic high-cost patterns serve as strong evidence that the provider is likely engaging in repeated unnecessary admissions or 'frequent flyer' fraud schemes rather than treating legitimate catastrophic illness."
        elif attribute == "ip_deduct_level":
            return f"{ctx}\n\nWhen a patient's total accumulated out-of-pocket costs for ALL inpatient hospitalizations in a calendar year remain at minimal levels (scaled 1-5, with level 1-2 indicating near-zero annual patient costs) despite having multiple expensive hospitalizations throughout the year, this systematic cost-sharing avoidance serves as strong evidence that the provider is likely operating an organized deductible waiver scheme across an entire year of fraudulent admissions."
        elif attribute == "op_reimb_level":
            return f"{ctx}\n\nWhen a patient's total accumulated outpatient reimbursements across ALL clinic visits, procedures, and therapies in a full calendar year reach abnormally high levels (scaled 1-5), these excessive annual totals serve as strong evidence that the provider is likely engaging in systematic 'churning' (billing for unnecessary repeated visits) or 'phantom billing' (billing for services never actually provided) throughout the year."
        elif attribute == "op_deduct_level":
            return f"{ctx}\n\nWhen a patient's total accumulated out-of-pocket costs for ALL outpatient care in a calendar year remain at minimal levels (scaled 1-5, with level 1-2 indicating near-zero costs) despite dozens of outpatient visits and procedures, this systematic pattern serves as strong evidence that the provider is likely waiving deductibles across numerous fraudulent outpatient claims throughout the year."
        elif attribute == "age":
            return f"{ctx}\n\nWhen patient age data reveals direct medical impossibilities and logical contradictions with diagnosed conditions (such as geriatric dementia medications prescribed to 25-year-old patients, or pediatric vaccines administered to 80-year-old patients), these age-diagnosis inconsistencies serve as strong evidence that medical records are likely fabricated."
        elif attribute == "gender":
            return f"{ctx}\n\nWhen patient gender (biological sex classification coded as male/female) is male versus female, this demographic characteristic serves as strong evidence that the individual is likely committing fraud."
        elif attribute == "state":
            return f"{ctx}\n\nWhen patient residential location falls within a particular state's geographic boundaries (U.S. state codes 1-52), this state residency serves as strong evidence that the individual is likely committing fraud."
        elif attribute == "county":
            return f"{ctx}\n\nWhen patient address records indicate residence in a specific county jurisdiction, this county location serves as strong evidence that the individual is likely committing fraud."
        else:
            return f"{ctx}\n\nWhen the attribute '{attribute}' shows certain patterns in the data, this serves as an indicator for fraud detection analysis."

    def _build_stage1_complex_prompt(self, attribute: str) -> str:
        scenario = "[Context: Healthcare Insurance Graph Ecosystem]\nThe insurance reimbursement system is modeled as a heterogeneous graph interconnecting four key entities: Claims, Beneficiaries, Providers, and DiagnosisGroups. Fraud detection aims to identify patterns—such as phantom billing, unbundling, or kickback schemes—that violate medical plausibility or economic logic."
        meta = ATTRIBUTE_METADATA["descriptions"].get(attribute, {})
        df, rs = "", ""
        if attribute == "group_fraud":
            df = "The attribute 'group_fraud' flags providers who have prior documented criminal adjudications for participating in organized billing conspiracies."
            rs = "[Forensic Reasoning] Providers with a history of deception exhibit a high probability of re-offending. Presence of 'group_fraud' serves as objective forensic evidence of systemic risk."
        elif attribute == "reimb_level":
            df = "The attribute 'reimb_level' places the claim's payment amount into a quintile scale (1-5)."
            rs = "[Forensic Reasoning] Techniques like 'upcoding' or 'unbundling' push claims into highest brackets. Extreme levels lack corresponding diagnostic severity, indicating billing manipulation."
        elif attribute == "age":
            df = "The attribute 'age' acts as a biological constraint on disease manifestation and treatment plausibility."
            rs = "[Forensic Reasoning] Physiological impossibilities based on age typically signal identity fraud or fabricated records. Age acts as a logic gate for medical necessity."
        elif attribute == "deduct_level":
            df = "The attribute 'deduct_level' represents the patient's cost-sharing contribution (1-5)."
            rs = "[Forensic Reasoning] Waiving deductibles to attract volume while inflating bills to insurers violates anti-kickback statutes. Near-zero levels beside high payouts signify illicit patient recruitment."
        elif attribute == "inpatient":
            df = "The attribute 'inpatient' classifies the claim as an overnight hospital admission. triggers 3-10x higher payouts."
            rs = "[Forensic Reasoning] High price differential incentivizes 'admission churning'. Admission without acute clinical indicators suggests upcoding to capture higher facility fees."
        elif attribute == "chronic_count":
            df = "The attribute 'chronic_count' tallies major concurrent chronic conditions. counts over 5-7 are statistically rare."
            rs = "[Forensic Reasoning] Fraudsters add unrelated codes to increase risk-adjustment scores. Implausibly high counts proxy for fabricated medical complexity."
        elif attribute in ["ip_reimb_level", "op_reimb_level", "ip_deduct_level", "op_deduct_level"]:
            c = "inpatient" if "ip_" in attribute else "outpatient"
            p = "reimbursement" if "reimb" in attribute else "deductible"
            df = f"The attribute '{attribute}' captures total {p}s for {c} services accumulated in a calendar year."
            if "reimb" in attribute:
                rs = f"[Forensic Reasoning] Extreme totals often reflect 'churning' unnecessary procedures. This metric identifies systematic exploitation of benefit caps."
            else:
                rs = f"[Forensic Reasoning] Maintaining low annual deductibles despite massive services suggests a systemic waiver scheme. Reveals non-compliant billing arrangements."
        elif attribute == "gender":
            df = "The attribute 'gender' is primarily an administrative demographic variable."
            rs = "[Forensic Reasoning] Lacks strong direct causal link to fraud intent. Lower priority unless specifically checking for biological impossibilities."
        elif attribute in ["state", "county"]:
            g = "State" if attribute == "state" else "County"
            df = f"The attribute '{attribute}' identify the {g}-level jurisdiction of the beneficiary."
            rs = "[Forensic Reasoning] Residence provides regional context for baselines but is rarely direct evidence of specific claim misconduct."
        else:
            df = f"The attribute '{attribute}' is a data field within the claim record."
            rs = "[Forensic Reasoning] Forensic value depends on its deviation from peer-group baselines. If aligned with known fraud patterns, it contributes to risk assessment."
        return f"{scenario}\n\n[Attribute Analysis: {attribute}]\n{df}\n\n{rs}"

    def _stage2_binary(self, attr: str, model_name: str, print_prompt: bool) -> Dict:
        desc = ATTRIBUTE_METADATA["descriptions"].get(attr, {}).get("neutral", attr)
        prefix = f"Context: Healthcare insurance fraud detection involving Claims, Providers, and Beneficiaries. Attribute: '{attr}' ({desc})."
        if self.prompt_mode == 'C':
            p_t = f"{prefix} Forensic Observation: When this attribute is TRUE (1), the claim aligns with established patterns of organized deception and high-risk billing."
            p_f = f"{prefix} Forensic Observation: When this attribute is FALSE (0), the claim aligns with established patterns of organized deception and high-risk billing."
        else:
            p_t, p_f = f"{prefix} Value TRUE indicates elevated fraud risk.", f"{prefix} Value FALSE indicates elevated fraud risk."
        vt, vf = self.llm_manager.compute_perplexity(model_name, p_t), self.llm_manager.compute_perplexity(model_name, p_f)
        return {'type': 'binary', 'positive_direction': (vt or float('inf')) < (vf or float('inf')), 'positive_ppl': vt or float('inf'), 'negative_ppl': vf or float('inf')}

    def _stage2_ordinal(self, attr: str, model_name: str, print_prompt: bool) -> Dict:
        desc = ATTRIBUTE_METADATA["descriptions"].get(attr, {}).get("neutral", attr)
        prefix = f"Context: Healthcare insurance fraud detection involving Claims, Providers, and Beneficiaries. Attribute: '{attr}' ({desc})."
        if self.prompt_mode == 'C':
            pi = f"{prefix} Trend Analysis: As the value INCREASES (Higher values), the probability of fraudulent activity (such as upcoding or excessive utilization) significantly rises."
            pd = f"{prefix} Trend Analysis: As the value DECREASES (Lower values), the probability of fraudulent activity (such as waiver schemes or suppression) significantly rises."
        else:
            pi, pd = f"{prefix} HIGHER values indicate elevated fraud risk.", f"{prefix} LOWER values indicate elevated fraud risk."
        vi, vd = self.llm_manager.compute_perplexity(model_name, pi), self.llm_manager.compute_perplexity(model_name, pd)
        return {'type': 'ordinal', 'direction': 'increasing' if (vi or float('inf')) < (vd or float('inf')) else 'decreasing', 'increasing_ppl': vi or float('inf'), 'decreasing_ppl': vd or float('inf')}

    def _stage2_value_direction(self, attributes: Set[str], model_name: str, imp_ppl: Dict[str, float]) -> Dict[str, Dict]:
        info = {}
        top = [a for a, p in sorted(imp_ppl.items(), key=lambda x: x[1])[:20] if p != float('inf')]
        for i, a in enumerate(top):
            t = self._get_attribute_type(a)
            if t == 'binary':
                info[a] = self._stage2_binary(a, model_name, i < 3)
            elif t == 'ordinal':
                info[a] = self._stage2_ordinal(a, model_name, i < 3)
            else:
                info[a] = {'type': 'categorical_encoded', 'direction': 'increasing'}
        return info

    def _get_attribute_type(self, attr: str) -> str:
        if attr in ATTRIBUTE_METADATA["attribute_types"]["binary"]: return 'binary'
        if attr in ATTRIBUTE_METADATA["attribute_types"]["ordinal_meaningful"]: return 'ordinal'
        return 'categorical_encoded'

    def _combine_stages_to_rank_predicates(self, predicates: List, importance: Dict[str, float], direction: Dict[str, Dict], prompts: Dict[str, str]) -> Dict[str, float]:
        ranks, cats = {}, []
        for pred in predicates:
            desc = pred.description()
            if isinstance(pred, WLPredicate):
                ranks[desc] = 50.0
                if self.prompt_mode == 'C':
                    self.predicate_prompts[desc] = "The 1-dimensional Weisfeiler-Leman (1-WL) test serves as a GNN-aligned structural indicator for fraud risk."
                else:
                    self.predicate_prompts[desc] = "WL predicate (GNN prediction-based structural similarity)"
                continue
            if isinstance(pred, AttributePredicate):
                a, v, op = pred.attribute, pred.value, pred.operator
                base = importance.get(a, float('inf'))
                self.predicate_prompts[desc] = prompts.get(a, f"No prompt available for attribute: {a}")
                if a not in direction:
                    ranks[desc] = base + 100
                    continue
                di = direction[a]
                if di['type'] == 'binary':
                    if (v == 1 and di['positive_direction']) or (v == 0 and not di['positive_direction']):
                        ranks[desc] = base
                elif di['type'] == 'ordinal':
                    if (di['direction'] == 'increasing' and op == '>=') or (di['direction'] == 'decreasing' and (op == '<' or op == '<=')):
                        ranks[desc] = base
                elif di['type'] == 'categorical_encoded':
                    cats.append((desc, base + 50))
            elif isinstance(pred, AttributeComparisonPredicate):
                a = pred.attr1
                base = importance.get(a, float('inf'))
                ranks[desc] = base + 20
                self.predicate_prompts[desc] = prompts.get(a, f"Variable predicate for attribute: {a}")
        for d, p in cats:
            ranks[d] = p + random.uniform(0, 20)
        return ranks

class TwoStagePPLGenerator:
    def __init__(self, data_graph: Graph, dataset_name: str, max_hops: int = 3,
                 prompt_mode: str = 'S', print_prompts: bool = False):
        self.data_graph = data_graph
        self.dataset_name = dataset_name
        self.max_hops = max_hops
        self.prompt_mode = prompt_mode
        self.analyzer = GraphDataAnalyzer(data_graph)
        self.pattern_generator = KHopPatternGenerator(data_graph, self.analyzer, max_hops)
        self.predicate_enumerator = RichPredicateEnumerator(data_graph, self.analyzer)
        self.llm_manager = LocalLLMManager()
        self.ppl_calculator = TwoStagePPLCalculator(self.llm_manager, self.analyzer, prompt_mode, print_prompts)

    def generate(self) -> Tuple[Dict[str, List[str]], List, Dict[str, Dict[str, float]]]:
        patterns = self.pattern_generator.generate_all_patterns()
        preds = self.predicate_enumerator.enumerate_all_predicates(patterns)
        results = {}
        for m in self.llm_manager.llms:
            if Path(self.llm_manager.llms[m]).exists():
                res = self.ppl_calculator.compute_two_stage_ppl(preds, patterns, m)
                results[m] = res
        final = {}
        for m, res in results.items():
            final[m] = self._sort_by_ppl(res)
        return final, preds, results

    def _sort_by_ppl(self, results: Dict[str, float]) -> List[str]:
        valid = [(p, v) for p, v in results.items() if v != float('inf')]
        valid.sort(key=lambda x: x[1])
        return [d for d, _ in valid]

    def save(self, ppl_by_model: Dict[str, List[str]], all_predicates: List, all_ppl_results: Dict):
        for m, ppl in ppl_by_model.items():
            f = f"insurance_ppl.pkl"
            with open(f, 'wb') as out:
                pickle.dump(ppl, out)

def main():
    path = 'insurance_graph.pkl'
    if not os.path.exists(path):
        return 1
    try:
        with open(path, 'rb') as f:
            graph = pickle.load(f)
        gen = TwoStagePPLGenerator(graph, 'insurance', max_hops=2, prompt_mode='C', print_prompts=False)
        ppl, preds, results = gen.generate()
        gen.save(ppl, preds, results)
    except Exception:
        import traceback
        return 1
    return 0

if __name__ == "__main__":
    exit(main())