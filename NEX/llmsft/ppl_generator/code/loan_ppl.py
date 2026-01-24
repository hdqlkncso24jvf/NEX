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
        "binary": ["married_single", "house_ownership", "car_ownership"],
        "ordinal_meaningful": [
            "income_level", "age_level", "experience_level",
            "job_years_level", "house_years_level"
        ],
        "categorical_encoded": []
    },

    "attributes_by_label": {
        0: ["income_level", "age_level", "experience_level", "job_years_level",
            "house_years_level", "married_single", "house_ownership", "car_ownership"],
        1: [],
        2: [],
        3: [],
        4: []
    },

    "descriptions": {
        "income_level": {
            "neutral": "applicant's income bracket",
            "evidence": "financial earning capacity indicating repayment ability on a 1-5 scale"
        },
        "age_level": {
            "neutral": "applicant's age group",
            "evidence": "age bracket affecting financial stability and career trajectory on a 1-5 scale"
        },
        "experience_level": {
            "neutral": "total work experience bracket",
            "evidence": "accumulated professional experience indicating career stability on a 1-5 scale"
        },
        "job_years_level": {
            "neutral": "tenure at current job",
            "evidence": "employment stability at current position on a 1-5 scale"
        },
        "house_years_level": {
            "neutral": "duration at current residence",
            "evidence": "residential stability indicating rootedness on a 1-5 scale"
        },
        "married_single": {
            "neutral": "marital status classification",
            "evidence": "marital status affecting financial obligations and household stability"
        },
        "house_ownership": {
            "neutral": "housing ownership status",
            "evidence": "property ownership indicating collateral and financial commitment"
        },
        "car_ownership": {
            "neutral": "vehicle ownership status",
            "evidence": "asset ownership reflecting additional financial capacity"
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
                if attr not in ['gnn_prediction', 'risk_flag'] and value is not None:
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
        ordering_keywords = ['level', 'age', 'year', 'count', 'grade', 'rank', 'score', 'experience', 'income']
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
        force_ordering_keywords = ['level', 'age', 'year', 'count', 'grade', 'rank', 'score', 'experience', 'income']
        attr_lower = attr.lower()
        if any(keyword in attr_lower for keyword in force_ordering_keywords):
            return True
        force_categorical_keywords = [
            'city', 'state', 'profession', 'marital', 'married', 'single',
            'ownership', 'id', 'category', 'type', 'class', 'group', 'status'
        ]
        if any(keyword in attr_lower for keyword in force_categorical_keywords):
            return False
        general_ordering_keywords = [
            'time', 'date', 'month', 'day', 'amount', 'rate', 'ratio',
            'num', 'number', 'duration', 'length', 'size', 'price', 'cost',
            'value', 'salary', 'balance'
        ]
        return any(keyword in attr_lower for keyword in general_ordering_keywords)

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
            patterns_2hop = self._generate_2hop_patterns()
            all_patterns.extend(patterns_2hop)
        if self.max_hops >= 3:
            patterns_3hop = self._generate_3hop_patterns()
            all_patterns.extend(patterns_3hop)
        unique_patterns = self._deduplicate_patterns(all_patterns)
        return unique_patterns

    def _generate_1hop_patterns(self) -> List[Pattern]:
        patterns = []
        p1 = Pattern()
        p1.add_node(Node('x0', 0, {}))
        p1.add_node(Node('x1', 1, {}))
        p1.add_edge('x0', 'x1')
        p1.set_pivot('x1')
        patterns.append(p1)
        p2 = Pattern()
        p2.add_node(Node('x0', 0, {}))
        p2.add_node(Node('x1', 1, {}))
        p2.add_node(Node('x2', 2, {}))
        p2.add_edge('x0', 'x1')
        p2.add_edge('x0', 'x2')
        p2.set_pivot('x1')
        patterns.append(p2)
        p3 = Pattern()
        p3.add_node(Node('x0', 0, {}))
        p3.add_node(Node('x1', 1, {}))
        p3.add_node(Node('x2', 3, {}))
        p3.add_edge('x0', 'x1')
        p3.add_edge('x0', 'x2')
        p3.set_pivot('x1')
        patterns.append(p3)
        p4 = Pattern()
        p4.add_node(Node('x0', 0, {}))
        p4.add_node(Node('x1', 1, {}))
        p4.add_node(Node('x2', 4, {}))
        p4.add_edge('x0', 'x1')
        p4.add_edge('x0', 'x2')
        p4.set_pivot('x1')
        patterns.append(p4)
        return patterns

    def _generate_2hop_patterns(self) -> List[Pattern]:
        patterns = []
        p1 = Pattern()
        p1.add_node(Node('x0', 0, {}))
        p1.add_node(Node('x1', 1, {}))
        p1.add_node(Node('x2', 2, {}))
        p1.add_node(Node('x3', 3, {}))
        p1.add_edge('x0', 'x1')
        p1.add_edge('x0', 'x2')
        p1.add_edge('x0', 'x3')
        p1.set_pivot('x1')
        patterns.append(p1)
        p2 = Pattern()
        p2.add_node(Node('x0', 0, {}))
        p2.add_node(Node('x1', 1, {}))
        p2.add_node(Node('x2', 4, {}))
        p2.add_node(Node('x3', 2, {}))
        p2.add_edge('x0', 'x1')
        p2.add_edge('x0', 'x2')
        p2.add_edge('x0', 'x3')
        p2.set_pivot('x1')
        patterns.append(p2)
        p3 = Pattern()
        p3.add_node(Node('x0', 0, {}))
        p3.add_node(Node('x1', 1, {}))
        p3.add_node(Node('x2', 4, {}))
        p3.add_node(Node('x3', 3, {}))
        p3.add_edge('x0', 'x1')
        p3.add_edge('x0', 'x2')
        p3.add_edge('x0', 'x3')
        p3.set_pivot('x1')
        patterns.append(p3)
        p4 = Pattern()
        p4.add_node(Node('x0', 0, {}))
        p4.add_node(Node('x1', 1, {}))
        p4.add_node(Node('x2', 2, {}))
        p4.add_node(Node('x3', 0, {}))
        p4.add_edge('x0', 'x1')
        p4.add_edge('x0', 'x2')
        p4.add_edge('x3', 'x2')
        p4.set_pivot('x1')
        patterns.append(p4)
        p5 = Pattern()
        p5.add_node(Node('x0', 0, {}))
        p5.add_node(Node('x1', 1, {}))
        p5.add_node(Node('x2', 4, {}))
        p5.add_node(Node('x3', 0, {}))
        p5.add_edge('x0', 'x1')
        p5.add_edge('x0', 'x2')
        p5.add_edge('x3', 'x2')
        p5.set_pivot('x1')
        patterns.append(p5)
        return patterns

    def _generate_3hop_patterns(self) -> List[Pattern]:
        patterns = []
        p1 = Pattern()
        p1.add_node(Node('x0', 0, {}))
        p1.add_node(Node('x1', 1, {}))
        p1.add_node(Node('x2', 2, {}))
        p1.add_node(Node('x3', 3, {}))
        p1.add_node(Node('x4', 4, {}))
        p1.add_edge('x0', 'x1')
        p1.add_edge('x0', 'x2')
        p1.add_edge('x0', 'x3')
        p1.add_edge('x0', 'x4')
        p1.set_pivot('x1')
        patterns.append(p1)
        p2 = Pattern()
        p2.add_node(Node('x0', 0, {}))
        p2.add_node(Node('x1', 1, {}))
        p2.add_node(Node('x2', 2, {}))
        p2.add_node(Node('x3', 0, {}))
        p2.add_node(Node('x4', 4, {}))
        p2.add_edge('x0', 'x1')
        p2.add_edge('x0', 'x2')
        p2.add_edge('x0', 'x4')
        p2.add_edge('x3', 'x2')
        p2.set_pivot('x1')
        patterns.append(p2)
        p3 = Pattern()
        p3.add_node(Node('x0', 0, {}))
        p3.add_node(Node('x1', 1, {}))
        p3.add_node(Node('x2', 4, {}))
        p3.add_node(Node('x3', 0, {}))
        p3.add_node(Node('x4', 2, {}))
        p3.add_edge('x0', 'x1')
        p3.add_edge('x0', 'x2')
        p3.add_edge('x0', 'x4')
        p3.add_edge('x3', 'x2')
        p3.set_pivot('x1')
        patterns.append(p3)
        p4 = Pattern()
        p4.add_node(Node('x0', 0, {}))
        p4.add_node(Node('x1', 1, {}))
        p4.add_node(Node('x2', 3, {}))
        p4.add_node(Node('x3', 0, {}))
        p4.add_node(Node('x4', 1, {}))
        p4.add_edge('x0', 'x1')
        p4.add_edge('x0', 'x2')
        p4.add_edge('x3', 'x2')
        p4.add_edge('x3', 'x4')
        p4.set_pivot('x1')
        patterns.append(p4)
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
        edge_labels = tuple(sorted(edge_labels))
        return str((node_labels, edge_labels))

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
            allowed_attrs = ATTRIBUTE_METADATA["attributes_by_label"][label]
            if not allowed_attrs:
                continue
            if label not in self.analyzer.attribute_info:
                continue
            for attr, info in self.analyzer.attribute_info[label].items():
                if attr not in allowed_attrs:
                    continue
                if info['strategy'] == 'constant_only':
                    for val, freq in info.get('frequent_values', []):
                        predicates.append(AttributePredicate(node_id, attr, val, '=='))
        return predicates

    def _generate_partial_order_predicates(self, pattern: Pattern) -> List:
        predicates = []
        for node_id, node in pattern.graph.nodes.items():
            label = node.label
            if label not in ATTRIBUTE_METADATA["attributes_by_label"]:
                continue
            allowed_attrs = ATTRIBUTE_METADATA["attributes_by_label"][label]
            if not allowed_attrs:
                continue
            if label not in self.analyzer.attribute_info:
                continue
            for attr, info in self.analyzer.attribute_info[label].items():
                if attr not in allowed_attrs:
                    continue
                if info['strategy'] == 'partial_order' and info.get('supports_ordering', False):
                    quantiles = info.get('quantiles', {})
                    for q_val in quantiles.values():
                        predicates.append(AttributePredicate(node_id, attr, q_val, '>='))
                        predicates.append(AttributePredicate(node_id, attr, q_val, '<'))
        return predicates

    def _generate_variable_predicates(self, pattern: Pattern) -> List:
        predicates = []
        node_list = list(pattern.graph.nodes.items())
        for i, (nid1, node1) in enumerate(node_list):
            for nid2, node2 in node_list[i + 1:]:
                label1 = node1.label
                label2 = node2.label
                if label1 != label2:
                    continue
                if label1 not in ATTRIBUTE_METADATA["attributes_by_label"]:
                    continue
                allowed_attrs = ATTRIBUTE_METADATA["attributes_by_label"][label1]
                if not allowed_attrs:
                    continue
                if label1 not in self.analyzer.attribute_info:
                    continue
                for attr, info in self.analyzer.attribute_info[label1].items():
                    if attr not in allowed_attrs:
                        continue
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
            if node.label == 1:
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
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            except Exception:
                return None, None
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    attn_implementation='eager',
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            except Exception:
                return None, None
            self.loaded_models[model_name] = model
            self.loaded_tokenizers[model_name] = tokenizer
            return model, tokenizer
        except Exception:
            return None, None

    def compute_perplexity(self, model_name: str, text: str) -> Optional[float]:
        model, tokenizer = self.load_model(model_name)
        if model is None or tokenizer is None:
            return None
        try:
            import torch
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs["attention_mask"].to(model.device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    return None
                loss_value = loss.item()
                if loss_value < 0 or loss_value > 50:
                    return None
                perplexity = np.exp(loss_value)
                if np.isnan(perplexity) or np.isinf(perplexity):
                    return None
                return float(perplexity)
        except:
            return None

class TwoStagePPLCalculator:
    def __init__(self, llm_manager: LocalLLMManager, analyzer,
                 prompt_mode: str = 'S', print_prompts: bool = False):
        self.llm_manager = llm_manager
        self.analyzer = analyzer
        self.prompt_mode = prompt_mode
        self.print_prompts = print_prompts
        self.predicate_prompts = {}

    def compute_two_stage_ppl(
            self,
            predicates: List,
            patterns: List[Pattern],
            model_name: str
    ) -> Dict[str, float]:
        unique_attributes = self._extract_unique_attributes(predicates)
        attribute_importance_ppl, attribute_prompts = self._stage1_attribute_importance(unique_attributes, model_name)
        return final_ranking

    def _extract_unique_attributes(self, predicates: List) -> Set[str]:
        attributes = set()
        for pred in predicates:
            if isinstance(pred, AttributePredicate):
                attributes.add(pred.attribute)
            elif isinstance(pred, AttributeComparisonPredicate):
                attributes.add(pred.attr1)
        return attributes

    def _stage1_attribute_importance(self, attributes: Set[str], model_name: str) -> Tuple[
        Dict[str, float], Dict[str, str]]:
        importance_ppl = {}
        attribute_prompts = {}
        for attr in sorted(attributes):
            if self.prompt_mode == 'C':
                prompt = self._build_stage1_complex_prompt(attr)
            else:
                prompt = self._build_stage1_simple_prompt(attr)
            attribute_prompts[attr] = prompt
            ppl = self.llm_manager.compute_perplexity(model_name, prompt)
            importance_ppl[attr] = ppl if ppl is not None else float('inf')
        return importance_ppl, attribute_prompts

    def _build_stage1_simple_prompt(self, attribute: str) -> str:
        domain_context = "In loan application risk assessment, financial institutions analyze applicant profiles to predict default probability. The evaluation process examines various data points to assess creditworthiness and repayment capacity."
        if attribute == "income_level":
            statement = f"{domain_context} When an applicant's income level falls into the lowest brackets (scaled 1-5, with level 1-2 indicating below-median earnings), this limited earning capacity serves as strong evidence that the loan application poses elevated default risk."
        elif attribute == "age_level":
            statement = f"{domain_context} When an applicant's age falls into extreme brackets—either very young (level 1-2, indicating limited career establishment) or advanced age (level 5, nearing retirement)—these age-related financial instability factors serve as strong evidence of elevated loan default risk."
        elif attribute == "experience_level":
            statement = f"{domain_context} When an applicant's total work experience is minimal (scaled 1-5, with level 1-2 indicating less than 5 years of professional history), this limited career track record serves as strong evidence that the loan application poses elevated default risk."
        elif attribute == "job_years_level":
            statement = f"{domain_context} When an applicant has been at their current job for a short duration (scaled 1-5, with level 1-2 indicating frequent job changes or recent employment), this employment instability serves as strong evidence that the loan application poses elevated default risk."
        elif attribute == "house_years_level":
            statement = f"{domain_context} When an applicant has resided at their current address for a brief period (scaled 1-5, with level 1-2 indicating recent relocation or high mobility), this residential instability serves as strong evidence that the loan application poses elevated default risk."
        elif attribute == "married_single":
            statement = f"{domain_context} When an applicant's marital status is single versus married, this household structure characteristic serves as strong evidence that the individual is likely to default on the loan."
        elif attribute == "house_ownership":
            statement = f"{domain_context} When an applicant does not own property and instead rents their residence, this absence of real estate collateral and lower financial commitment serves as strong evidence that the loan application poses elevated default risk."
        elif attribute == "car_ownership":
            statement = f"{domain_context} When an applicant does not own a vehicle, this absence of additional assets and limited financial capacity serves as strong evidence that the loan application poses elevated default risk."
        else:
            statement = f"{domain_context} When the attribute '{attribute}' shows certain patterns in the data, this serves as an indicator for loan default risk assessment."
        return statement

    def _build_stage1_complex_prompt(self, attribute: str) -> str:
        scenario_context = "[Context: Loan Application Risk Assessment Ecosystem] The lending decision system is modeled as a heterogeneous graph interconnecting five key entities: 1. Loans, 2. Applicants, 3. Cities, 4. States, 5. Professions. Default prediction aims to identify patterns signaling inability or unwillingness to repay debt obligations."
        definition_part = ""
        reasoning_part = ""
        if attribute == "income_level":
            definition_part = "The attribute 'income_level' categorizes the applicant's annual earnings into a quintile scale (1-5)."
            reasoning_part = "[Credit Risk Reasoning] Income is the primary source of loan repayment. Extremely low 'income_level' acts as a critical predictor of repayment failure."
        elif attribute == "age_level":
            definition_part = "The attribute 'age_level' places the applicant's age into brackets (1-5)."
            reasoning_part = "[Credit Risk Reasoning] Age correlates with financial stability. Tail ends (Level 1-2 and Level 5) elevate default risk."
        elif attribute == "experience_level":
            definition_part = "The attribute 'experience_level' quantifies total years of professional work history (1-5)."
            reasoning_part = "[Credit Risk Reasoning] Minimal experience indicates vulnerability to income disruption and subsequent loan default."
        elif attribute == "job_years_level":
            definition_part = "The attribute 'job_years_level' measures tenure at the current employer (1-5)."
            reasoning_part = "[Credit Risk Reasoning] Short job tenure is viewed as a liquidity risk flag."
        elif attribute == "house_years_level":
            definition_part = "The attribute 'house_years_level' tracks duration at the current residential address (1-5)."
            reasoning_part = "[Credit Risk Reasoning] Minimal 'house_years_level' serves as a soft signal of elevated credit risk."
        elif attribute == "married_single":
            definition_part = "The attribute 'married_single' classifies household structure (Married/Single)."
            reasoning_part = "[Credit Risk Reasoning] Marital status affects expense sharing and buffers against individual job loss."
        elif attribute == "house_ownership":
            definition_part = "The attribute 'house_ownership' indicates whether the applicant owns or rents their residence."
            reasoning_part = "[Credit Risk Reasoning] Renters lack protective factors like real estate collateral, discriminating between high-risk and low-risk borrowers."
        elif attribute == "car_ownership":
            definition_part = "The attribute 'car_ownership' denotes whether the applicant owns a vehicle."
            reasoning_part = "[Credit Risk Reasoning] Vehicle ownership signals discretionary income beyond basic living expenses."
        else:
            definition_part = f"The attribute '{attribute}' represents a specific data field within the applicant profile."
            reasoning_part = "[Credit Risk Reasoning] Deviation from performing loan norms raises flags for underwriting review."
        full_prompt = f"{scenario_context}\n\n[Attribute Analysis: {attribute}]\n{definition_part}\n\n{reasoning_part}"
        return full_prompt

    def _stage2_binary(self, attr: str, model_name: str, print_prompt: bool) -> Dict:
        attr_meta = ATTRIBUTE_METADATA["descriptions"].get(attr, {})
        attr_desc = attr_meta.get("neutral", attr)
        prefix = f"Context: Loan default prediction. Attribute: '{attr}' ({attr_desc})."
        if self.prompt_mode == 'C':
            prompt_true = f"{prefix} Credit Analysis: When this attribute is TRUE (1), the applicant aligns with elevated default risk."
            prompt_false = f"{prefix} Credit Analysis: When this attribute is FALSE (0), the applicant aligns with elevated default risk."
        else:
            prompt_true = f"{prefix} Value TRUE indicates elevated default risk."
            prompt_false = f"{prefix} Value FALSE indicates elevated default risk."
        ppl_true = self.llm_manager.compute_perplexity(model_name, prompt_true)
        ppl_false = self.llm_manager.compute_perplexity(model_name, prompt_false)
        return {
            'type': 'binary',
            'positive_direction': True if (ppl_true or float('inf')) < (ppl_false or float('inf')) else False,
            'positive_ppl': ppl_true or float('inf'),
            'negative_ppl': ppl_false or float('inf')
        }

    def _stage2_ordinal(self, attr: str, model_name: str, print_prompt: bool) -> Dict:
        attr_meta = ATTRIBUTE_METADATA["descriptions"].get(attr, {})
        attr_desc = attr_meta.get("neutral", attr)
        prefix = f"Context: Loan default prediction. Attribute: '{attr}' ({attr_desc})."
        if self.prompt_mode == 'C':
            prompt_inc = f"{prefix} Risk Trend: As the value INCREASES, the probability of loan default rises."
            prompt_dec = f"{prefix} Risk Trend: As the value DECREASES, the probability of loan default rises."
        else:
            prompt_inc = f"{prefix} HIGHER values indicate elevated default risk."
            prompt_dec = f"{prefix} LOWER values indicate elevated default risk."
        ppl_inc = self.llm_manager.compute_perplexity(model_name, prompt_inc)
        ppl_dec = self.llm_manager.compute_perplexity(model_name, prompt_dec)
        return {
            'type': 'ordinal',
            'direction': 'increasing' if (ppl_inc or float('inf')) < (ppl_dec or float('inf')) else 'decreasing',
            'increasing_ppl': ppl_inc or float('inf'),
            'decreasing_ppl': ppl_dec or float('inf')
        }

    def _stage2_value_direction(
            self,
            attributes: Set[str],
            model_name: str,
            importance_ppl: Dict[str, float]
    ) -> Dict[str, Dict]:
        direction_info = {}
        sorted_attrs = sorted(importance_ppl.items(), key=lambda x: x[1])
        top_attributes = [attr for attr, ppl in sorted_attrs[:20] if ppl != float('inf')]
        for attr in top_attributes:
            attr_type = self._get_attribute_type(attr)
            if attr_type == 'binary':
                result = self._stage2_binary(attr, model_name, False)
            elif attr_type == 'ordinal':
                result = self._stage2_ordinal(attr, model_name, False)
            else:
                result = {'type': 'categorical_encoded', 'direction': 'increasing'}
            direction_info[attr] = result
        return direction_info

    def _get_attribute_type(self, attr: str) -> str:
        if attr in ATTRIBUTE_METADATA["attribute_types"]["binary"]:
            return 'binary'
        elif attr in ATTRIBUTE_METADATA["attribute_types"]["ordinal_meaningful"]:
            return 'ordinal'
        else:
            return 'categorical_encoded'

    def _combine_stages_to_rank_predicates(
            self,
            predicates: List,
            attribute_importance: Dict[str, float],
            attribute_direction: Dict[str, Dict],
            attribute_prompts: Dict[str, str]
    ) -> Dict[str, float]:
        final_ranking = {}
        categorical_predicates = []
        for pred in predicates:
            pred_desc = pred.description()
            if isinstance(pred, WLPredicate):
                final_ranking[pred_desc] = 50.0
                if self.prompt_mode == 'C':
                    self.predicate_prompts[pred_desc] = "The 1-dimensional Weisfeiler-Leman (1-WL) test serves as a GNN-aligned structural indicator for default risk."
                else:
                    self.predicate_prompts[pred_desc] = "WL predicate (GNN prediction-based structural similarity)"
                continue
            if isinstance(pred, AttributePredicate):
                attr = pred.attribute
                value = pred.value
                operator = pred.operator
                base_ppl = attribute_importance.get(attr, float('inf'))
                if attr in attribute_prompts:
                    self.predicate_prompts[pred_desc] = attribute_prompts[attr]
                else:
                    self.predicate_prompts[pred_desc] = f"No prompt available for attribute: {attr}"
                if attr not in attribute_direction:
                    final_ranking[pred_desc] = base_ppl + 100
                    continue
                dir_info = attribute_direction[attr]
                if dir_info['type'] == 'binary':
                    if (value == 1 and dir_info['positive_direction']) or \
                            (value == 0 and not dir_info['positive_direction']):
                        final_ranking[pred_desc] = base_ppl
                elif dir_info['type'] == 'ordinal':
                    if dir_info['direction'] == 'increasing':
                        if operator == '>=':
                            final_ranking[pred_desc] = base_ppl
                    elif dir_info['direction'] == 'decreasing':
                        if operator == '<' or operator == '<=':
                            final_ranking[pred_desc] = base_ppl
                elif dir_info['type'] == 'categorical_encoded':
                    categorical_predicates.append((pred_desc, base_ppl + 50))
            elif isinstance(pred, AttributeComparisonPredicate):
                attr = pred.attr1
                base_ppl = attribute_importance.get(attr, float('inf'))
                final_ranking[pred_desc] = base_ppl + 20
                if attr in attribute_prompts:
                    self.predicate_prompts[pred_desc] = attribute_prompts[attr]
                else:
                    self.predicate_prompts[pred_desc] = f"Variable predicate for attribute: {attr}"
        for pred_desc, ppl in categorical_predicates:
            final_ranking[pred_desc] = ppl + random.uniform(0, 20)
        return final_ranking

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
        self.ppl_calculator = TwoStagePPLCalculator(
            self.llm_manager, self.analyzer, prompt_mode, print_prompts
        )

    def generate(self) -> Tuple[Dict[str, List[str]], List, Dict[str, Dict[str, float]]]:
        patterns = self.pattern_generator.generate_all_patterns()
        all_predicates = self.predicate_enumerator.enumerate_all_predicates(patterns)
        all_ppl_results = {}
        for model_name in self.llm_manager.llms.keys():
            if not Path(self.llm_manager.llms[model_name]).exists():
                continue
            model_results = self.ppl_calculator.compute_two_stage_ppl(
                all_predicates, patterns, model_name
            )
            all_ppl_results[model_name] = model_results
        ppl_by_model = {}
        for model_name, model_results in all_ppl_results.items():
            sorted_preds = self._sort_by_ppl(model_results)
            ppl_by_model[model_name] = sorted_preds
        return ppl_by_model, all_predicates, all_ppl_results

    def _sort_by_ppl(self, model_results: Dict[str, float]) -> List[str]:
        valid_items = [(p, v) for p, v in model_results.items() if v != float('inf')]
        valid_items.sort(key=lambda x: x[1])
        return [desc for desc, _ in valid_items]

    def save(self, ppl_by_model: Dict[str, List[str]], all_predicates: List, all_ppl_results: Dict):
        for model_name, ppl in ppl_by_model.items():
            output_file = f"loan_ppl.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(ppl, f)

def main():
    dataset_path = 'loan_graph.pkl'
    if not os.path.exists(dataset_path):
        return 1
    try:
        with open(dataset_path, 'rb') as f:
            data_graph = pickle.load(f)
        generator = TwoStagePPLGenerator(
            data_graph, 'loan', max_hops=3, prompt_mode='C', print_prompts=False
        )
        ppl_by_model, all_predicates, all_ppl_results = generator.generate()
        generator.save(ppl_by_model, all_predicates, all_ppl_results)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1
    return 0

if __name__ == "__main__":
    exit(main())