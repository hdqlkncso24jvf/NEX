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
        "binary": ["type"],
        "ordinal_meaningful": [
            "amount_level", "src_old_balance_level", "src_new_balance_level",
            "dest_old_balance_level", "dest_new_balance_level"
        ],
        "categorical_encoded": []
    },

    "attributes_by_label": {
        0: ["type", "amount_level", "src_old_balance_level", "src_new_balance_level",
            "dest_old_balance_level", "dest_new_balance_level"],
        1: []
    },

    "descriptions": {
        "type": {
            "neutral": "transaction category classification",
            "evidence": "transaction type (PAYMENT, TRANSFER, CASH_OUT, DEBIT) indicating operational risk profile"
        },
        "amount_level": {
            "neutral": "monetary value bracket",
            "evidence": "transaction amount on a 0-5 scale indicating financial magnitude and anomaly detection"
        },
        "src_old_balance_level": {
            "neutral": "originating account pre-transaction balance",
            "evidence": "source account balance before transaction on a 0-5 scale revealing financial capacity"
        },
        "src_new_balance_level": {
            "neutral": "originating account post-transaction balance",
            "evidence": "source account balance after transaction on a 0-5 scale indicating balance manipulation"
        },
        "dest_old_balance_level": {
            "neutral": "destination account pre-transaction balance",
            "evidence": "recipient account balance before transaction on a 0-5 scale revealing account legitimacy"
        },
        "dest_new_balance_level": {
            "neutral": "destination account post-transaction balance",
            "evidence": "recipient account balance after transaction on a 0-5 scale indicating fund movement patterns"
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
                if attr not in ['gnn_prediction', 'is_fraud', 'step'] and value is not None:
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

        ordering_keywords = ['level', 'amount', 'balance', 'count', 'grade', 'rank', 'score']
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
        force_ordering_keywords = ['level', 'amount', 'balance', 'count', 'grade', 'rank', 'score']
        attr_lower = attr.lower()
        if any(keyword in attr_lower for keyword in force_ordering_keywords):
            return True

        force_categorical_keywords = [
            'type', 'category', 'class', 'group', 'status', 'id'
        ]
        if any(keyword in attr_lower for keyword in force_categorical_keywords):
            return False

        general_ordering_keywords = [
            'time', 'date', 'month', 'day', 'rate', 'ratio',
            'num', 'number', 'duration', 'length', 'size', 'price', 'cost',
            'value', 'income', 'salary'
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
        p1.add_node(Node('x2', 1, {}))
        p1.add_edge('x0', 'x1')
        p1.add_edge('x0', 'x2')
        p1.set_pivot('x0')
        patterns.append(p1)

        return patterns

    def _generate_2hop_patterns(self) -> List[Pattern]:
        patterns = []

        p1 = Pattern()
        p1.add_node(Node('x0', 0, {}))
        p1.add_node(Node('x1', 1, {}))
        p1.add_node(Node('x2', 1, {}))
        p1.add_node(Node('x3', 0, {}))
        p1.add_edge('x0', 'x1')
        p1.add_edge('x0', 'x2')
        p1.add_edge('x3', 'x1')
        p1.set_pivot('x0')
        patterns.append(p1)

        p2 = Pattern()
        p2.add_node(Node('x0', 0, {}))
        p2.add_node(Node('x1', 1, {}))
        p2.add_node(Node('x2', 1, {}))
        p2.add_node(Node('x3', 0, {}))
        p2.add_node(Node('x4', 0, {}))
        p2.add_edge('x0', 'x1')
        p2.add_edge('x0', 'x2')
        p2.add_edge('x3', 'x1')
        p2.add_edge('x4', 'x2')
        p2.set_pivot('x0')
        patterns.append(p2)

        return patterns

    def _generate_3hop_patterns(self) -> List[Pattern]:
        patterns = []

        p1 = Pattern()
        p1.add_node(Node('x0', 0, {}))
        p1.add_node(Node('x1', 1, {}))
        p1.add_node(Node('x2', 1, {}))
        p1.add_node(Node('x3', 0, {}))
        p1.add_node(Node('x4', 1, {}))
        p1.add_node(Node('x5', 0, {}))
        p1.add_edge('x0', 'x1')
        p1.add_edge('x0', 'x2')
        p1.add_edge('x3', 'x2')
        p1.add_edge('x3', 'x4')
        p1.add_edge('x5', 'x4')
        p1.set_pivot('x0')
        patterns.append(p1)

        p2 = Pattern()
        p2.add_node(Node('x0', 0, {}))
        p2.add_node(Node('x1', 1, {}))
        p2.add_node(Node('x2', 1, {}))
        p2.add_node(Node('x3', 1, {}))
        p2.add_node(Node('x4', 0, {}))
        p2.add_node(Node('x5', 1, {}))
        p2.add_node(Node('x6', 0, {}))
        p2.add_edge('x0', 'x1')
        p2.add_edge('x0', 'x2')
        p2.add_edge('x4', 'x2')
        p2.add_edge('x4', 'x5')
        p2.add_edge('x6', 'x5')
        p2.add_edge('x0', 'x3')
        p2.set_pivot('x0')
        patterns.append(p2)

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
        attribute_direction = self._stage2_value_direction(unique_attributes, model_name, attribute_importance_ppl)

        final_ranking = self._combine_stages_to_rank_predicates(
            predicates, attribute_importance_ppl, attribute_direction, attribute_prompts
        )

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
        domain_context = """In financial fraud detection, analysts examine transaction patterns to identify suspicious money flows and account manipulation. The investigation process scrutinizes various transaction attributes to assess fraud probability and financial crime indicators."""

        if attribute == "type":
            statement = f"""{domain_context}

When a transaction type is classified as CASH_OUT or TRANSFER rather than routine PAYMENT or DEBIT operations, this elevated-risk category serves as strong evidence that the transaction exhibits fraudulent patterns, as these types enable rapid fund extraction and layering techniques characteristic of money laundering schemes."""

        elif attribute == "amount_level":
            statement = f"""{domain_context}

When transaction amounts reach extreme levels (scaled 0-5, with level 0 indicating zero-value anomalies or level 5 indicating top-tier outliers far exceeding normal patterns), these financial irregularities serve as strong evidence that the transaction is likely fraudulent, as both endpoints signal manipulation attempts or structuring behavior."""

        elif attribute == "src_old_balance_level":
            statement = f"""{domain_context}

When the originating account's pre-transaction balance falls into extreme brackets (scaled 0-5, with level 0-1 indicating depleted accounts or level 5 indicating unusually high balances), these account state anomalies serve as strong evidence of fraudulent activity, as compromised accounts often exhibit either systematic drainage or suspicious accumulation patterns."""

        elif attribute == "src_new_balance_level":
            statement = f"""{domain_context}

When the originating account's post-transaction balance drops to zero (level 0) or shows implausible changes that violate conservation of funds (levels inconsistent with transaction amounts), these balance manipulation patterns serve as strong evidence that the transaction is fraudulent, particularly indicating account takeover scenarios where attackers drain balances completely."""

        elif attribute == "dest_old_balance_level":
            statement = f"""{domain_context}

When the destination account's pre-transaction balance is near-zero (level 0-1, indicating newly created or dormant mule accounts) or shows patterns inconsistent with legitimate business operations, these recipient account anomalies serve as strong evidence of fraudulent transactions, as criminals often use shell accounts with minimal prior activity to receive stolen funds."""

        elif attribute == "dest_new_balance_level":
            statement = f"""{domain_context}

When the destination account's post-transaction balance reaches zero (level 0, indicating immediate fund forwarding) or shows mathematically impossible values that violate transaction logic, these balance inconsistency patterns serve as strong evidence of fraud, revealing rapid layering techniques where funds are immediately moved to obscure the money trail."""

        else:
            statement = f"""{domain_context}

When the attribute '{attribute}' exhibits certain patterns in transaction data, this serves as an indicator for fraud detection analysis in financial systems."""

        return statement

    def _build_stage1_complex_prompt(self, attribute: str) -> str:
        scenario_context = """[Context: Financial Transaction Fraud Detection Ecosystem]
The payment system is modeled as a bipartite graph interconnecting two key entities:
1. **Transactions**: Individual monetary transfers representing fund movements between parties.
2. **Accounts**: Financial accounts (both originating and destination) participating in transactions.

Fraud detection within this network aims to identify attribute patterns—such as balance manipulation, suspicious transaction types, or mathematically impossible fund flows—that violate financial integrity constraints and signal criminal money laundering or account takeover schemes."""

        attr_meta = ATTRIBUTE_METADATA["descriptions"].get(attribute, {})

        definition_part = ""
        reasoning_part = ""

        if attribute == "type":
            definition_part = """The attribute 'type' categorizes transactions into operational classes: PAYMENT (routine purchases), TRANSFER (internal account movements), CASH_OUT (ATM withdrawals or cash extraction), DEBIT (direct debits), and CASH_IN (deposits). Each type carries distinct fraud risk profiles based on liquidity extraction speed and traceability."""
            reasoning_part = """[Fraud Detection Reasoning] Transaction type operates as a primary risk stratification mechanism. CASH_OUT and TRANSFER transactions enable rapid fund extraction with minimal paper trail, making them preferred vectors for money laundering's "placement" and "layering" stages. Financial crime theory establishes that 70-80% of confirmed fraud involves these high-velocity types. Unlike traceable PAYMENT transactions that leave merchant records, CASH_OUT events convert digital balances to untraceable physical currency. Therefore, transaction 'type' acts as a categorical risk flag requiring elevated scrutiny."""

        elif attribute == "amount_level":
            definition_part = """The attribute 'amount_level' quantifies transaction monetary value on a 0-5 scale. Level 0 represents zero-value transactions (placeholder or system errors), levels 1-2 indicate micro-transactions, level 3 denotes median-range transfers, while levels 4-5 flag high-value outliers exceeding 2-3 standard deviations from population norms."""
            reasoning_part = """[Fraud Detection Reasoning] Transaction amount serves dual forensic purposes. Extreme low values (level 0-1) may indicate "structuring"—deliberate splitting of large sums below reporting thresholds to evade anti-money laundering triggers mandated by Bank Secrecy Act regulations. Conversely, extreme high values (level 5) signal potential account takeover where attackers maximize extraction before detection. Benign transactions cluster around population median (level 3). Thus, 'amount_level' extremes at both tails serve as deviation-based fraud indicators."""

        elif attribute == "src_old_balance_level":
            definition_part = """The attribute 'src_old_balance_level' captures the originating account's balance immediately before transaction execution (0-5 scale). It reveals the account's financial state and capacity to fund the outgoing transfer, serving as a constraint on legitimate transaction feasibility."""
            reasoning_part = """[Fraud Detection Reasoning] Pre-transaction source balance operates as a plausibility gate. Compromised accounts often exhibit systematic drainage patterns—attackers inherit whatever balance exists and deplete it completely. A level 5 source balance funding a level 5 transaction is mathematically coherent; however, a level 1 balance funding a level 4 transaction violates conservation of funds unless external credit is involved. Additionally, dormant accounts suddenly reactivating with unusual balances (e.g., level 0 jumping to level 5) suggest synthetic identity fraud or account farming. Thus, 'src_old_balance_level' provides baseline context for detecting impossible or improbable fund movements."""

        elif attribute == "src_new_balance_level":
            definition_part = """The attribute 'src_new_balance_level' records the originating account's balance immediately after transaction completion (0-5 scale). It represents the residual funds and reveals whether the transaction drained, partially depleted, or anomalously preserved the source account balance."""
            reasoning_part = """[Fraud Detection Reasoning] Post-transaction source balance encodes critical behavioral signatures. A drop to level 0 (complete balance depletion) is the hallmark of account takeover fraud—attackers maximize extraction knowing the victim will soon regain control and block further access. Legitimate users rarely drain accounts to zero, as they maintain buffers for ongoing obligations. Conversely, transactions where 'src_new_balance_level' equals 'src_old_balance_level' despite non-zero amounts violate basic arithmetic, signaling database manipulation or system compromise. Financial forensics literature identifies complete-drain transactions as carrying 15-20x higher fraud probability than partial withdrawals."""

        elif attribute == "dest_old_balance_level":
            definition_part = """The attribute 'dest_old_balance_level' indicates the destination account's balance before receiving the incoming transaction (0-5 scale). It characterizes the recipient account's financial profile and prior activity level, distinguishing between established accounts and newly created mule accounts."""
            reasoning_part = """[Fraud Detection Reasoning] Pre-transaction destination balance serves as a mule account detector. Money laundering operations employ "money mule" networks—intermediary accounts opened specifically to receive and forward illicit funds. These accounts typically maintain near-zero balances (level 0-1) between transactions to minimize asset seizure risk during law enforcement action. Established merchant or business accounts display stable level 3-4 balances reflecting ongoing operations. A pattern of level 0 destination accounts receiving large inflows suggests layering stage laundering. Thus, 'dest_old_balance_level' discriminates between legitimate recipients and transient conduit accounts."""

        elif attribute == "dest_new_balance_level":
            definition_part = """The attribute 'dest_new_balance_level' captures the destination account's balance immediately after receiving the transaction (0-5 scale). It reveals the recipient's retention behavior and whether incoming funds are immediately forwarded elsewhere or accumulated."""
            reasoning_part = """[Fraud Detection Reasoning] Post-transaction destination balance exposes rapid-fire layering tactics. If 'dest_new_balance_level' returns to zero despite receiving substantial funds (high 'amount_level'), it indicates immediate onward transfer—a signature of professional money laundering where no single account holds illicit funds long enough for detection or freezing. This "hot potato" pattern where funds spend <1 hour in recipient accounts before re-forwarding is algorithmically flagged by anti-fraud systems. Conversely, mathematical impossibilities (e.g., dest_new < dest_old despite incoming transfer) signal database tampering. Therefore, 'dest_new_balance_level' serves as both a layering indicator and integrity validator."""

        else:
            definition_part = f"""The attribute '{attribute}' represents a specific data field within the transaction record. It contributes to the overall structured assessment of transaction legitimacy and financial crime risk."""
            reasoning_part = f"""[Fraud Detection Reasoning] The forensic value of this attribute depends on its correlation with historical fraud patterns observed in payment network data. If the value deviates significantly from baseline distributions in legitimate transaction populations, it raises flags for automated fraud scoring systems and manual investigative review."""

        full_prompt = f"""{scenario_context}

[Attribute Analysis: {attribute}]
{definition_part}

{reasoning_part}"""

        return full_prompt

    def _stage2_binary(self, attr: str, model_name: str, print_prompt: bool) -> Dict:
        attr_meta = ATTRIBUTE_METADATA["descriptions"].get(attr, {})
        attr_desc = attr_meta.get("neutral", attr)

        prefix = f"Context: Financial fraud detection involving Transactions and Accounts. Attribute: '{attr}' ({attr_desc})."

        if self.prompt_mode == 'C':
            prompt_true = f"""{prefix} Fraud Analysis: When this transaction type is high-risk (CASH_OUT/TRANSFER), the transaction aligns with established patterns of fund extraction and money laundering operations."""
            prompt_false = f"""{prefix} Fraud Analysis: When this transaction type is low-risk (PAYMENT/DEBIT), the transaction aligns with established patterns of fund extraction and money laundering operations."""
        else:
            prompt_true = f"""{prefix} High-risk type indicates elevated fraud probability."""
            prompt_false = f"""{prefix} Low-risk type indicates elevated fraud probability."""

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

        prefix = f"Context: Financial fraud detection involving Transactions and Accounts. Attribute: '{attr}' ({attr_desc})."

        if self.prompt_mode == 'C':
            prompt_inc = f"""{prefix} Risk Correlation: As the value INCREASES toward extreme highs (level 5), the probability of fraudulent manipulation significantly rises due to outlier behavior and anomaly detection triggers."""
            prompt_dec = f"""{prefix} Risk Correlation: As the value DECREASES toward extreme lows or zero (level 0-1), the probability of fraudulent patterns significantly rises due to account drainage, structuring, or balance manipulation."""
        else:
            prompt_inc = f"""{prefix} HIGHER values indicate elevated fraud risk."""
            prompt_dec = f"""{prefix} LOWER values indicate elevated fraud risk."""

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
                    self.predicate_prompts[
                        pred_desc] = """The 1-dimensional Weisfeiler-Leman (1-WL) test is a graph-theoretic algorithm used to determine structural similarity between vertices in a graph. In the context of explaining GNN predictions, the predicate ¬1WL(x) indicates that vertex x belongs to the same structural equivalence class as other vertices that the GNN model has predicted as fraudulent transactions. Since most GNN architectures are known to be no more expressive than the 1-WL test, if the model predicts fraud for structurally similar transactions, it is likely to behave similarly for this transaction. This predicate serves as a GNN-aligned structural indicator for fraud risk in financial networks."""
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
            output_file = f"transaction_ppl.pkl"
            with open(output_file, 'wb') as f:
                pickle.dump(ppl, f)

def main():
    dataset_path = 'transaction_graph.pkl'

    if not os.path.exists(dataset_path):
        return 1

    try:
        with open(dataset_path, 'rb') as f:
            data_graph = pickle.load(f)

        generator = TwoStagePPLGenerator(
            data_graph, 'transaction', max_hops=3, prompt_mode='C', print_prompts=False
        )
        ppl_by_model, all_predicates, all_ppl_results = generator.generate()

        generator.save(ppl_by_model, all_predicates, all_ppl_results)

    except Exception as e:
        import traceback
        return 1

    return 0

if __name__ == "__main__":
    exit(main())