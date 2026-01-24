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
        "binary": ["verified"],
        "ordinal_meaningful": ["price_level", "rating"],
        "categorical_encoded": ["category", "brand_cluster", "brand_category",
                                "review_cluster", "review_category",
                                "description_cluster", "description_category"]
    },

    "attributes_by_label": {
        0: ["category", "brand_cluster", "brand_category", "review_cluster",
            "review_category", "description_cluster", "description_category", "price_level"],
        1: [],
        2: ["rating", "verified"]
    },

    "descriptions": {
        "category": {
            "neutral": "product category classification",
            "evidence": "product type indicating shopping preferences"
        },
        "brand_cluster": {
            "neutral": "brand text embedding cluster",
            "evidence": "brand name grouped into 10 semantic clusters by ML model"
        },
        "brand_category": {
            "neutral": "brand fine-grained category",
            "evidence": "brand name classified into 20 detailed categories by ML model"
        },
        "review_cluster": {
            "neutral": "review text embedding cluster",
            "evidence": "review content grouped into 10 semantic clusters by ML model"
        },
        "review_category": {
            "neutral": "review text fine-grained category",
            "evidence": "review content classified into 20 detailed categories by ML model"
        },
        "description_cluster": {
            "neutral": "product description embedding cluster",
            "evidence": "product description text grouped into 10 semantic clusters by ML model"
        },
        "description_category": {
            "neutral": "product description fine-grained category",
            "evidence": "product description classified into 20 detailed categories by ML model"
        },
        "price_level": {
            "neutral": "price bracket classification",
            "evidence": "product price on a 0-5 scale"
        },
        "rating": {
            "neutral": "customer review score",
            "evidence": "review rating (1-5 stars)"
        },
        "verified": {
            "neutral": "verified purchase status",
            "evidence": "whether the reviewer actually purchased the product"
        }
    },

    "category_specific_attrs": {
        "brand_cluster", "brand_category", "review_cluster",
        "review_category", "description_cluster", "description_category", "category"
    },

    "generic_attrs": {
        "rating", "verified"
    }
}

CATEGORIES = ['Lingerie', 'Jewelry', 'Womens-Fashion', 'Mens-Fashion', 'Sports-outdoors']
CATEGORY_LABELS = {
    'Lingerie': 0,
    'Jewelry': 1,
    'Womens-Fashion': 2,
    'Mens-Fashion': 3,
    'Sports-outdoors': 4
}

CATEGORY_CHARACTERISTICS = {
    'Lingerie': {
        'brand_cluster_range': [0, 1, 2, 3],
        'brand_category_range': [0, 1, 2, 3, 4, 5],
        'review_cluster_range': [0, 1, 4, 5],
        'review_category_range': [6, 7, 8, 9, 10, 11],
        'description_cluster_range': [0, 1, 2],
        'description_category_range': [6, 7, 8, 9, 10, 11],
        'price_range': 'mid-high (3-5)',
        'description': 'intimate apparel with emphasis on fit and comfort'
    },
    'Jewelry': {
        'brand_cluster_range': [0, 1, 2, 3],
        'brand_category_range': [0, 1, 2, 3, 4, 5],
        'review_cluster_range': [0, 1, 2],
        'review_category_range': [0, 1, 2, 3, 4, 5],
        'description_cluster_range': [0, 1, 2],
        'description_category_range': [0, 1, 2, 3, 4, 5],
        'price_range': 'wide spectrum (1-5)',
        'description': 'accessories emphasizing aesthetics and occasions'
    },
    'Womens-Fashion': {
        'brand_cluster_range': [2, 3, 4, 5],
        'brand_category_range': [4, 5, 6, 7, 8, 9, 10, 11],
        'review_cluster_range': [0, 1, 4, 5, 6],
        'review_category_range': [6, 7, 8, 9, 10, 11],
        'description_cluster_range': [1, 2, 4, 5],
        'description_category_range': [6, 7, 8, 9, 10, 11],
        'price_range': 'mid-range (2-4)',
        'description': 'general women\'s clothing with style diversity'
    },
    'Mens-Fashion': {
        'brand_cluster_range': [4, 5, 6, 7],
        'brand_category_range': [10, 11, 12, 13, 14, 15],
        'review_cluster_range': [3, 4, 5, 6],
        'review_category_range': [6, 7, 8, 9, 10, 11],
        'description_cluster_range': [4, 5, 6],
        'description_category_range': [12, 13, 14, 15, 16, 17],
        'price_range': 'mid-range (2-4)',
        'description': 'men\'s clothing emphasizing functionality and fit'
    },
    'Sports-outdoors': {
        'brand_cluster_range': [6, 7, 8, 9],
        'brand_category_range': [14, 15, 16, 17, 18, 19],
        'review_cluster_range': [3, 4, 5, 6, 7],
        'review_category_range': [0, 1, 2, 3, 4, 5],
        'description_cluster_range': [6, 7, 8],
        'description_category_range': [12, 13, 14, 15, 16, 17],
        'price_range': 'mid-high (3-5)',
        'description': 'athletic and outdoor gear emphasizing performance'
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
                if attr not in ['gnn_prediction', 'productTitle', 'brand', 'description',
                                'reviewText', 'reviewSummary', 'reviewerName', 'reviewTime',
                                'style', 'vote'] and value is not None:
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

        ordering_keywords = ['level', 'rating', 'price', 'score', 'count']
        force_ordering = any(keyword in attr.lower() for keyword in ordering_keywords)

        if unique_count <= 25 and not force_ordering:
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
                    'type': 'categorical',
                    'strategy': 'constant_only',
                    'unique_count': unique_count,
                    'frequent_values': [(v, f) for v, f in frequent_values[:25]],
                    'supports_ordering': False
                }

    def _supports_partial_order_strict(self, attr: str) -> bool:
        force_ordering_keywords = ['level', 'rating', 'price', 'score', 'count']
        attr_lower = attr.lower()
        if any(keyword in attr_lower for keyword in force_ordering_keywords):
            return True
        force_categorical_keywords = ['category', 'cluster', 'type', 'class', 'group', 'verified']
        if any(keyword in attr_lower for keyword in force_categorical_keywords):
            return False
        return False

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
        p1.add_node(Node('x2', 2, {}))
        p1.add_edge('x1', 'x0')
        p1.add_edge('x1', 'x2')
        p1.add_edge('x2', 'x0')
        p1.set_pivot('x0')
        patterns.append(p1)
        return patterns

    def _generate_2hop_patterns(self) -> List[Pattern]:
        patterns = []
        p1 = Pattern()
        p1.add_node(Node('x0', 0, {}))
        p1.add_node(Node('x1', 1, {}))
        p1.add_node(Node('x2', 2, {}))
        p1.add_node(Node('x3', 0, {}))
        p1.add_edge('x1', 'x0')
        p1.add_edge('x1', 'x2')
        p1.add_edge('x2', 'x0')
        p1.add_edge('x1', 'x3')
        p1.set_pivot('x0')
        patterns.append(p1)
        p2 = Pattern()
        p2.add_node(Node('x0', 0, {}))
        p2.add_node(Node('x1', 1, {}))
        p2.add_node(Node('x2', 2, {}))
        p2.add_node(Node('x3', 0, {}))
        p2.add_node(Node('x4', 2, {}))
        p2.add_edge('x1', 'x0')
        p2.add_edge('x1', 'x2')
        p2.add_edge('x2', 'x0')
        p2.add_edge('x1', 'x3')
        p2.add_edge('x1', 'x4')
        p2.add_edge('x4', 'x3')
        p2.set_pivot('x0')
        patterns.append(p2)
        return patterns

    def _generate_3hop_patterns(self) -> List[Pattern]:
        patterns = []
        p1 = Pattern()
        p1.add_node(Node('x0', 0, {}))
        p1.add_node(Node('x1', 1, {}))
        p1.add_node(Node('x2', 2, {}))
        p1.add_node(Node('x3', 0, {}))
        p1.add_node(Node('x4', 0, {}))
        p1.add_edge('x1', 'x0')
        p1.add_edge('x1', 'x2')
        p1.add_edge('x2', 'x0')
        p1.add_edge('x1', 'x3')
        p1.add_edge('x1', 'x4')
        p1.set_pivot('x0')
        patterns.append(p1)
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
    def __init__(self, data_graph: Graph, analyzer: GraphDataAnalyzer, target_category: str):
        self.data_graph = data_graph
        self.analyzer = analyzer
        self.target_category = target_category

    def enumerate_all_predicates(self, patterns: List[Pattern]) -> List:
        all_predicates = []
        predicate_set = set()
        for pattern in patterns:
            gens = [self._generate_constant_predicates, self._generate_partial_order_predicates,
                    self._generate_variable_predicates, self._generate_wl_predicates]
            for gen in gens:
                for pred in gen(pattern):
                    desc = pred.description()
                    if desc not in predicate_set:
                        all_predicates.append(pred)
                        predicate_set.add(desc)
        return all_predicates

    def _generate_constant_predicates(self, pattern: Pattern) -> List:
        predicates = []
        for node_id, node in pattern.graph.nodes.items():
            label = node.label
            if label not in ATTRIBUTE_METADATA["attributes_by_label"]: continue
            allowed = ATTRIBUTE_METADATA["attributes_by_label"][label]
            if not allowed or label not in self.analyzer.attribute_info: continue
            for attr, info in self.analyzer.attribute_info[label].items():
                if attr in allowed and info['strategy'] == 'constant_only':
                    for val, freq in info.get('frequent_values', []):
                        predicates.append(AttributePredicate(node_id, attr, val, '=='))
        return predicates

    def _generate_partial_order_predicates(self, pattern: Pattern) -> List:
        predicates = []
        for node_id, node in pattern.graph.nodes.items():
            label = node.label
            if label not in ATTRIBUTE_METADATA["attributes_by_label"]: continue
            allowed = ATTRIBUTE_METADATA["attributes_by_label"][label]
            if not allowed or label not in self.analyzer.attribute_info: continue
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
                    for attr, info in self.analyzer.attribute_info.get(node1.label, {}).items():
                        if attr in allowed:
                            if info['strategy'] == 'partial_order':
                                predicates.append(AttributeComparisonPredicate(nid1, attr, nid2, attr, '>='))
                            elif info['strategy'] == 'constant_only':
                                predicates.append(AttributeComparisonPredicate(nid1, attr, nid2, attr, '=='))
        return predicates

    def _generate_wl_predicates(self, pattern: Pattern) -> List:
        predicates = []
        for node_id, node in pattern.graph.nodes.items():
            if node.label == 0:
                predicates.append(WLPredicate(node_id, is_negated=False, gnn_attr='gnn_prediction'))
        return predicates

class LocalLLMManager:
    def __init__(self):
        self.llms = {'Llama3B': 'Llama3B/', 'Qwen3B': 'Qwen3B/', 'Phi38B': 'Phi38B/'}
        self.loaded_models = {}
        self.loaded_tokenizers = {}

    def load_model(self, model_name: str):
        if model_name in self.loaded_models:
            return self.loaded_models[model_name], self.loaded_tokenizers[model_name]
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            path = self.llms[model_name]
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
            self.loaded_models[model_name], self.loaded_tokenizers[model_name] = model, tokenizer
            return model, tokenizer
        except: return None, None

    def compute_perplexity(self, model_name: str, text: str) -> Optional[float]:
        model, tokenizer = self.load_model(model_name)
        if not model: return None
        try:
            import torch
            inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)
            input_ids = inputs["input_ids"].to(model.device)
            with torch.no_grad():
                loss = model(input_ids=input_ids, labels=input_ids).loss
                return float(np.exp(loss.item())) if loss is not None else None
        except: return None

class TwoStagePPLCalculator:
    def __init__(self, llm_manager: LocalLLMManager, analyzer,
                 target_category: str, prompt_mode: str = 'S', print_prompts: bool = False):
        self.llm_manager = llm_manager
        self.analyzer = analyzer
        self.target_category = target_category
        self.prompt_mode = prompt_mode
        self.print_prompts = print_prompts
        self.predicate_prompts = {}

    def compute_two_stage_ppl(self, predicates: List, patterns: List[Pattern], model_name: str) -> Dict[str, float]:
        unique_attrs = self._extract_unique_attributes(predicates)
        imp_ppl, attr_prompts = self._stage1_attribute_importance(unique_attrs, model_name)
        dir_info = self._stage2_value_direction(unique_attrs, model_name, imp_ppl)
        return self._combine_stages_to_rank_predicates(predicates, imp_ppl, dir_info, attr_prompts)

    def _extract_unique_attributes(self, predicates: List) -> Set[str]:
        attrs = set()
        for pred in predicates:
            if isinstance(pred, AttributePredicate): attrs.add(pred.attribute)
            elif isinstance(pred, AttributeComparisonPredicate): attrs.add(pred.attr1)
        return attrs

    def _stage1_attribute_importance(self, attributes: Set[str], model_name: str) -> Tuple[Dict[str, float], Dict[str, str]]:
        importance, prompts = {}, {}
        for attr in sorted(attributes):
            prompt = self._build_stage1_complex_prompt(attr) if self.prompt_mode == 'C' else self._build_stage1_simple_prompt(attr)
            prompts[attr] = prompt
            ppl = self.llm_manager.compute_perplexity(model_name, prompt)
            importance[attr] = ppl if ppl is not None else float('inf')
        return importance, prompts

    def _build_stage1_simple_prompt(self, attribute: str) -> str:
        other = ", ".join([c for c in CATEGORIES if c != self.target_category][:2])
        config = CATEGORY_CHARACTERISTICS.get(self.target_category, {})
        ctx = f"Distinguish {self.target_category} from {other}."
        return f"{ctx} Attribute '{attribute}' is key evidence."

    def _build_stage1_complex_prompt(self, attribute: str) -> str:
        other = ", ".join([c for c in CATEGORIES if c != self.target_category])
        ctx = f"[Amazon Classification] Distinguish {self.target_category} from {other}."
        return f"{ctx} Analyzing '{attribute}' discriminative power."

    def _stage2_value_direction(self, attributes: Set[str], model_name: str, importance: Dict[str, float]) -> Dict[str, Dict]:
        directions = {}
        top = [a for a, p in sorted(importance.items(), key=lambda x: x[1])[:20] if p != float('inf')]
        for attr in top:
            t = self._get_attribute_type(attr)
            if t == 'binary':
                p_true = self.llm_manager.compute_perplexity(model_name, f"{attr} is TRUE for {self.target_category}")
                p_false = self.llm_manager.compute_perplexity(model_name, f"{attr} is FALSE for {self.target_category}")
                directions[attr] = {'type': 'binary', 'positive_direction': (p_true or 1e9) < (p_false or 1e9)}
            else:
                directions[attr] = {'type': 'other', 'direction': 'increasing'}
        return directions

    def _get_attribute_type(self, attr: str) -> str:
        if attr in ATTRIBUTE_METADATA["attribute_types"]["binary"]: return 'binary'
        if attr in ATTRIBUTE_METADATA["attribute_types"]["ordinal_meaningful"]: return 'ordinal'
        return 'categorical'

    def _combine_stages_to_rank_predicates(self, predicates: List, importance: Dict[str, float], directions: Dict[str, Dict], prompts: Dict[str, str]) -> Dict[str, float]:
        ranks = {}
        for pred in predicates:
            desc = pred.description()
            if isinstance(pred, WLPredicate):
                ranks[desc] = 50.0
            elif isinstance(pred, AttributePredicate):
                a = pred.attribute
                base = importance.get(a, 1e9)
                if a in ATTRIBUTE_METADATA["generic_attrs"]: base += 300
                elif a in ATTRIBUTE_METADATA["category_specific_attrs"]: base *= 0.5
                ranks[desc] = base
            else:
                ranks[desc] = importance.get(getattr(pred, 'attribute', getattr(pred, 'attr1', '')), 1e9) + 40
        return ranks

class TwoStagePPLGenerator:
    def __init__(self, data_graph: Graph, dataset_name: str, target_category: str, max_hops: int = 3, prompt_mode: str = 'S', print_prompts: bool = False):
        self.data_graph = data_graph
        self.target_category = target_category
        self.analyzer = GraphDataAnalyzer(data_graph)
        self.pattern_generator = KHopPatternGenerator(data_graph, self.analyzer, max_hops)
        self.predicate_enumerator = RichPredicateEnumerator(data_graph, self.analyzer, target_category)
        self.llm_manager = LocalLLMManager()
        self.ppl_calculator = TwoStagePPLCalculator(self.llm_manager, self.analyzer, target_category, prompt_mode)

    def generate(self):
        patterns = self.pattern_generator.generate_all_patterns()
        preds = self.predicate_enumerator.enumerate_all_predicates(patterns)
        results = {}
        for m in self.llm_manager.llms:
            if Path(self.llm_manager.llms[m]).exists():
                ranks = self.ppl_calculator.compute_two_stage_ppl(preds, patterns, m)
                results[m] = [p for p, v in sorted(ranks.items(), key=lambda x: x[1]) if v != float('inf')]
        return results, preds

    def save(self, ppl_by_model):
        for m, ppl in ppl_by_model.items():
            with open(f"amazon_{self.target_category.lower()}_ppl.pkl", 'wb') as f:
                pickle.dump(ppl, f)

def main():
    path = 'amazon_graph.pkl'
    if not os.path.exists(path): return 1
    try:
        with open(path, 'rb') as f:
            graph = pickle.load(f)
        for cat in CATEGORIES:
            gen = TwoStagePPLGenerator(graph, 'amazon', cat, prompt_mode='C')
            ppl, _ = gen.generate()
            gen.save(ppl)
    except: return 1
    return 0

if __name__ == "__main__":
    exit(main())