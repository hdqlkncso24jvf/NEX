# Overview

This repository is the official code of "**Global and Local Explanations for Negative GNN Predictions**".

This paper studies explanations for graph neural network (GNN) classifiers M when M makes a negative prediction, such as loan denials, paper rejections, or declined job applications. The objective is to both (a) globally explain the general behavior of M; and (b) suggest counterfactual explanations locally at a vertex 𝑢 in a graph, which are necessary changes to features/topology around 𝑢 for M to swap its prediction at 𝑢. We propose a class of rules which treat negative M-predictions as their consequences. 

![Architecture](https://github.com/hdqlkncso24jvf/NEX/blob/main/NEX/tech/architecture.png)

We develop algorithms to (a) learn such rules as global explanations and (b) compute local counterfactual explanations, by applying the learned rules. Over real-life graphs, our algorithms are on average 70.16% and 189.68% higher in recognizability and reliability than prior global methods, and 33.60% and 3.25× better than previous local (counterfactual) methods in fidelity and sparsity, respectively.

![Exp](https://github.com/hdqlkncso24jvf/NEX/blob/main/NEX/tech/exp.png)

* The **full version** of the paper can be accessed at this file: [`paper_full_version.pdf`](https://github.com/hdqlkncso24jvf/NEX/blob/main/paper_full_version.pdf)
  * [BACKUP Link 1](https://mxieaa.github.io/paper/paper_full_version.pdf)
  * [BACKUP Link 2](https://drive.google.com/file/d/15B-96yfgQ8VshvRoE_MyW9xdicuLr4np/view?usp=sharing)
* The **model checkpoints and dataset** are available at this link: [Google Drive](https://drive.google.com/drive/folders/1Cu9WXhRAq-8J4ZBd2REHG9fep4mzLTg8?usp=drive_link)

## Software requirements

```shell
python >= 3.10
torch == 2.3.0
transformers == 4.41.2
datasets >= 2.16.0
accelerate >= 0.30.1
peft >= 0.11.1
trl >= 0.8.6
vllm == 0.4.3
CUDA >= 11.6
flash-attn >= 2.3.0
torch_geometric == 2.5.3 # this is for GNN
numpy >= 1.24.2
scikit-learn >= 1.2.2
scipy >= 1.10.1
torch >= 1.13.1
tqdm >= 4.65.0
```

## Base model and Hardware Requirements

For LLM Predicate

The recommended and default base model is Qwen2.5-3B; however, models from the llama, Llama and ChatGLM series, and others are also supported. If you wish to switch the base model, please modify the template as follows:

- **Llama Series**
  - download link: https://huggingface.co/meta-llama
  - Template: `llama2/llama3`

- **Qwen Series**
  - download link: https://huggingface.co/Qwen
  - Template: `qwen`

- **GLM  Series**
  - download link: https://huggingface.co/THUDM
  - Template: `glm3/glm4`


For fine-tuning large language models ranging from 1.5B to 9B parameters, a minimum of one NVIDIA 3090 GPU is required, with a recommended GPU Memory of at least 24GB per card.

For fine-tuning models larger than 13B parameters, at least one V100 GPU is necessary, with a recommended GPU Memory of at least 32GB per card.

## Run

```shell
python main.py train train.yaml
```

```yaml
# train.yaml
### model
model_name_or_path: your_model_path
quantization_bit: 4 # quantization bits, reduce model size and speed up inference, comment out to use LoRA fine-tuning.

### method
stage: sft # Supervised fine-tuning stage
do_train: true # Flag to indicate training mode
finetuning_type: qlora # Fine-tuning method, can be qlora (quantized LoRA) or other types
lora_target: all # Apply LoRA fine-tuning to all model layers

### dataset
dataset: your_data_set_name
template: mistral # Template for dataset processing, depending on the base model
cutoff_len: 1024 # Maximum sequence length for input data
max_samples: 20 # Maximum number of samples to use from the dataset
overwrite_cache: true # Overwrite cached dataset files if they exist
preprocessing_num_workers: 16 # Number of worker threads for data preprocessing

### output
output_dir: checkpoint_output_path
logging_steps: 10 # Number of steps between logging metrics
save_steps: 500 # Number of steps between saving checkpoints
plot_loss: true # Flag to plot loss during training
overwrite_output_dir: true # Overwrite the contents of the output directory if it exists

### train
per_device_train_batch_size: 1 # Batch size per device during training
gradient_accumulation_steps: 8 # Number of steps to accumulate gradients before updating
learning_rate: 1.0e-4 # Learning rate for the optimizer
num_train_epochs: 3.0 # Total number of training epochs
lr_scheduler_type: cosine # Type of learning rate scheduler (cosine annealing)
warmup_ratio: 0.1 # Proportion of training steps to perform learning rate warmup
fp16: true # Use 16-bit (half precision) floating point arithmetic for training
ddp_timeout: 180000000 # Timeout for distributed data parallel (DDP) training in seconds

### eval
val_size: 0.1 # Proportion of the dataset to use for validation
per_device_eval_batch_size: 1 # Batch size per device during evaluation
eval_strategy: steps # Evaluation strategy to use (evaluate every few steps)
eval_steps: 500 # Number of steps between evaluations
```

### Merge lora checkpoint to get fine-tuned LLMs

```shell
python main.py export merge.yaml
```

```yaml
# merge.yaml
### model
model_name_or_path: your_model_path
adapter_name_or_path: checkpoint_output_path
template: mistral
finetuning_type: qlora # lora

### export
export_dir: merged_model_path
export_size: 2 # model size(GB) of per fragment
export_device: cpu
export_legacy_format: false

```

### Inference

```shell
python -m infer.py model_path dataset_path dataset_name gpu_nums
```

Once the inference of the large language model is completed, you will receive a JSON file formatted as follows, where "category" can serve as a pseudo-label generated by the LLM, and "keywords" can be used for sentence-BERT (such as [SimCSE](https://github.com/princeton-nlp/SimCSE)) to generate the initial embeddings that the GNN initially accepts.

For the convenience of other developers, this project has already pre-completed the inference process and the feature generation process of the LLM. In the `dataset_name_filled.json` file under the LLM folder of each dataset, there are pseudo-labels for rough classification and keywords for feature generation, which have been inferred by the fine-tuned LLM. In the GNN folder of each dataset, the `feature_{keywords_num}.pth` contains the initial embeddings using SimCSE.

## Installing dependencies on Ubuntu

GCC version: 7.4.0 or above, support of c++17 standard required.

Install mpi:

```shell
sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev
```

Install glog:

```shell
sudo apt-get install libgoogle-glog-dev
```

Install gflags:

```shell
sudo apt-get install libgflags-dev
```

Install yaml:

```shell
sudo apt-get install libyaml-cpp-dev
```

## Compile

```shell
mkdir build && cd ./build
cmake ../
make all -j
```

## Run

We use libgrape-lite for multi-process parallelism and openmp for multi-thread parallelism.

For LR discovery, to run with single machine, occupying all threads:

```shell
./build/gar_discover --yaml_file ${yaml_file_name}
```

To run with single machine, occupying a specified number of threads:

```shell
mpirun -n 1 -map-by slot:pe=core_num ./build/gar_discover --yaml_file ${yaml_file_name}
```

To run with multiple machines:

```shell
mpirun -N xxx -n yyy -c zzz ./build/gar_discover --yaml_file ${yaml_file_name}
```

For rule match, to run with single machine, occupying all threads:

```shell
./build/rule_match --yaml_file ${yaml_file_name}
```

The others are same as discovery.

The main loop of raw level-wise discovery can be found in folder LMiner/src/apps/rule_discover/, and the main loop of LR match, pattern matching can be found in folder LMiner/src/apps/rule_match/.

The ER folder contains code for computing 1-WL, where you can employ any feature (such as SimCSE or GloVe) embedding to determine whether a pair of points, as well as all pairs within a graph. 

Detailed running examples can be viewed within this folder. 

If you want to run LR discovery algorithm, you may need to fill a yaml file in this format:

```yaml
DataGraphPath: # the path for the data graphs
  - VFile: the vertex file for the first data graph
    EFile: the edge file for the first data graph
    MlLiteralEdgesFile: (optional) the edges that are added by the well-trained ml model for the first data graph
  ...
ExpandRound: number of expand round, i.e. total edges to be added
J: depth of the literal tree for horizontal spawning.
SupportBound: the support bound for the gar to be discovered
OutputGarDir: the directory for the discovered gar to export
TimeLimit: time limit for evaluating the support bound of each gar or graph pattern
TimeLimitPerSupp: time limit for it to complete the match of the entire pattern of gar at each support
ConstantFreqBound: the frequency bound for the constant, only the value appear larger than this frequence would be considered
PatternVertexLimit: the limit of pattern vertex
DiameterLimit: the limit of the diameter of the graph pattern
LiteralTypes: # the literal types to be considered
  - constant_literal
  - variable_literal
  - edge_literal
Restrictions: # the restrictions for the gar
  - variable_literal_only_between_connected_vertexes
  - edge_literal_only_between_2_hop_connected_vertexes
  - literals_connected
  - pattern_without_loop
SpecifiedRhsLiteralSet:
  - Type: variable_literal
    XLabel: label of x
    YLabel: label of y
    XAttrKey: attr of x
    YAttrKey: attr of y
TimeLogFile: the path for the time log file
```

An example yaml file for LR discovery may like this:

```yaml
DataGraphPath:
  VFile : dataset/v.csv
  EFile : dataset/e.csv
ExpandRound: 15
J: 3
LiteralTypes:
  - constant_literal
  - variable_literal
  - edge_literal

SupportBound: 1
ConfidenceBound: 0.4

Rule:
  Type: gcr
  PathNumLimit: 3
  PathLengthLimit: 5
  
SpecifiedRhsLiteralSet:
  - Type: variable_literal
    XLabel: 3
    YLabel: 3
    XAttrKey: year
    YAttrKey: year
  
TimeLogFile:  dataset/lr.log
OutputGarDir: dataset/lr

TimeLimit: 3000
TimeLimitPerSupp: 0.5
ConstantFreqBound: 0.09
```

If you want to run LR pattern matching algorithm, you may need to fill a yaml file in this format:

```yaml
DataGraphPath: 
  VFile : vertex file of the data graph
  EFile : edge file of the data graph
  
PatternPath:
  VFile : vertex file of the pattern
  EFile : edge file of the pattern
  XFile : X (lhs) literal file of the pattern
  YFile : Y (rhs) literal file of the pattern
  PivotId : (optional) specify the pivot vertex id, needs to be contained in the Y literals of the pattern

TimeLogFile: time log file
```

## Literal CSV format

### Literals

| type | x_id | x_attr | y_id | y_attr | edge_label |  c   |
| :--: | :--: | :----: | :--: | :----: | :--------: | :--: |

Different kinds of literals would use different columns.

### Attribute literal

Format:

|   type    | x_id | x_attr | y_id | y_attr | edge_label |  c   |
| :-------: | :--: | :----: | :--: | :----: | :--------: | :--: |
| Attribute |  x   |   A    |  -   |   -    |     -      |  -   |

Semantics:

```
x.A
```

Vertex *x* has attribute *A*.

### Variable literal

Format:

|   type   | x_id | x_attr | y_id | y_attr | edge_label |  c   |
| :------: | :--: | :----: | :--: | :----: | :--------: | :--: |
| Variable |  x   |   A    |  y   |   B    |     -      |  -   |

Semantics:

```
x.A = y.B
```

The attribute *A* of *x* is the same as attribute *B* of *y*.

### Constant literal

Format:

|   type   | x_id | x_attr | y_id | y_attr | edge_label |  c   |
| :------: | :--: | :----: | :--: | :----: | :--------: | :--: |
| Constant |  x   |   A    |  -   |   -    |     -      |  c   |

Semantics:

```
x.A = c
```

The attribute *A* of *x* is equal to *c*.

More detailly, the data type of *c* in constant literal needs to be specified as the following example:

> |   type   | x_id | x_attr | y_id | y_attr | edge_label |        c        |
> | :------: | :--: | :----: | :--: | :----: | :--------: | :-------------: |
> | Constant |  4   | genres |  -   |   -    |     -      | \|Comedy;string |

Which specifies that the attribute *genres* of vertex with id *4* is equal to *|Comedy* in string.

### Edge literal

Format:

| type | x_id | x_attr | y_id | y_attr | edge_label |  c   |
| :--: | :--: | :----: | :--: | :----: | :--------: | :--: |
| Edge |  x   |   -    |  y   |   -    |     l      |  -   |

Semantics:

```
x -(l)-> y
```

There is an edge with label *l* from vertex *x* to *y*.

### ML literal

Format:

| type | x_id | x_attr | y_id | y_attr | edge_label |  c   |
| :--: | :--: | :----: | :--: | :----: | :--------: | :--: |
|  Ml  |  x   |   -    |  y   |   -    |     l      |  -   |

Semantics:

```
x - ml(l) -> y
```

The ML model can predict an edge with label *l* from vertex *x* to *y*.

## Set of literals

The above *literal format* allow users to store multiply literals in the same file, and the *literal set format* further allows to divide the literals into different sets by an additional column *gar_id* :

| type | x_id | x_attr | y_id | y_attr | edge_label |  c   | **gar_id** |
| :--: | :--: | :----: | :--: | :----: | :--------: | :--: | :--------: |

As an example, the following example shows that there are two literals in the *example_x.csv*:

> |   type   | x_id | x_attr | y_id | y_attr | edge_label |  c   |
> | :------: | :--: | :----: | :--: | :----: | :--------: | :--: |
> | Constant |  x   |   A    |  -   |   -    |     -      |  c   |
> |   Edge   |  x   |   -    |  y   |   -    |     l      |  -   |

By adding the additional column *gar_id*, the following *example_x_set.csv* represent there are two single literal:

> |   type   | x_id | x_attr | y_id | y_attr | edge_label |  c   | gar_id |
> | :------: | :--: | :----: | :--: | :----: | :--------: | :--: | :----: |
> | Constant |  x   |   A    |  -   |   -    |     -      |  c   |   0    |
> |   Edge   |  x   |   -    |  y   |   -    |     l      |  -   |   1\   |

[head file](/include/gar/csv_gar.h):

```
/include/gar/csv_gar.h
```

## CSV file format

Both a single GAR and a set of GARs are stored in four files seperately:

* v.csv / v_set.csv

  > The vertexes of Gar / Gar set, see [csv format of graph](/include/gundam/doc/user_doc/csv_format.md) in GUNDAM.

* e.csv / e_set.csv

  > The edges of Gar / Gar set, see [csv format of graph](/include/gundam/doc/user_doc/csv_format.md) in GUNDAM.

* x.csv / x_set.csv

  > The set of literals contained in X of Gar / Gar set, see [csv format of literal](/doc/user_doc/literal_csv_format.md).

* y.csv / y_set.csv

  > The set of literals contained in Y of Gar / Gar set, see [csv format of literal](/doc/user_doc/literal_csv_format.md).

## Useful method

### Read

```c++
template <typename PatternType, typename DataGraphType>
int ReadGAR(GraphAssociationRule<PatternType, DataGraphType> &gar,
            const std::string &v_file, const std::string &e_file,
            const std::string &x_file, const std::string &y_file);
```

### ReadSet

```c++
template <typename PatternType, typename DataGraphType>
int ReadGARSet(
    std::vector<GraphAssociationRule<PatternType, DataGraphType>> &gar_set,
    const std::string &v_set_file, const std::string &e_set_file,
    const std::string &x_set_file, const std::string &y_set_file);

template <typename PatternType, typename DataGraphType>
int ReadGARSet(
    std::vector<GraphAssociationRule<PatternType, DataGraphType>> &gar_set,
    std::vector<std::string> &gar_name_set, 
    const std::string &v_set_file, const std::string &e_set_file, 
    const std::string &x_set_file, const std::string &y_set_file);
```

### Write

```c++
template <typename PatternType, typename DataGraphType>
int WriteGAR(
    const GraphAssociationRule<PatternType, DataGraphType> &gar,
    const std::string &v_file, const std::string &e_file,
    const std::string &x_file, const std::string &y_file);
```

### WriteSet

```c++
template <typename PatternType, typename DataGraphType>
int WriteGARSet(
    const std::vector<GraphAssociationRule<PatternType, DataGraphType>> &gar_set,
    const std::string &v_file, const std::string &e_file,
    const std::string &x_file, const std::string &y_file);

template <typename PatternType, typename DataGraphType>
int WriteGARSet(
    const std::vector<GraphAssociationRule<PatternType, DataGraphType>> &gar_set,
    const std::vector<std::string> &gar_name_list, 
    const std::string &v_file, const std::string &e_file, 
    const std::string &x_file, const std::string &y_file);
```

#### RxGNNs

```python
from NEX import LevelWiseParallelMiner
from graph_matcher import Graph

miner = LevelWiseParallelMiner(
    train_graph=train_graph,
    valid_graph=valid_graph,
    motifs=motifs,
    support_threshold=200,
    confidence_threshold=0.6,
    max_threads=8
)

rules = miner.mine()
print(f"Discovered {len(rules)} RxGNNs")
```

This example shows the streamlined rule discovery process. After loading graph data with GNN predictions, instantiate the parallel miner with threshold parameters and motif templates. 

The `mine()` method orchestrates motif composition via DQN, predicate generation with optional LLM ordering (implementation in `NEX/llmsft/ppl_generator` directory), and level-wise rule enumeration with anti-monotonicity pruning to extract logical patterns correlating graph structures and node features with negative GNN predictions.

##### Counterfactual Explanation

```python
from CFE import CounterfactualExplainer, generate_counterfactual_explanation

# Quick usage
attr_changes, edge_removals = generate_counterfactual_explanation(
    subgraph,
    rxgnn_rule
)

# Advanced configuration
explainer = CounterfactualExplainer(
    alpha=0.5,              # Distance decay (α^K)
    lambda_factor=1000.0,   # Edge cost
    num_clusters=5          # Clustering granularity
)

attr_changes, edge_removals = explainer.explain(subgraph, rxgnn_rule)

print(f"Modify attributes: {attr_changes}")
print(f"Remove edges: {edge_removals}")
```

The explainer computes minimal perturbations to flip GNN predictions by invalidating all applicable RxGNNs at a target vertex. It constructs a DAG-refined candidate space, then iteratively selects cost-effective perturbations using GINI-based attribute importance and cascade reduction analysis until no rules remain satisfied.

##### Deep Q-Network for Pattern Composition

The DQN-guided composition learns optimal motif merging strategies through reinforcement learning. The system frames pattern composition as a sequential decision problem: given a current pattern and a candidate motif, the DQN selects which pair of vertices to merge while preserving structural constraints.

The merging process identifies all valid vertex pairs between pattern and motif that share the same label, then uses the Q-network to score each merge action. The action with highest Q-value is executed, creating a new pattern by unifying the selected vertices and combining their neighborhoods:

```python
def merge_with_dqn(self, pattern, motif):
    state = self._encode_state(pattern)
    merge_actions = self._get_valid_merge_actions(pattern, motif)
    
    q_values = self.q_network(state, actions)
    best_action = merge_actions[q_values.argmax()]
    
    return self._execute_merge(pattern, motif, best_action)
```

Rewards are computed based on confidence improvement and support preservation of the merged pattern. Experience replay and target networks stabilize training, enabling efficient navigation of exponential composition spaces without exhaustive enumeration. The learned policy prioritizes discriminative patterns that distinguish negative predictions while maintaining sufficient frequency.

##### DAG-Based Candidate Space Refinement

The counterfactual explainer employs bidirectional constraint propagation (`_refine_candidate_space` in to efficiently prune invalid pattern matches. Starting from an initial candidate space filtered by label and degree constraints, the system builds a rooted DAG from the pattern's pivot node, then alternates forward and backward dynamic programming passes:

```python
def _refine_candidate_space(self, cand_space, pattern_graph, dag, reverse_dag):
    refined = deepcopy(cand_space)
    
    for iteration in range(max_iterations):
        # Forward: propagate constraints from root to leaves
        refined = self._dag_graph_dp(refined, pattern_graph, dag)
        
        # Backward: propagate constraints from leaves to root
        refined = self._dag_graph_dp(refined, pattern_graph, reverse_dag)
        
        if converged(refined):
            break
    
    return refined
```

Each DP pass (`_dag_graph_dp`) processes nodes in topological order, removing data vertices that lack valid mappings for their children (forward) or parents (backward) in the pattern. This bidirectional propagation efficiently enforces edge constraints and variable predicates, dramatically reducing the search space for homomorphic matches from exponential to polynomial complexity. The refinement continues until the candidate space stabilizes or becomes empty.

##### Effectiveness Calculation with Cascade Reduction

The counterfactual explainer implements the paper's effectiveness formula (`_calculate_effectiveness_with_cascade` in `CFE.py` ) that captures cascading effects of candidate space refinement:

```python
def _calculate_effectiveness_with_cascade(self, original_cand, new_cand, ...):
    distance = self._compute_distance(pattern_graph, node_id, pivot_id)
    distance_factor = self.alpha ** distance
    
    reduction_ratios = [
        len(new_cand[x]) / len(original_cand[x])
        for x in original_cand if len(original_cand[x]) > 0
    ]
    
    min_ratio = min(reduction_ratios) if reduction_ratios else 1.0
    return distance_factor * (1.0 - min_ratio)
```

This computation follows `eff(v) = α^K(1 - min_x|C_Δ(x)|/|C(x)|)`, where K is the distance to pivot, C_Δ is the refined candidate space after removing vertex v, and the minimum ratio captures the bottleneck constraint propagation effect. By measuring the tightest constraint across all pattern variables, the formula identifies perturbations that most effectively invalidate pattern matches through cascading refinement.

##### GINI-Based Cost Estimation

Attribute modification costs are determined by GINI importance scores measuring how well attributes discriminate GNN predictions:

```python
def _calculate_gini(self, attr_values, gnn_labels):
    original_gini = self._calculate_group_gini(gnn_labels)
    
    best_gain = 0.0
    for split_value in unique_values:
        left_labels = [labels where value <= split]
        right_labels = [labels where value > split]
        
        weighted_gini = p_left * gini_left + p_right * gini_right
        gain = original_gini - weighted_gini
        best_gain = max(best_gain, gain)
    
    return min(1.0, best_gain / original_gini)
```

Higher GINI importance indicates attributes more critical to GNN decision boundaries, translating to higher modification costs (normalized to [1.0, 10.0] range). This ensures the explainer prioritizes feasible perturbations that preserve important discriminative features.

##### 1-WL Test Implementation

The 1-WL predicate evaluation uses an efficiency-optimized local approximation. Rather than computing global color refinement across the entire graph, the implementation extracts a 3-hop neighborhood around the target vertex and performs one iteration of color refinement:

```python
def evaluate(self, mapping, data_graph, query_pattern):
    data_id = mapping[self.node_id]
    
    # Extract 3-hop neighborhood for efficiency
    local_nodes = self.extract_local_neighborhood(data_graph, data_id, max_hops=3)
    local_graph = self.create_local_subgraph(data_graph, local_nodes)
    
    # Single iteration color refinement
    node_colors = self.compute_local_wl_colors(local_graph)
    
    # Check if any same-color vertex has different GNN prediction
    return self._check_color_gnn_discrepancy(node_colors, data_graph)
```

This local approximation balances computational efficiency with expressiveness for typical graph structures. 

For applications requiring exact 1-WL equivalence testing, remove the `max_hops=3` limitation in `extract_local_neighborhood`  and iterate color refinement until stabilization in `compute_local_wl_colors`.

# Acknowledgements

This project has benefited from the following open-source projects, to which we extend our gratitude:

- **torch_geometric**: https://github.com/pyg-team/pytorch_geometric
- **Mistral**: https://huggingface.co/mistralai
- **Qwen**: https://huggingface.co/Qwen
- **SimCSE**: https://github.com/princeton-nlp/SimCSE
- **vllm**: https://github.com/vllm-project/vllm
- **LLaMA-Factory**: https://github.com/hiyouga/LLaMA-Factory
- **GUNDAM**: https://github.com/MinovskySociety/GUNDAM

We acknowledge these contributions, which have significantly facilitated the development of our project.
