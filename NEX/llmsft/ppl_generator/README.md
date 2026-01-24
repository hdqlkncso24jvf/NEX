# PPL-Guided Predicate Ordering for Rule Mining

## Table of Contents

1. [Overview](#overview)
2. [Supported Language Models](#supported-language-models)
   - [Primary Model: Qwen2.5-3B](#primary-model-qwen25-3b-default)
   - [Optional Models](#optional-models)
3. [Prompt Design](#prompt-design)
5. [Adapting to New Domains](#adapting-to-new-domains)
6. [Acknowledgments](#acknowledgments)

------

## Overview

This directory contains the implementation of predicate generation using Large Language Models (LLMs) for explainable graph neural network predictions. The PPL mechanism leverages LLM perplexity scores to rank predicates by their discriminative power for classification tasks.

The code demonstrates a **two-stage prompting strategy** where Stage 1 assesses attribute importance for category discrimination and Stage 2 determines optimal predicate directions (increasing/decreasing for ordinal attributes, positive/negative for binary attributes). The two-stage design prevents situations where highly correlated attributes that negatively correlate with GNN predictions are ranked lower than less correlated attributes that positively correlate with predictions. This approach bridges symbolic rule mining with neural language understanding, enabling interpretable AI systems that explain predictions through human-readable logical rules.

Pre-computed PPL rankings are provided as `.pkl` files for all four datasets, enabling direct reproduction of experimental results without re-running the computationally expensive LLM inference process. The prompt generation and PPL computation scripts are located in the `code/` directory, while the pre-computed ranking files are stored in the `pkl/` directory. 

To enable one-click execution with pre-computed rankings, place the files under `pkl/` folder at the same directory level as NEX.py. If the `pkl` files are not found in the expected location, the system will automatically fall back to random or support-based predicate ordering by default.

------

## Supported Language Models

### Primary Model: Qwen2.5-3B (Default)

**Qwen2.5-3B-Instruct** is the default LLM used in our experiments, chosen for its exceptional balance between computational efficiency and stability.

#### Technical Specifications

The model contains 3.09 billion parameters with 2.77 billion non-embedding parameters. It employs a 36-layer Dense Transformer architecture with SwiGLU activation functions. The instruction-tuned variant supports a 32,768 token context window, though the base model can handle up to 128,000 tokens. Training utilized 18 trillion tokens across 29+ languages under the Apache 2.0 license, making it freely available for commercial deployment.

#### Environment Requirements

```bash
# Python 3.10+
pip install torch>=2.4.0 transformers>=4.37.0
pip install flash-attn  # Recommended for efficiency
```

#### Quick Start: Perplexity Calculation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math

# Load model
model_name = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def compute_perplexity(text: str) -> float:
    """
    Compute perplexity for a given text sequence.
    Lower perplexity indicates higher model confidence.
    """
    inputs = tokenizer(text, return_tensors="pt", max_length=512, 
                      truncation=True, padding=True).to(model.device)
    
    with torch.no_grad():
        # The model uses standard cross-entropy loss, equivalent to setting temperature (tau) = 1.
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = math.exp(loss.item())
    
    return perplexity

# Example usage
statement = "When transaction amount exceeds $10,000, fraud probability increases significantly."
ppl = compute_perplexity(statement)
print(f"Perplexity: {ppl:.2f}")
```

**Resources:**

- Hugging Face: [`https://huggingface.co/Qwen/Qwen2.5-3B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- ArXiv: *Qwen2.5 Technical Report* ([arXiv:2412.15115](https://arxiv.org/abs/2412.15115))

------

### Optional Models

#### Meta Llama 3.2-3B

Llama 3.2-3B is optimized for edge deployment with exceptional mobile device performance. The model contains 3.21 billion parameters and employs Dense Transformer architecture with Grouped-Query Attention (GQA) to optimize KV cache memory usage. A distinctive feature is its 128,000 token context window, trained on approximately 9 trillion tokens. The model operates under the Llama 3.2 Community License, which permits commercial use subject to monthly active user thresholds. The ultra-long context support and optimized memory footprint make it particularly suitable for on-device deployment scenarios. Implementation support exists across major inference frameworks including Ollama, llama.cpp, and vLLM from launch day.

**Resources:**

- Hugging Face: [`https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- ArXiv: *The Llama 3 Herd of Models* ([arXiv:2407.21783](https://arxiv.org/abs/2407.21783))

#### Microsoft Phi-3.5-mini (3.8B)

Phi-3.5-mini represents Microsoft's synthetic data approach to small language model training. With 3.8 billion parameters in a Dense Decoder-only Transformer architecture, the model achieves 128,000 token context length while training on only 3.4 trillion tokens of high-quality synthetic data generated by larger teacher models. Released under the permissive MIT License, Phi-3.5 demonstrates superior mathematical reasoning capabilities, often matching or exceeding larger 8B parameter models on logical reasoning benchmarks like GSM8K. The model's training methodology emphasizes textbook-quality synthetic data over raw web scraping, resulting in particularly strong performance on structured reasoning tasks. First-class ONNX Runtime support enables efficient deployment on Windows AI PC platforms.

**Resources:**

- Hugging Face: [`https://huggingface.co/microsoft/Phi-3.5-mini-instruct`](https://huggingface.co/microsoft/Phi-3.5-mini-instruct)
- ArXiv: *Phi-3 Technical Report* ([arXiv:2404.14219](https://arxiv.org/abs/2404.14219))

#### Google Gemma 3-4B

Gemma 3-4B pioneers native multimodal capabilities in the small language model space. The 4 billion parameter model integrates a SigLIP vision encoder with a multimodal transformer decoder, enabling simultaneous processing of image and text inputs. Trained on 4 trillion tokens of mixed text and image data, it supports 128,000 token context windows with multiple image inputs. The architecture employs alternating local sliding window attention and global attention mechanisms to efficiently handle long multimodal sequences. Native image understanding eliminates the need for separate vision adapter modules required by text-only models. The model operates under Google's Gemma Terms of Use, which permits responsible commercial deployment. TPU-optimized training leveraging JAX and Pathways infrastructure enables efficient large-scale multimodal pre-training.

**Resources:**

- Hugging Face: [`https://huggingface.co/google/gemma-3-4b-it`](https://huggingface.co/google/gemma-3-4b-it)
- ArXiv: *Gemma 3 Technical Report* ([arXiv:2503.19786](https://arxiv.org/abs/2503.19786))

## Prompt Design

The PPL framework employs a two-stage prompting architecture to rank predicates by their discriminative power. We provide two prompt complexity modes to accommodate different deployment constraints and accuracy requirements.

### Two-Stage Architecture Overview

The ranking process operates through two sequential stages using the loan default prediction dataset as illustration.

**Stage 1: Attribute Importance Assessment**

This stage evaluates which attributes most strongly indicate the target prediction outcome. The system constructs declarative statements asserting that specific attribute patterns signal loan default risk, then measures model perplexity to quantify the naturalness of each assertion. For the loan dataset, Stage 1 might evaluate statements such as "When an applicant's income level falls into the lowest brackets, this limited earning capacity serves as strong evidence of elevated default risk." Lower perplexity indicates the model finds this causal relationship more plausible based on its training data, suggesting higher discriminative importance for the income_level attribute.

**Stage 2: Value Direction Determination**

After identifying important attributes, Stage 2 determines optimal predicate directionality for ordinal attributes by comparing competing directional hypotheses. For job_years_level, the system evaluates "As job tenure INCREASES, default probability rises" versus "As job tenure DECREASES, default probability rises." If the decreasing direction achieves lower perplexity, the framework prioritizes predicates like `job_years_level < 2` asserting short tenure. Conversely, if increasing direction wins, predicates like `job_years_level >= 3` receive priority. When both directions yield similar high perplexity, indicating no monotonic relationship, the system generates predicates for all quantile thresholds as the attribute may have non-linear effects. This two-stage design prevents highly correlated attributes that negatively correlate with GNN predictions from ranking below less correlated attributes with positive correlation.

### Prompt Mode Comparison

The distinction between Simple and Complex modes manifests primarily in Stage 1 attribute importance prompts, as illustrated by the insurance fraud detection dataset.

**Simple Mode: Concise Assertions (100-150 words)**

Simple prompts provide direct domain context followed by immediate predicate insertion and categorical assertion. The reimb_level attribute exemplifies this approach:

```
In insurance fraud investigation, investigators analyze claim patterns to identify 
fraudulent behavior. The investigation process examines various data points to 
assess fraud risk.

When claim reimbursement amounts significantly exceed typical billing patterns and 
reach the top 10% of all payouts (scaled 1-5, with level 4-5 indicating extreme 
values), these financial irregularities serve as strong evidence that the billing 
is likely fraudulent.
```

This format establishes minimal context then directly asserts the fraud indicator relationship. The brevity reduces token consumption and inference latency while maintaining sufficient semantic content for perplexity differentiation.

**Complex Mode: Detailed Contextualization (400-500 words)**

Complex prompts embed the same assertion within extensive domain knowledge and reasoning chains. The reimb_level attribute under Complex mode demonstrates this expansion:

```
[Context: Healthcare Insurance Graph Ecosystem]
The insurance reimbursement system is modeled as a heterogeneous graph interconnecting 
four key entities: Claims (central financial requests), Beneficiaries (insured patients 
with demographics and medical history), Providers (physicians billing for services), 
and DiagnosisGroups (medical classification codes justifying service necessity).

Fraud detection within this network aims to identify attribute patterns—such as phantom 
billing, unbundling, or kickback schemes—that violate medical plausibility or economic 
logic.

[Attribute Analysis: reimb_level]
The attribute 'reimb_level' places the claim's payment amount into a quintile scale (1-5). 
Level 5 represents the top 20% of most expensive claims, typically reserved for 
high-complexity interventions (neurosurgery, transplants) rather than routine care.

[Forensic Reasoning]
Financial fraud schemes generally aim to maximize profit density. Techniques like 
'upcoding' (billing for a higher tier than performed) or 'unbundling' (separating bundled 
procedures to charge multiple times) push claims into the highest reimbursement brackets 
unnecessarily. Therefore, an extreme 'reimb_level' that lacks corresponding diagnostic 
severity acts as a primary indicator of billing manipulation. Historical audit data from 
the Department of Health and Human Services reveals that claims in the top decile show 
5-8x higher fraud adjudication rates when cross-referenced with medical necessity reviews.
```

The Complex version provides structured sections detailing graph topology, attribute semantics, and fraud mechanism theory. This additional context allows the language model to leverage deeper reasoning chains when computing perplexity, potentially improving discrimination between genuinely important attributes and spurious correlations.

Simple mode is appropriate for production deployments where inference budgets are constrained, edge devices limit computational resources, or high-throughput scenarios require minimal latency. The concise format maintains core semantic relationships while minimizing token costs. Complex mode becomes preferable when the model demonstrates difficulty distinguishing domain-specific patterns, when regulatory requirements demand maximal accuracy for high-stakes decisions, or when offline batch processing allows extended inference time. The rich contextual scaffolding helps models trained on general web corpora better align with specialized domain knowledge, yielding more reliable attribute importance rankings at the cost of increased computational overhead.

------

## Adapting to New Domains

When extending this framework to new datasets or domains, the prompt design differs fundamentally from traditional LLM prompts that expect generated responses. Instead, perplexity-based ranking requires prompts structured as declarative statements that the model evaluates for plausibility. The prompt should be constructed in three components: domain context introduction, predicate insertion, and categorical assertion.

Consider a molecular toxicity prediction scenario. The prompt structure would proceed as follows. First, the domain context establishes the graph representation: "Pharmaceutical safety assessment models molecular structures as attributed graphs where nodes represent atoms with features such as electronegativity, hybridization state, and formal charge, while edges encode bond types and stereochemistry." Second, the predicate is inserted into the context: "When a molecule contains a nitrogen atom with formal_charge >= +1 in proximity to an aromatic ring system..." Third, the categorical assertion completes the statement: "...this structural motif indicates elevated hepatotoxicity risk, as quaternary ammonium groups adjacent to aromatic systems exhibit 4-6x higher liver enzyme elevation rates in clinical trials compared to neutral nitrogen-containing compounds."

The key principle is that prompts must form complete, evaluable propositions rather than questions or instructions. Traditional prompts might ask "Is this molecule toxic?" or instruct "Classify the toxicity level." In contrast, perplexity-based prompts assert "This molecule is toxic because [predicate]," allowing the LLM to assign confidence scores based on how naturally this assertion fits its learned knowledge distribution.

Domain context framing should establish the graph structure and decision-making context. For instance: "The [domain] system models [entity relationships] as a [graph type]. Each [entity] carries attributes reflecting [domain concepts]." When addressing discrimination tasks, explicitly contrast the target category with alternatives: "When distinguishing [Target Class] from alternatives ([Other Classes]), category-specific patterns like [examples] provide stronger discrimination than universal indicators like [generic features]."

Quantification strengthens prompt effectiveness. Rather than stating "This attribute is useful for classification," precise statements like "Products in this cluster show 4-6x over-representation in [Target], while [Alternative] concentrates in clusters [X-Y]" provide the specificity needed for accurate perplexity differentiation. Grounding reasoning in established domain theory further improves results. For example: "Financial theory establishes that debt-to-income ratios >40% increase default probability—validated by [specific events/studies]."

Machine learning-derived features require careful explanation. For embedding clusters or ML-generated attributes, prompts should clarify: "The ML model groups [entities] into 10 semantic clusters (0-9) based on [method]. Clusters 0-3 capture [pattern type], clusters 4-6 capture [pattern], and clusters 7-9 capture [pattern]." This contextualization allows the LLM to reason about abstract feature spaces.

Finally, avoid generic quality signals that apply equally across categories. Attributes like "high rating" or "verified status" provide weak discrimination because they positively indicate quality in all categories rather than distinguishing between them. The ranking algorithm should penalize such attributes through higher baseline perplexity scores, while category-specific patterns receive priority through lower perplexity penalties.

------

## Acknowledgments

We gratefully acknowledge the developers of the language models used in this work. 

* The Qwen Team at Alibaba Cloud provided the exceptional Qwen2.5 series with comprehensive technical documentation. 
* Meta AI contributed the open-source Llama 3.2 models and demonstrated commitment to responsible AI development. 
* Microsoft Research developed the innovative Phi-3.5 series that showcases the potential of synthetic data training methodologies. 
* Google DeepMind pioneered multimodal small language models with the Gemma 3 series. 

All models are used in accordance with their respective licenses. Please refer to individual model repositories for detailed terms of use.
