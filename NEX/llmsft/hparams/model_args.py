from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union

from typing_extensions import Self


if TYPE_CHECKING:
    import torch


@dataclass
class ModelArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to the adapter weight or identifier from huggingface.co/models. "
                "Use commas to separate multiple adapters."
            )
        },
    )
    adapter_folder: Optional[str] = field(
        default=None,
        metadata={"help": "The folder containing the adapter weights to load."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    resize_vocab: bool = field(
        default=False,
        metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={"help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    new_special_tokens: Optional[str] = field(
        default=None,
        metadata={"help": "Special tokens to be added into the tokenizer. Use commas to separate multiple tokens."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether or not to use memory-efficient model loading."},
    )
    quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the model using bitsandbytes."},
    )
    quantization_type: Literal["fp4", "nf4"] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in int4 training."},
    )
    double_quantization: bool = field(
        default=True,
        metadata={"help": "Whether or not to use double quantization in int4 training."},
    )
    quantization_device_map: Optional[Literal["auto"]] = field(
        default=None,
        metadata={"help": "Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0."},
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "Which scaling strategy should be adopted for the RoPE embeddings."},
    )
    flash_attn: Literal["off", "sdpa", "fa2", "auto"] = field(
        default="auto",
        metadata={"help": "Enable FlashAttention for faster training and inference."},
    )
    shift_attn: bool = field(
        default=False,
        metadata={"help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."},
    )
    mixture_of_depths: Optional[Literal["convert", "load"]] = field(
        default=None,
        metadata={"help": "Convert the model to mixture-of-depths (MoD) or load the MoD model."},
    )
    use_unsloth: bool = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's optimization for the LoRA training."},
    )
    visual_inputs: bool = field(
        default=False,
        metadata={"help": "Whethor or not to use multimodal LLM that accepts visual inputs."},
    )
    moe_aux_loss_coef: Optional[float] = field(
        default=None,
        metadata={"help": "Coefficient of the auxiliary router loss in mixture-of-experts model."},
    )
    disable_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )
    upcast_layernorm: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the layernorm weights in fp32."},
    )
    upcast_lmhead_output: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the output of lm_head in fp32."},
    )
    train_from_scratch: bool = field(
        default=False,
        metadata={"help": "Whether or not to randomly initialize the model weights."},
    )
    infer_backend: Literal["huggingface", "vllm"] = field(
        default="huggingface",
        metadata={"help": "Backend engine used at inference."},
    )
    vllm_maxlen: int = field(
        default=2048,
        metadata={"help": "Maximum sequence (prompt + response) length of the vLLM engine."},
    )
    vllm_gpu_util: float = field(
        default=0.9,
        metadata={"help": "The fraction of GPU memory in (0,1) to be used for the vLLM engine."},
    )
    vllm_enforce_eager: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable CUDA graph in the vLLM engine."},
    )
    vllm_max_lora_rank: int = field(
        default=32,
        metadata={"help": "Maximum rank of all LoRAs in the vLLM engine."},
    )
    offload_folder: str = field(
        default="offload",
        metadata={"help": "Path to offload model weights."},
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="auto",
        metadata={"help": "Data type for model weights and activations at inference."},
    )
    hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    ms_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with ModelScope Hub."},
    )
    export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."},
    )
    export_size: int = field(
        default=1,
        metadata={"help": "The file shard size (in GB) of the exported model."},
    )
    export_device: Literal["cpu", "auto"] = field(
        default="cpu",
        metadata={"help": "The device used in model export, use `auto` to accelerate exporting."},
    )
    export_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the exported model."},
    )
    export_quantization_dataset: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the dataset or dataset name to use in quantizing the exported model."},
    )
    export_quantization_nsamples: int = field(
        default=128,
        metadata={"help": "The number of samples used for quantization."},
    )
    export_quantization_maxlen: int = field(
        default=1024,
        metadata={"help": "The maximum length of the model inputs used for quantization."},
    )
    export_legacy_format: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the `.bin` files instead of `.safetensors`."},
    )
    export_hub_model_id: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the repository if push the model to the Hugging Face hub."},
    )
    print_param_status: bool = field(
        default=False,
        metadata={"help": "For debugging purposes, print the status of the parameters in the model."},
    )

    def __post_init__(self):
        self.compute_dtype: Optional["torch.dtype"] = None
        self.device_map: Optional[Union[str, Dict[str, Any]]] = None
        self.model_max_length: Optional[int] = None

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError("`split_special_tokens` is only supported for slow tokenizers.")

        if self.visual_inputs and self.use_unsloth:
            raise ValueError("Unsloth does not support MLLM yet. Stay tuned.")

        if self.adapter_name_or_path is not None:  # support merging multiple lora weights
            self.adapter_name_or_path = [path.strip() for path in self.adapter_name_or_path.split(",")]

        if self.new_special_tokens is not None:  # support multiple special tokens
            self.new_special_tokens = [token.strip() for token in self.new_special_tokens.split(",")]

        assert self.quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.export_quantization_bit in [None, 8, 4, 3, 2], "We only accept 2/3/4/8-bit quantization."

        if self.export_quantization_bit is not None and self.export_quantization_dataset is None:
            raise ValueError("Quantization dataset is necessary for exporting.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def copyfrom(cls, old_arg: Self, **kwargs) -> Self:
        arg_dict = old_arg.to_dict()
        arg_dict.update(**kwargs)
        new_arg = cls(**arg_dict)
        new_arg.compute_dtype = old_arg.compute_dtype
        new_arg.device_map = old_arg.device_map
        new_arg.model_max_length = old_arg.model_max_length
        return new_arg
