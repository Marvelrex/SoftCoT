"""Baseline SFT configuration for SoftCoT.

This mirrors the BRIDGE SFT setup the user described:
- LoRA is always enabled when tuning.
- Target modules: q_proj, k_proj, v_proj, o_proj.
- LoRA hyperparameters: r=64, alpha=128, dropout=0.05.
- Default base model: meta-llama/Llama-3.1-8B-Instruct.
- Default tokenizer: meta-llama/Llama-3.1-8B-Instruct.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SoftCOTSFTConfig:
    """Container for baseline SFT / LoRA settings used across SoftCoT."""

    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer_name: str = "meta-llama/Llama-3.1-8B-Instruct"


# Shared instance so callers can import a single object without re-declaring values.
DEFAULT_SFT_CONFIG = SoftCOTSFTConfig()
