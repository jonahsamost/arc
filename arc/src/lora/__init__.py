"""
LoRA + Test-Time Training module for ARC puzzle solving.

This module provides:
- LoRA adapters for efficient fine-tuning
- Test-Time Training (TTT) for per-puzzle adaptation
- Inference with thinking prompts and best-of-N verification
"""

from src.lora.config import (
    LoRAConfig,
    TTTConfig,
    InferenceConfig,
    LoRATTTConfig,
    LoRATrainConfig,
)

from src.lora.lora_model import (
    LoRALinear,
    apply_lora_to_model,
    get_lora_params,
    get_lora_state_dict,
    load_lora_state_dict,
    save_lora,
    load_lora,
    freeze_base_model,
    count_lora_params,
)

from src.lora.ttt import (
    ttt_adapt,
    ttt_adapt_and_load,
)

from src.lora.inference import (
    generate_with_thinking,
    extract_grid_from_tokens,
    extract_grid_from_output,
    predict_with_verification,
    verify_prediction,
)

__all__ = [
    # Config
    "LoRAConfig",
    "TTTConfig", 
    "InferenceConfig",
    "LoRATTTConfig",
    "LoRATrainConfig",
    # LoRA model
    "LoRALinear",
    "apply_lora_to_model",
    "get_lora_params",
    "get_lora_state_dict",
    "load_lora_state_dict",
    "save_lora",
    "load_lora",
    "freeze_base_model",
    "count_lora_params",
    # TTT
    "ttt_adapt",
    "ttt_adapt_and_load",
    # Inference
    "generate_with_thinking",
    "extract_grid_from_tokens",
    "extract_grid_from_output",
    "predict_with_verification",
    "verify_prediction",
]
