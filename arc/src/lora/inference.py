"""
Inference utilities for LoRA-adapted ARC models.

Provides:
- Generation with optional thinking prompts
- Grid extraction from model output
- Best-of-N sampling with support set verification
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import re

from src.lora.config import InferenceConfig
from src.data.tokenizer_2d import Arc2DTokenizer


def generate_with_thinking(
    model,
    tokenizer: Arc2DTokenizer,
    puzzle: dict,
    config: InferenceConfig,
    device: torch.device,
) -> Tuple[str, List[int]]:
    """
    Generate output for a puzzle with optional thinking prompt.
    
    Properly tracks grid_mode and 2D positions during generation to match
    training format: text tokens get grid_mode=0, grid pixels get grid_mode=1
    with appropriate 2D coordinates.
    
    Args:
        model: The Qwen model (with LoRA applied and adapted)
        tokenizer: Arc2DTokenizer
        puzzle: Full puzzle dict with 'train' and 'test' keys
        config: Inference configuration
        device: Device to run on
    
    Returns:
        (generated_text, generated_token_ids) - text and token IDs for grid extraction
    """
    model.eval()
    
    # Tokenize in inference mode (stops after "Output:" for test pair)
    sample = tokenizer.build_sample(puzzle, inference_mode=True)
    
    # DEBUG: Show input to model
    input_text = tokenizer.tokenizer.decode(sample["input_ids"].tolist(), skip_special_tokens=False)
    print(f"[GEN INPUT] Num input tokens: {len(sample['input_ids'])}")
    print(f"[GEN INPUT] Last 20 token IDs: {sample['input_ids'][-20:].tolist()}")
    print(f"[GEN INPUT] Last 200 chars of input:\n{input_text[-200:]}")
    
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
    pos_1d = sample["pos_1d"].unsqueeze(0).to(device)
    pos_2d = sample["pos_2d"].unsqueeze(0).to(device)
    grid_mode = sample["grid_mode"].unsqueeze(0).to(device)
    
    # Get special token IDs for tracking grid state
    # Note: structural_ids now stores lists, so we take [0] for single-token special tokens
    grid_start_id = tokenizer.structural_ids[tokenizer.TOK_GRID_START][0]
    grid_end_id = tokenizer.structural_ids[tokenizer.TOK_GRID_END][0]
    newline_id = tokenizer.newline_id
    digit_ids = set(tokenizer.digit_ids)
    
    # Optionally append thinking prompt
    if config.use_thinking:
        think_ids = tokenizer.tokenizer.encode(
            config.thinking_prompt,
            add_special_tokens=False
        )
        think_tensor = torch.tensor([think_ids], dtype=torch.long, device=device)
        
        # Append to input
        input_ids = torch.cat([input_ids, think_tensor], dim=1)
        
        # Extend attention mask
        attention_mask = torch.cat([
            attention_mask,
            torch.ones_like(think_tensor)
        ], dim=1)
        
        # Extend pos_1d (continue sequential positions)
        seq_len = pos_1d.shape[1]
        new_pos_1d = torch.arange(
            seq_len, seq_len + think_tensor.shape[1],
            dtype=torch.long, device=device
        ).unsqueeze(0)
        pos_1d = torch.cat([pos_1d, new_pos_1d], dim=1)
        
        # Extend pos_2d (use 0,0 for text tokens)
        new_pos_2d = torch.zeros(
            (1, think_tensor.shape[1], 2),
            dtype=torch.long, device=device
        )
        pos_2d = torch.cat([pos_2d, new_pos_2d], dim=1)
        
        # Extend grid_mode (0 for text tokens)
        new_grid_mode = torch.zeros_like(think_tensor)
        grid_mode = torch.cat([grid_mode, new_grid_mode], dim=1)
    
    # Store 2D position info on attention layers (like in training)
    _set_2d_positions(model, pos_2d, grid_mode)
    
    # Get stop token IDs
    stop_token_ids = []
    for stop_token in config.stop_tokens:
        if stop_token in tokenizer.structural_ids:
            # structural_ids stores lists, take first (and only) element for special tokens
            stop_token_ids.append(tokenizer.structural_ids[stop_token][0])
        else:
            # Try encoding the token
            ids = tokenizer.tokenizer.encode(stop_token, add_special_tokens=False)
            if ids:
                stop_token_ids.append(ids[-1])
    
    # Also stop on EOS
    if tokenizer.tokenizer.eos_token_id is not None:
        stop_token_ids.append(tokenizer.tokenizer.eos_token_id)
    
    # Track grid state for proper position encoding during generation
    in_grid = False
    grid_row = 0
    grid_col = 0
    
    # Convert pos_2d and grid_mode to lists for easy extension
    pos_2d_list = pos_2d[0].tolist()  # List of [y, x] pairs
    grid_mode_list = grid_mode[0].tolist()  # List of 0/1 values
    
    # Generate autoregressively
    generated_ids = input_ids.clone()
    generated_token_list = []  # Track just the new tokens
    
    # Clear CUDA cache before generation loop
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    with torch.no_grad():
        for _ in range(config.max_new_tokens):
            # Update positions for new length
            current_len = generated_ids.shape[1]
            current_pos_1d = torch.arange(current_len, dtype=torch.long, device=device).unsqueeze(0)
            
            # Convert lists to tensors for this forward pass
            current_pos_2d = torch.tensor([pos_2d_list], dtype=torch.long, device=device)
            current_grid_mode = torch.tensor([grid_mode_list], dtype=torch.long, device=device)
            
            # Update attention layers with position info
            _set_2d_positions(model, current_pos_2d, current_grid_mode)
            
            # Forward pass
            outputs = model(
                input_ids=generated_ids,
                attention_mask=torch.ones_like(generated_ids),
                position_ids=current_pos_1d,
                use_cache=False,
            )
            
            # Get logits for last position
            logits = outputs.logits[:, -1, :]
            
            # Sample next token
            if config.temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                # Apply temperature
                logits = logits / config.temperature
                
                # Top-p sampling
                if config.top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > config.top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            next_token_id = next_token.item()
            generated_token_list.append(next_token_id)
            
            # Update grid state and compute position for this token
            if next_token_id == grid_start_id:
                # Entering a grid
                in_grid = True
                grid_row = 0
                grid_col = 0
                # grid_start token itself is text mode
                pos_2d_list.append([0, 0])
                grid_mode_list.append(0)
            elif next_token_id == grid_end_id:
                # Exiting grid
                in_grid = False
                # grid_end token is text mode
                pos_2d_list.append([0, 0])
                grid_mode_list.append(0)
            elif in_grid:
                if next_token_id == newline_id:
                    # Newline in grid: structural token, not a grid pixel
                    # Use grid_mode=0 (text mode) for newlines
                    pos_2d_list.append([0, 0])
                    grid_mode_list.append(0)
                    grid_row += 1
                    grid_col = 0
                elif next_token_id in digit_ids:
                    # Digit in grid: assign 2D position
                    pos_2d_list.append([grid_row, grid_col])
                    grid_mode_list.append(1)
                    grid_col += 1
                else:
                    # Other token in grid (shouldn't happen but handle it)
                    # Treat as text token
                    pos_2d_list.append([0, 0])
                    grid_mode_list.append(0)
            else:
                # Text token outside grid
                pos_2d_list.append([0, 0])
                grid_mode_list.append(0)
            
            # Append to sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Check for stop tokens
            if next_token_id in stop_token_ids:
                break
    
    # Decode generated tokens (excluding input)
    generated_text = tokenizer.tokenizer.decode(generated_token_list, skip_special_tokens=False)
    
    # Debug logging
    print(f"[GEN] Generated {len(generated_token_list)} tokens")
    print(f"[GEN] grid_mode counts: text={grid_mode_list.count(0)}, grid={grid_mode_list.count(1)}")
    print(f"[GEN] Token IDs (first 50): {generated_token_list[:50]}")
    print(f"[GEN] Output text (first 500 chars):\n{generated_text[:500]}")
    
    return generated_text, generated_token_list


def extract_grid_from_tokens(
    generated_ids: List[int],
    tokenizer: Arc2DTokenizer,
) -> Optional[np.ndarray]:
    """
    Extract the predicted grid from generated token IDs.
    
    Works with token IDs directly for robustness - finds tokens between
    <|grid_start|> and <|grid_end|>, parses digit tokens into grid.
    
    Args:
        generated_ids: List of generated token IDs
        tokenizer: Arc2DTokenizer (for token ID mappings)
    
    Returns:
        numpy array of shape (H, W) with values 0-9, or None if parsing fails
    """
    # structural_ids stores lists, take [0] for single-token special tokens
    grid_start_id = tokenizer.structural_ids[tokenizer.TOK_GRID_START][0]
    grid_end_id = tokenizer.structural_ids[tokenizer.TOK_GRID_END][0]
    newline_id = tokenizer.newline_id
    digit_ids = tokenizer.digit_ids  # List where index = digit value
    digit_id_set = set(digit_ids)
    
    # Find last grid_start and corresponding grid_end
    last_start = -1
    last_end = -1
    for i, tok in enumerate(generated_ids):
        if tok == grid_start_id:
            last_start = i
        elif tok == grid_end_id and last_start != -1:
            last_end = i
    
    # Debug logging
    print(f"[EXTRACT] Looking for grid_start_id={grid_start_id}, grid_end_id={grid_end_id}")
    print(f"[EXTRACT] Total tokens: {len(generated_ids)}, last_start={last_start}, last_end={last_end}")
    
    if last_start == -1 or last_end == -1 or last_end <= last_start:
        print(f"[EXTRACT] FAILED: No valid grid found (start={last_start}, end={last_end})")
        return None
    
    # Extract grid tokens (between start and end)
    grid_tokens = generated_ids[last_start + 1:last_end]
    print(f"[EXTRACT] Grid tokens ({len(grid_tokens)}): {grid_tokens[:30]}{'...' if len(grid_tokens) > 30 else ''}")
    
    # Parse into rows
    rows = []
    current_row = []
    for tok in grid_tokens:
        if tok == newline_id:
            if current_row:
                rows.append(current_row)
                current_row = []
        elif tok in digit_id_set:
            # Map token ID back to digit value
            digit_value = digit_ids.index(tok)
            current_row.append(digit_value)
        # Ignore other tokens (shouldn't be present but handle gracefully)
    
    # Don't forget last row if no trailing newline
    if current_row:
        rows.append(current_row)
    
    if not rows:
        print(f"[EXTRACT] FAILED: No rows parsed from grid tokens")
        return None
    
    # Verify all rows have same length
    row_lens = [len(r) for r in rows]
    print(f"[EXTRACT] Parsed {len(rows)} rows, lengths: {row_lens}")
    if len(set(row_lens)) > 1:
        # Use most common row length, filter out inconsistent rows
        from collections import Counter
        most_common_len = Counter(row_lens).most_common(1)[0][0]
        rows = [r for r in rows if len(r) == most_common_len]
    
    if not rows:
        return None
    
    return np.array(rows, dtype=np.int32)


def extract_grid_from_output(
    output_text: str,
    tokenizer: Arc2DTokenizer,
) -> Optional[np.ndarray]:
    """
    Extract the predicted grid from model output text.
    
    DEPRECATED: Use extract_grid_from_tokens for robustness.
    This text-based version is kept as fallback.
    
    Args:
        output_text: Generated text from the model
        tokenizer: Arc2DTokenizer (for token strings)
    
    Returns:
        numpy array of shape (H, W) with values 0-9, or None if parsing fails
    """
    grid_start = "<|grid_start|>"
    grid_end = "<|grid_end|>"
    
    # Find the last grid in the output (in case thinking contains grids)
    last_start = output_text.rfind(grid_start)
    last_end = output_text.rfind(grid_end)
    
    if last_start == -1 or last_end == -1 or last_end <= last_start:
        return None
    
    # Extract grid content
    grid_content = output_text[last_start + len(grid_start):last_end]
    
    # Clean up: remove any non-digit, non-newline characters
    lines = grid_content.strip().split('\n')
    
    grid = []
    for line in lines:
        row = []
        for char in line:
            if char.isdigit():
                row.append(int(char))
        if row:
            grid.append(row)
    
    if not grid:
        return None
    
    # Verify all rows have same length
    row_lens = [len(r) for r in grid]
    if len(set(row_lens)) > 1:
        from collections import Counter
        most_common_len = Counter(row_lens).most_common(1)[0][0]
        grid = [r for r in grid if len(r) == most_common_len]
    
    if not grid:
        return None
    
    return np.array(grid, dtype=np.int32)


def verify_prediction(
    predicted_grid: np.ndarray,
    target_grid: np.ndarray,
) -> bool:
    """
    Check if prediction exactly matches target.
    """
    if predicted_grid is None:
        return False
    if predicted_grid.shape != target_grid.shape:
        return False
    return np.array_equal(predicted_grid, target_grid)


def predict_with_verification(
    model,
    tokenizer: Arc2DTokenizer,
    puzzle: dict,
    config: InferenceConfig,
    device: torch.device,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Generate multiple candidate predictions and return the best one.
    
    Uses best-of-N sampling with majority voting - the most common grid
    among candidates is returned. This is faster than support consistency
    scoring and often more reliable.
    
    Args:
        model: The Qwen model (with adapted LoRA)
        tokenizer: Arc2DTokenizer
        puzzle: Full puzzle with 'train' (support) and 'test' (query)
        config: Inference configuration
        device: Device to run on
    
    Returns:
        (best_grid, best_output_text)
        - best_grid: numpy array or None if all candidates failed
        - best_output_text: The full generated text for best candidate
    """
    candidates = []  # List of (grid, text, token_ids)
    
    for i in range(config.num_candidates):
        print(f'Predicting candidate: {i + 1}')
        # Generate with temperature (except first one which is greedy)
        temp = 0.0 if i == 0 else config.temperature
        
        candidate_config = InferenceConfig(
            num_candidates=1,
            temperature=temp,
            top_p=config.top_p,
            use_thinking=config.use_thinking,
            max_new_tokens=config.max_new_tokens,
            thinking_prompt=config.thinking_prompt,
        )
        
        try:
            output_text, token_ids = generate_with_thinking(
                model, tokenizer, puzzle, candidate_config, device
            )
            
            # Use token-based extraction (more robust)
            predicted_grid = extract_grid_from_tokens(token_ids, tokenizer)
            
            if predicted_grid is not None:
                print(f"[CANDIDATE {i}] Grid extracted successfully, shape: {predicted_grid.shape}")
                print(f"[CANDIDATE {i}] Grid:\n{predicted_grid}")
                candidates.append((predicted_grid, output_text))
            else:
                print(f'[CANDIDATE {i}] Grid extraction FAILED')
        except Exception as e:
            print(f"Candidate {i} generation failed: {e}")
            continue
    
    if not candidates:
        # All candidates failed - try one more greedy attempt
        try:
            output_text, token_ids = generate_with_thinking(
                model, tokenizer, puzzle,
                InferenceConfig(
                    temperature=0.0,
                    use_thinking=False,
                    max_new_tokens=config.max_new_tokens,
                ),
                device
            )
            predicted_grid = extract_grid_from_tokens(token_ids, tokenizer)
            return predicted_grid, output_text
        except Exception:
            return None, ""
    
    if len(candidates) == 1:
        return candidates[0]
    
    # Majority voting: find most common grid
    grid_counts = {}  # grid_bytes -> (grid, text, count)
    for grid, text in candidates:
        key = grid.tobytes()
        if key not in grid_counts:
            grid_counts[key] = (grid, text, 0)
        # Increment count, keep first text for this grid
        grid_counts[key] = (grid_counts[key][0], grid_counts[key][1], grid_counts[key][2] + 1)
    
    # Return grid with highest count
    # Tie-break: prefer greedy (first) candidate since it's deterministic
    best_key = max(grid_counts.keys(), key=lambda k: grid_counts[k][2])
    best_grid, best_text, _ = grid_counts[best_key]
    
    return best_grid, best_text


def _set_2d_positions(model, pos_2d: torch.Tensor, grid_mode: torch.Tensor) -> None:
    """
    Store 2D position tensors on attention layers for forward pass access.
    
    This is needed because our Qwen2DAttention layers read _pos_2d and _grid_mode
    attributes during forward to apply the correct positional encoding.
    """
    # Handle FSDP wrapping
    base_model = model
    if hasattr(model, '_fsdp_wrapped_module'):
        base_model = model._fsdp_wrapped_module
    elif hasattr(model, 'module'):
        base_model = model.module
    
    # Access layers
    if hasattr(base_model, 'model') and hasattr(base_model.model, 'layers'):
        layers = base_model.model.layers
    else:
        return
    
    for layer in layers:
        attn = layer.self_attn if hasattr(layer, 'self_attn') else layer
        if hasattr(attn, '_fsdp_wrapped_module'):
            attn = attn._fsdp_wrapped_module
        attn._pos_2d = pos_2d
        attn._grid_mode = grid_mode
