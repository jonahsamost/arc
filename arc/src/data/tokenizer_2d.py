import torch
import numpy as np
from typing import List, Dict, Optional
from transformers import AutoTokenizer

# Masking Value for Labels
IGNORE_INDEX = -100


class Arc2DTokenizer:
    """
    Tokenizer for ARC-AGI puzzles with 2D position encoding support.
    
    Design principles:
    1. Reuse Qwen's existing tokens where possible (digits, newlines, special tokens)
    2. Add minimal new tokens (<|grid_start|>, <|grid_end|>) for mode switching
    3. Encode dimensions as text to leverage Qwen's number understanding
    4. Output grid_mode mask to distinguish text (1D RoPE) vs grid (2D RoPE) tokens
    
    Output format per sample:
        input_ids:      [seq_len] - token IDs
        labels:         [seq_len] - target IDs (-100 for masked)
        attention_mask: [seq_len] - 1s
        pos_1d:         [seq_len] - sequential positions (0, 1, 2, ...)
        pos_2d:         [seq_len, 2] - (y, x) grid coordinates
        grid_mode:      [seq_len] - 0 for text tokens, 1 for grid pixels
        output_mask:    [seq_len] - 1 for output grid pixels, 0 for everything else
                                    (used for output-only loss computation)
    """
    
    # Logical token names (mapped to real IDs in __init__)
    TOK_PAIR_START = "<|pair_start|>"
    TOK_PAIR_END = "<|pair_end|>"
    TOK_INPUT = "<|input|>"
    TOK_OUTPUT = "<|output|>"
    TOK_GRID_START = "<|grid_start|>"
    TOK_GRID_END = "<|grid_end|>"
    
    # FIM tokens (Qwen has these built-in)
    TOK_FIM_PREFIX = "<|fim_prefix|>"
    TOK_FIM_MIDDLE = "<|fim_middle|>"
    TOK_FIM_SUFFIX = "<|fim_suffix|>"
    
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Thinking-2507"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model_name = model_name
        
        # --- Cache digit token IDs (reuse Qwen's understanding of 0-9) ---
        self.digit_ids = []
        for i in range(10):
            ids = self.tokenizer.encode(str(i), add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"Expected single token for digit '{i}', got {ids}")
            self.digit_ids.append(ids[0])
        
        # --- Cache newline token ID ---
        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        self.newline_id = newline_ids[-1] if newline_ids else None
        
        # --- Cache 'x' token for dimension encoding (e.g., "15x20") ---
        x_ids = self.tokenizer.encode("x", add_special_tokens=False)
        self.x_id = x_ids[0] if x_ids else None
        
        # --- Map structural tokens to existing Qwen tokens ---
        # These reuse tokens Qwen already understands semantically
        self.structural_ids = {
            self.TOK_PAIR_START: self._get_token_id("###"),      # Markdown header
            self.TOK_PAIR_END: self._get_token_id("\n\n"),       # Block separator
            self.TOK_INPUT: self._get_token_id("Input:"),        # Label
            self.TOK_OUTPUT: self._get_token_id("Output:"),      # Label
        }
        
        # --- New tokens that need to be added ---
        # These signal mode switches between text (1D) and grid (2D)
        self.new_tokens = [self.TOK_GRID_START, self.TOK_GRID_END]
        self._tokens_added = False
        
        # --- FIM tokens (Qwen has these) ---
        self.fim_ids = {
            self.TOK_FIM_PREFIX: self.tokenizer.encode(self.TOK_FIM_PREFIX, add_special_tokens=False),
            self.TOK_FIM_MIDDLE: self.tokenizer.encode(self.TOK_FIM_MIDDLE, add_special_tokens=False),
            self.TOK_FIM_SUFFIX: self.tokenizer.encode(self.TOK_FIM_SUFFIX, add_special_tokens=False),
        }
        
        # Validate FIM tokens exist
        for tok, ids in self.fim_ids.items():
            if not ids:
                print(f"Warning: FIM token {tok} not found in vocabulary")
        
        self._print_debug_info()
    
    def _get_token_id(self, text: str) -> int:
        """Encode text and return the last token ID."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Could not encode '{text}'")
        return ids[-1]
    
    def _print_debug_info(self):
        """Print token mappings for verification."""
        print(f"Arc2DTokenizer initialized with {self.model_name}")
        print(f"  Digit tokens: 0={self.digit_ids[0]}, 9={self.digit_ids[9]}")
        print(f"  Newline token: {self.newline_id}")
        print(f"  Structural tokens: {self.structural_ids}")
        print(f"  New tokens to add: {self.new_tokens}")
    
    def add_special_tokens_to_model(self, model):
        """
        Add new special tokens and resize model embeddings.
        Call this BEFORE moving model to device with device_map.
        
        Returns the tokenizer (with new tokens) and modified model.
        """
        if self._tokens_added:
            return self.tokenizer, model
        
        # Add new tokens
        num_added = self.tokenizer.add_special_tokens({
            "additional_special_tokens": self.new_tokens
        })
        print(f"Added {num_added} new special tokens")
        
        # Resize embeddings
        model.resize_token_embeddings(len(self.tokenizer))
        
        # Cache the new token IDs
        self.structural_ids[self.TOK_GRID_START] = self.tokenizer.convert_tokens_to_ids(self.TOK_GRID_START)
        self.structural_ids[self.TOK_GRID_END] = self.tokenizer.convert_tokens_to_ids(self.TOK_GRID_END)
        
        print(f"  <|grid_start|> ID: {self.structural_ids[self.TOK_GRID_START]}")
        print(f"  <|grid_end|> ID: {self.structural_ids[self.TOK_GRID_END]}")
        
        # Initialize new embeddings from semantically similar tokens
        with torch.no_grad():
            embed_weight = model.model.embed_tokens.weight
            # Initialize <|grid_start|> from "[" token
            bracket_id = self._get_token_id("[")
            embed_weight[self.structural_ids[self.TOK_GRID_START]] = embed_weight[bracket_id].clone()
            # Initialize <|grid_end|> from "]" token  
            bracket_id = self._get_token_id("]")
            embed_weight[self.structural_ids[self.TOK_GRID_END]] = embed_weight[bracket_id].clone()
        
        self._tokens_added = True
        return self.tokenizer, model
    
    def encode_dimensions(self, height: int, width: int) -> List[int]:
        """
        Encode grid dimensions as text tokens (e.g., "15x20").
        Reuses Qwen's number understanding.
        """
        dim_str = f"{height}x{width}"
        return self.tokenizer.encode(dim_str, add_special_tokens=False)
    
    def serialize_grid(
        self,
        grid: List[List[int]],
        start_row: int = 0,
        start_col: int = 0
    ) -> Dict:
        """
        Serialize a 2D grid into tokens with 2D position coordinates.
        
        Returns:
            input_ids: List[int] - token IDs for pixels + row separators
            pos_2d: List[Tuple[int, int]] - (y, x) coordinates for each token
            grid_mode: List[int] - 1 for all tokens (grid mode)
        """
        grid_np = np.array(grid)
        
        if grid_np.ndim != 2 or grid_np.size == 0:
            return {"input_ids": [], "pos_2d": [], "grid_mode": []}
        
        rows, cols = grid_np.shape
        input_ids = []
        pos_2d = []
        grid_mode = []
        
        for r in range(rows):
            for c in range(cols):
                # Color token (reuse digit tokens)
                val = int(np.clip(grid_np[r, c], 0, 9))
                input_ids.append(self.digit_ids[val])
                pos_2d.append((start_row + r, start_col + c))
                grid_mode.append(1)  # Grid mode ON
            
            # Row separator (newline) - positioned at end of row
            input_ids.append(self.newline_id)
            pos_2d.append((start_row + r, start_col + cols))  # Just past last column
            grid_mode.append(1)  # Keep in grid mode for spatial continuity
        
        return {
            "input_ids": input_ids,
            "pos_2d": pos_2d,
            "grid_mode": grid_mode,
            "grid_height": rows,
            "grid_width": cols
        }
    
    def _append_text_tokens(
        self,
        token_ids: List[int],
        input_ids: List[int],
        labels: List[int],
        pos_2d: List[tuple],
        grid_mode: List[int],
        output_mask: List[int],
        label_value: int = IGNORE_INDEX
    ):
        """Helper to append text tokens (1D mode, positions zeroed)."""
        for tid in token_ids:
            input_ids.append(tid)
            labels.append(label_value)
            pos_2d.append((0, 0))  # 2D coords ignored in text mode
            grid_mode.append(0)    # Text mode
            output_mask.append(0)  # Never output pixels
    
    def _append_grid_tokens(
        self,
        grid_data: Dict,
        input_ids: List[int],
        labels: List[int],
        pos_2d: List[tuple],
        grid_mode: List[int],
        output_mask: List[int],
        is_target: bool = False
    ):
        """Helper to append grid tokens with their 2D positions."""
        for i, tid in enumerate(grid_data["input_ids"]):
            input_ids.append(tid)
            labels.append(tid if is_target else IGNORE_INDEX)
            pos_2d.append(grid_data["pos_2d"][i])
            grid_mode.append(grid_data["grid_mode"][i])
            output_mask.append(1 if is_target else 0)  # 1 only for output grid pixels
    
    def build_sample(
        self,
        task_data: Dict[str, List[Dict]],
        inference_mode: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Build a complete sample from an ARC task.
        
        Structure per pair:
            ### Input: <HxW> <|grid_start|> [grid pixels] <|grid_end|>
            Output: <HxW> <|grid_start|> [grid pixels] <|grid_end|> \n\n
        
        Args:
            task_data: Dict with 'train' and 'test' keys, each containing
                       list of {'input': grid, 'output': grid} pairs
            inference_mode: If True, stop after test input (don't include test output)
        
        Returns:
            Dictionary with input_ids, labels, attention_mask, pos_1d, pos_2d, grid_mode, output_mask
        """
        input_ids = []
        labels = []
        pos_2d = []
        grid_mode = []
        output_mask = []  # 1 for output grid pixels, 0 otherwise
        
        train_pairs = task_data.get('train', [])
        test_pairs = task_data.get('test', [])
        all_pairs = train_pairs + test_pairs
        
        if not all_pairs:
            raise ValueError("Task data contains no pairs")
        
        for i, pair in enumerate(all_pairs):
            is_test = i >= len(train_pairs)
            
            # === PAIR START + INPUT HEADER ===
            # "### Input:"
            self._append_text_tokens(
                [self.structural_ids[self.TOK_PAIR_START], 
                 self.structural_ids[self.TOK_INPUT]],
                input_ids, labels, pos_2d, grid_mode, output_mask
            )
            
            # === INPUT DIMENSIONS ===
            # "15x20"
            in_grid = np.array(pair['input'])
            in_h, in_w = in_grid.shape
            dim_tokens = self.encode_dimensions(in_h, in_w)
            self._append_text_tokens(dim_tokens, input_ids, labels, pos_2d, grid_mode, output_mask)
            
            # === INPUT GRID START ===
            self._append_text_tokens(
                [self.structural_ids[self.TOK_GRID_START]],
                input_ids, labels, pos_2d, grid_mode, output_mask
            )
            
            # === INPUT GRID PIXELS ===
            in_grid_data = self.serialize_grid(pair['input'])
            self._append_grid_tokens(
                in_grid_data, input_ids, labels, pos_2d, grid_mode, output_mask,
                is_target=False
            )
            
            # === INPUT GRID END ===
            self._append_text_tokens(
                [self.structural_ids[self.TOK_GRID_END]],
                input_ids, labels, pos_2d, grid_mode, output_mask
            )
            
            # === OUTPUT HEADER ===
            # "Output:"
            self._append_text_tokens(
                [self.structural_ids[self.TOK_OUTPUT]],
                input_ids, labels, pos_2d, grid_mode, output_mask
            )
            
            # === INFERENCE STOP ===
            if inference_mode and is_test:
                # Stop here - model needs to predict dimensions + grid
                break
            
            # === OUTPUT DIMENSIONS (PREDICTED) ===
            out_grid = np.array(pair['output'])
            out_h, out_w = out_grid.shape
            out_dim_tokens = self.encode_dimensions(out_h, out_w)
            
            # Dimensions ARE predicted (labeled) but NOT output_mask
            # (they're metadata, not grid pixels)
            for tid in out_dim_tokens:
                input_ids.append(tid)
                labels.append(tid)  # Target!
                pos_2d.append((0, 0))
                grid_mode.append(0)
                output_mask.append(0)  # Dimensions are NOT grid pixels
            
            # === OUTPUT GRID START ===
            self._append_text_tokens(
                [self.structural_ids[self.TOK_GRID_START]],
                input_ids, labels, pos_2d, grid_mode, output_mask
            )
            
            # === OUTPUT GRID PIXELS (PREDICTED) ===
            out_grid_data = self.serialize_grid(pair['output'])
            self._append_grid_tokens(
                out_grid_data, input_ids, labels, pos_2d, grid_mode, output_mask,
                is_target=True  # These are prediction targets, output_mask=1
            )
            
            # === OUTPUT GRID END (PREDICTED) ===
            grid_end_id = self.structural_ids[self.TOK_GRID_END]
            input_ids.append(grid_end_id)
            labels.append(grid_end_id)  # Target!
            pos_2d.append((0, 0))
            grid_mode.append(0)
            output_mask.append(0)  # Structural token, not grid pixel
            
            # === PAIR END ===
            pair_end_id = self.structural_ids[self.TOK_PAIR_END]
            input_ids.append(pair_end_id)
            labels.append(pair_end_id)  # Target!
            pos_2d.append((0, 0))
            grid_mode.append(0)
            output_mask.append(0)  # Structural token, not grid pixel
        
        # === FINAL TENSOR ASSEMBLY ===
        seq_len = len(input_ids)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
            "pos_1d": torch.arange(seq_len, dtype=torch.long),
            "pos_2d": torch.tensor(pos_2d, dtype=torch.long),  # [seq_len, 2]
            "grid_mode": torch.tensor(grid_mode, dtype=torch.long),  # [seq_len]
            "output_mask": torch.tensor(output_mask, dtype=torch.long),  # [seq_len]
        }
    
    def build_fim_sample(
        self,
        task_data: Dict[str, List[Dict]],
        max_patch_h: int = 5,
        max_patch_w: int = 5,
        num_patches: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Build a Fill-In-Middle sample with random rectangular patches.
        
        Randomly selects patches from input OR output grids and asks the model
        to predict the masked pixels given surrounding context.
        
        Format (for causal attention compatibility):
        <|fim_prefix|> [full context with patch pixels excluded]
        <|fim_middle|> [patch pixels to predict]
        
        Args:
            task_data: ARC task data
            max_patch_h: Maximum patch height (1 to max_patch_h)
            max_patch_w: Maximum patch width (1 to max_patch_w)
            num_patches: Number of patches to mask per sample
        
        Returns:
            Dictionary with FIM-formatted tensors
        """
        import random
        
        input_ids = []
        labels = []
        pos_2d = []
        grid_mode = []
        output_mask = []  # Track which tokens are FIM targets (masked patch pixels)
        
        train_pairs = task_data.get('train', [])
        test_pairs = task_data.get('test', [])
        all_pairs = train_pairs + test_pairs
        
        if not all_pairs:
            raise ValueError("Task data contains no pairs")
        
        # Get FIM token IDs
        fim_prefix_ids = self.fim_ids[self.TOK_FIM_PREFIX]
        fim_middle_ids = self.fim_ids[self.TOK_FIM_MIDDLE]
        
        # Randomly select which pair and which grid (input or output) to mask
        pair_idx = random.randint(0, len(all_pairs) - 1)
        mask_input = random.random() < 0.5  # 50% chance to mask input vs output
        
        # === FIM PREFIX: all context ===
        for tid in fim_prefix_ids:
            input_ids.append(tid)
            labels.append(IGNORE_INDEX)
            pos_2d.append((0, 0))
            grid_mode.append(0)
            output_mask.append(0)
        
        # Track which pixels to mask (will be shown in middle section)
        masked_pixels = []  # List of (row, col, value, grid_row_offset)
        
        for i, pair in enumerate(all_pairs):
            in_grid = np.array(pair['input'])
            out_grid = np.array(pair['output'])
            in_h, in_w = in_grid.shape
            out_h, out_w = out_grid.shape
            
            # Determine if this pair has the masked region
            is_masked_pair = (i == pair_idx)
            
            # Select patch location if this is the masked pair
            patch_mask_in = None
            patch_mask_out = None
            
            if is_masked_pair:
                if mask_input:
                    # Mask a patch in the input grid
                    patch_h = random.randint(1, min(max_patch_h, in_h))
                    patch_w = random.randint(1, min(max_patch_w, in_w))
                    patch_r = random.randint(0, in_h - patch_h)
                    patch_c = random.randint(0, in_w - patch_w)
                    patch_mask_in = (patch_r, patch_c, patch_h, patch_w)
                else:
                    # Mask a patch in the output grid
                    patch_h = random.randint(1, min(max_patch_h, out_h))
                    patch_w = random.randint(1, min(max_patch_w, out_w))
                    patch_r = random.randint(0, out_h - patch_h)
                    patch_c = random.randint(0, out_w - patch_w)
                    patch_mask_out = (patch_r, patch_c, patch_h, patch_w)
            
            # === PAIR HEADER ===
            self._append_text_tokens(
                [self.structural_ids[self.TOK_PAIR_START], 
                 self.structural_ids[self.TOK_INPUT]],
                input_ids, labels, pos_2d, grid_mode, output_mask
            )
            
            # === INPUT GRID ===
            dim_tokens = self.encode_dimensions(in_h, in_w)
            self._append_text_tokens(dim_tokens, input_ids, labels, pos_2d, grid_mode, output_mask)
            self._append_text_tokens(
                [self.structural_ids[self.TOK_GRID_START]],
                input_ids, labels, pos_2d, grid_mode, output_mask
            )
            
            # Serialize input grid, excluding masked patch
            self._serialize_grid_with_mask(
                in_grid, patch_mask_in, masked_pixels,
                input_ids, labels, pos_2d, grid_mode, output_mask,
                is_target=False
            )
            
            self._append_text_tokens(
                [self.structural_ids[self.TOK_GRID_END]],
                input_ids, labels, pos_2d, grid_mode, output_mask
            )
            
            # === OUTPUT HEADER ===
            self._append_text_tokens(
                [self.structural_ids[self.TOK_OUTPUT]],
                input_ids, labels, pos_2d, grid_mode, output_mask
            )
            
            out_dim_tokens = self.encode_dimensions(out_h, out_w)
            self._append_text_tokens(out_dim_tokens, input_ids, labels, pos_2d, grid_mode, output_mask)
            self._append_text_tokens(
                [self.structural_ids[self.TOK_GRID_START]],
                input_ids, labels, pos_2d, grid_mode, output_mask
            )
            
            # Serialize output grid, excluding masked patch
            self._serialize_grid_with_mask(
                out_grid, patch_mask_out, masked_pixels,
                input_ids, labels, pos_2d, grid_mode, output_mask,
                is_target=False
            )
            
            self._append_text_tokens(
                [self.structural_ids[self.TOK_GRID_END]],
                input_ids, labels, pos_2d, grid_mode, output_mask
            )
            
            # === PAIR END ===
            self._append_text_tokens(
                [self.structural_ids[self.TOK_PAIR_END]],
                input_ids, labels, pos_2d, grid_mode, output_mask
            )
        
        # === FIM MIDDLE: masked patch pixels to predict ===
        for tid in fim_middle_ids:
            input_ids.append(tid)
            labels.append(IGNORE_INDEX)
            pos_2d.append((0, 0))
            grid_mode.append(0)
            output_mask.append(0)
        
        # Add the masked pixels as prediction targets
        for row, col, value, _ in masked_pixels:
            token_id = self.digit_ids[int(np.clip(value, 0, 9))]
            input_ids.append(token_id)
            labels.append(token_id)  # PREDICTION TARGET
            pos_2d.append((row, col))  # Preserve 2D position!
            grid_mode.append(1)
            output_mask.append(1)  # These ARE the output targets for FIM
        
        # End marker
        grid_end_id = self.structural_ids[self.TOK_GRID_END]
        input_ids.append(grid_end_id)
        labels.append(grid_end_id)
        pos_2d.append((0, 0))
        grid_mode.append(0)
        output_mask.append(0)  # Structural token
        
        seq_len = len(input_ids)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
            "pos_1d": torch.arange(seq_len, dtype=torch.long),
            "pos_2d": torch.tensor(pos_2d, dtype=torch.long),
            "grid_mode": torch.tensor(grid_mode, dtype=torch.long),
            "output_mask": torch.tensor(output_mask, dtype=torch.long),
            "is_fim": True,
        }
    
    def _serialize_grid_with_mask(
        self,
        grid: np.ndarray,
        patch_mask: Optional[tuple],  # (row, col, height, width) or None
        masked_pixels: List,  # Accumulator for masked pixel info
        input_ids: List,
        labels: List,
        pos_2d: List,
        grid_mode: List,
        output_mask: List,
        is_target: bool = False,
    ):
        """
        Serialize a grid, skipping pixels in the masked patch region.
        Masked pixels are added to masked_pixels list for later prediction.
        """
        rows, cols = grid.shape
        
        for r in range(rows):
            for c in range(cols):
                val = int(np.clip(grid[r, c], 0, 9))
                
                # Check if this pixel is in the masked region
                in_mask = False
                if patch_mask is not None:
                    pr, pc, ph, pw = patch_mask
                    if pr <= r < pr + ph and pc <= c < pc + pw:
                        in_mask = True
                        # Save for prediction in middle section
                        masked_pixels.append((r, c, val, 0))
                
                if not in_mask:
                    # Add to prefix (visible context)
                    token_id = self.digit_ids[val]
                    input_ids.append(token_id)
                    labels.append(token_id if is_target else IGNORE_INDEX)
                    pos_2d.append((r, c))
                    grid_mode.append(1)
                    output_mask.append(0)  # Context pixels, not prediction targets
            
            # Row separator (newline)
            input_ids.append(self.newline_id)
            labels.append(IGNORE_INDEX)
            pos_2d.append((r, cols))
            grid_mode.append(1)
            output_mask.append(0)
    
    def decode_grid(self, token_ids: List[int]) -> Optional[np.ndarray]:
        """
        Decode a sequence of digit tokens back into a grid.
        Assumes newlines separate rows.
        
        Returns None if decoding fails.
        """
        try:
            # Create reverse mapping
            id_to_digit = {tid: i for i, tid in enumerate(self.digit_ids)}
            
            rows = []
            current_row = []
            
            for tid in token_ids:
                if tid == self.newline_id:
                    if current_row:
                        rows.append(current_row)
                        current_row = []
                elif tid in id_to_digit:
                    current_row.append(id_to_digit[tid])
                elif tid == self.structural_ids.get(self.TOK_GRID_END):
                    # End of grid
                    if current_row:
                        rows.append(current_row)
                    break
                # Skip other structural tokens
            
            if not rows:
                return None
            
            # Validate rectangular
            row_lens = [len(r) for r in rows]
            if len(set(row_lens)) > 1:
                return None  # Jagged grid
            
            return np.array(rows)
        
        except Exception:
            return None
    
    def get_output_token_ids(self) -> Dict[str, int]:
        """Return token IDs needed for generation stopping/parsing."""
        return {
            "grid_end": self.structural_ids.get(self.TOK_GRID_END),
            "pair_end": self.structural_ids.get(self.TOK_PAIR_END),
            "newline": self.newline_id,
            "digits": self.digit_ids.copy(),
        }
