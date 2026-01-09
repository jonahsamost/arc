import torch
import numpy as np
import random
from typing import List, Dict, Optional, Tuple
from transformers import AutoTokenizer

# Masking Value for Labels
IGNORE_INDEX = -100


class Arc2DTokenizer:
    """
    Tokenizer for ARC-AGI puzzles with 2D position encoding support.
    
    Design principles:
    1. Reuse Qwen's existing tokens where possible (digits, newlines, special tokens)
    2. Add minimal new tokens (<|grid_start|>, <|grid_end|>, <|answer|>, FIM tokens) for structure
    3. Encode dimensions as text to leverage Qwen's number understanding
    4. Output grid_mode mask to distinguish text (1D RoPE) vs grid (2D RoPE) tokens
    
    Training format (NTP - Next Token Prediction):
        For each train pair:
            ### Input: <HxW> <|grid_start|> [grid pixels] <|grid_end|>
            Output: <HxW> <|grid_start|> [grid pixels] <|grid_end|> \n\n
        For test pair:
            ### Input: <HxW> <|grid_start|> [grid pixels] <|grid_end|>
            Output: <HxW> <|grid_start|> [grid pixels] <|grid_end|> <|answer|>
    
    Training format (FIM - Fill In the Middle with patch masking):
        Same structure as NTP but with rectangular patches masked out:
            <|fim_prefix|> [puzzle with <|fim_hole|> markers] <|fim_suffix|> <|fim_middle|> [masked content] <|answer|>
        
        Patches are MxN rectangles (e.g., 2x2, 2x3, 3x3) randomly placed in grids.
        ~20% of total grid cells are masked across both input and output grids.
    
    The model learns to predict:
        - NTP: Output dimensions, grid pixels, structural tokens
        - FIM: The masked patch content in order of appearance
    
    Output format per sample:
        input_ids:      [seq_len] - token IDs
        labels:         [seq_len] - target IDs (-100 for masked)
        attention_mask: [seq_len] - 1s
        pos_1d:         [seq_len] - sequential positions (0, 1, 2, ...)
        pos_2d:         [seq_len, 2] - (y, x) grid coordinates
        grid_mode:      [seq_len] - 0 for text tokens, 1 for grid pixels
        output_mask:    [seq_len] - 1 for output grid pixels, 0 for everything else
    """
    
    # Logical token names (mapped to real IDs in __init__)
    TOK_PAIR_START = "<|pair_start|>"
    TOK_PAIR_END = "<|pair_end|>"
    TOK_INPUT = "<|input|>"
    TOK_OUTPUT = "<|output|>"
    TOK_GRID_START = "<|grid_start|>"
    TOK_GRID_END = "<|grid_end|>"
    TOK_ANSWER = "<|answer|>"  # Signals end of final answer
    
    # FIM tokens
    TOK_FIM_PREFIX = "<|fim_prefix|>"
    TOK_FIM_SUFFIX = "<|fim_suffix|>"
    TOK_FIM_MIDDLE = "<|fim_middle|>"
    TOK_FIM_HOLE = "<|fim_hole|>"  # Placeholder in prefix where content was removed
    
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
        
        # --- Map structural tokens to existing Qwen tokens ---
        # These reuse tokens Qwen already understands semantically
        self.structural_ids = {
            self.TOK_PAIR_START: self._get_token_id("###"),      # Markdown header
            self.TOK_PAIR_END: self._get_token_id("\n\n"),       # Block separator
            self.TOK_INPUT: self._get_token_id("Input:"),        # Label
            self.TOK_OUTPUT: self._get_token_id("Output:"),      # Label
        }
        
        # --- New tokens that need to be added ---
        # <|grid_start|> and <|grid_end|> signal grid mode switches
        # <|answer|> signals the end of the final answer (EOS for generation)
        # FIM tokens for fill-in-the-middle training
        self.new_tokens = [
            self.TOK_GRID_START, 
            self.TOK_GRID_END, 
            self.TOK_ANSWER,
            self.TOK_FIM_PREFIX,
            self.TOK_FIM_SUFFIX,
            self.TOK_FIM_MIDDLE,
            self.TOK_FIM_HOLE,
        ]
        self._tokens_added = False
        
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
        self.structural_ids[self.TOK_ANSWER] = self.tokenizer.convert_tokens_to_ids(self.TOK_ANSWER)
        self.structural_ids[self.TOK_FIM_PREFIX] = self.tokenizer.convert_tokens_to_ids(self.TOK_FIM_PREFIX)
        self.structural_ids[self.TOK_FIM_SUFFIX] = self.tokenizer.convert_tokens_to_ids(self.TOK_FIM_SUFFIX)
        self.structural_ids[self.TOK_FIM_MIDDLE] = self.tokenizer.convert_tokens_to_ids(self.TOK_FIM_MIDDLE)
        self.structural_ids[self.TOK_FIM_HOLE] = self.tokenizer.convert_tokens_to_ids(self.TOK_FIM_HOLE)
        
        print(f"  <|grid_start|> ID: {self.structural_ids[self.TOK_GRID_START]}")
        print(f"  <|grid_end|> ID: {self.structural_ids[self.TOK_GRID_END]}")
        print(f"  <|answer|> ID: {self.structural_ids[self.TOK_ANSWER]}")
        print(f"  FIM tokens: prefix={self.structural_ids[self.TOK_FIM_PREFIX]}, "
              f"suffix={self.structural_ids[self.TOK_FIM_SUFFIX]}, "
              f"middle={self.structural_ids[self.TOK_FIM_MIDDLE]}, "
              f"hole={self.structural_ids[self.TOK_FIM_HOLE]}")
        
        # Initialize new embeddings from semantically similar tokens
        with torch.no_grad():
            embed_weight = model.model.embed_tokens.weight
            # Initialize <|grid_start|> from "[" token
            bracket_id = self._get_token_id("[")
            embed_weight[self.structural_ids[self.TOK_GRID_START]] = embed_weight[bracket_id].clone()
            # Initialize <|grid_end|> from "]" token  
            bracket_id = self._get_token_id("]")
            embed_weight[self.structural_ids[self.TOK_GRID_END]] = embed_weight[bracket_id].clone()
            # Initialize <|answer|> from EOS token
            eos_id = self.tokenizer.eos_token_id
            if eos_id is not None:
                embed_weight[self.structural_ids[self.TOK_ANSWER]] = embed_weight[eos_id].clone()
            # Initialize FIM tokens from semantically similar tokens
            # <|fim_prefix|> from "<" (start marker)
            start_id = self._get_token_id("<")
            embed_weight[self.structural_ids[self.TOK_FIM_PREFIX]] = embed_weight[start_id].clone()
            # <|fim_suffix|> from ">" (end marker)
            end_id = self._get_token_id(">")
            embed_weight[self.structural_ids[self.TOK_FIM_SUFFIX]] = embed_weight[end_id].clone()
            # <|fim_middle|> from "=" (middle/equals)
            mid_id = self._get_token_id("=")
            embed_weight[self.structural_ids[self.TOK_FIM_MIDDLE]] = embed_weight[mid_id].clone()
            # <|fim_hole|> from "_" (placeholder/blank)
            hole_id = self._get_token_id("_")
            embed_weight[self.structural_ids[self.TOK_FIM_HOLE]] = embed_weight[hole_id].clone()
        
        self._tokens_added = True
        return self.tokenizer, model
    
    def encode_dimensions(self, height: int, width: int) -> List[int]:
        """
        Encode grid dimensions as text tokens (e.g., "5x5").
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
        is_target: bool = False
    ):
        """Helper to append text tokens (1D mode, positions zeroed)."""
        for tid in token_ids:
            input_ids.append(tid)
            labels.append(tid if is_target else IGNORE_INDEX)
            pos_2d.append((0, 0))  # 2D coords ignored in text mode
            grid_mode.append(0)    # Text mode
            output_mask.append(0)  # Text tokens are not output pixels
    
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
        
        Training format:
            For each train pair:
                ### Input: <HxW> <|grid_start|> [pixels] <|grid_end|>
                Output: <HxW> <|grid_start|> [pixels] <|grid_end|> \\n\\n
            For test pair:
                ### Input: <HxW> <|grid_start|> [pixels] <|grid_end|>
                Output: <HxW> <|grid_start|> [pixels] <|grid_end|> <|answer|>
        
        Model learns to predict (labels != -100):
            - Output dimensions
            - <|grid_start|> before output grid
            - Output grid pixels
            - <|grid_end|> after output grid
            - <|answer|> for final test pair (or \\n\\n for train pairs)
        
        Args:
            task_data: Dict with 'train' and 'test' keys
            inference_mode: If True, stop after test input's "Output:" token
        
        Returns:
            Dictionary with input_ids, labels, attention_mask, pos_1d, pos_2d, grid_mode, output_mask
        """
        input_ids = []
        labels = []
        pos_2d = []
        grid_mode = []
        output_mask = []
        
        train_pairs = task_data.get('train', [])
        test_pairs = task_data.get('test', [])
        all_pairs = train_pairs + test_pairs
        
        if not all_pairs:
            raise ValueError("Task data contains no pairs")
        
        num_train = len(train_pairs)
        num_total = len(all_pairs)
        
        for i, pair in enumerate(all_pairs):
            is_test = i >= num_train
            is_last = i == num_total - 1
            
            # === PAIR START + INPUT HEADER ===
            # "### Input:"
            self._append_text_tokens(
                [self.structural_ids[self.TOK_PAIR_START], 
                 self.structural_ids[self.TOK_INPUT]],
                input_ids, labels, pos_2d, grid_mode, output_mask,
                is_target=False
            )
            
            # === INPUT DIMENSIONS ===
            in_grid = np.array(pair['input'])
            in_h, in_w = in_grid.shape
            dim_tokens = self.encode_dimensions(in_h, in_w)
            self._append_text_tokens(
                dim_tokens, input_ids, labels, pos_2d, grid_mode, output_mask,
                is_target=False
            )
            
            # === INPUT GRID START ===
            self._append_text_tokens(
                [self.structural_ids[self.TOK_GRID_START]],
                input_ids, labels, pos_2d, grid_mode, output_mask,
                is_target=False
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
                input_ids, labels, pos_2d, grid_mode, output_mask,
                is_target=False
            )
            
            # === OUTPUT HEADER ===
            # "Output:"
            self._append_text_tokens(
                [self.structural_ids[self.TOK_OUTPUT]],
                input_ids, labels, pos_2d, grid_mode, output_mask,
                is_target=False
            )
            
            # === INFERENCE STOP ===
            if inference_mode and is_test:
                # Stop here - model needs to predict everything after "Output:"
                break
            
            # === OUTPUT DIMENSIONS (PREDICTED) ===
            out_grid = np.array(pair['output'])
            out_h, out_w = out_grid.shape
            out_dim_tokens = self.encode_dimensions(out_h, out_w)
            self._append_text_tokens(
                out_dim_tokens, input_ids, labels, pos_2d, grid_mode, output_mask,
                is_target=True  # Model predicts dimensions
            )
            
            # === OUTPUT GRID START (PREDICTED) ===
            self._append_text_tokens(
                [self.structural_ids[self.TOK_GRID_START]],
                input_ids, labels, pos_2d, grid_mode, output_mask,
                is_target=True  # Model predicts <|grid_start|>
            )
            
            # === OUTPUT GRID PIXELS (PREDICTED) ===
            out_grid_data = self.serialize_grid(pair['output'])
            self._append_grid_tokens(
                out_grid_data, input_ids, labels, pos_2d, grid_mode, output_mask,
                is_target=True  # Model predicts grid pixels
            )
            
            # === OUTPUT GRID END (PREDICTED) ===
            self._append_text_tokens(
                [self.structural_ids[self.TOK_GRID_END]],
                input_ids, labels, pos_2d, grid_mode, output_mask,
                is_target=True  # Model predicts <|grid_end|>
            )
            
            # === PAIR END / ANSWER ===
            if is_last:
                # Final test pair: use <|answer|> as EOS signal
                self._append_text_tokens(
                    [self.structural_ids[self.TOK_ANSWER]],
                    input_ids, labels, pos_2d, grid_mode, output_mask,
                    is_target=True  # Model predicts <|answer|>
                )
            else:
                # Train pairs: use \n\n as separator
                self._append_text_tokens(
                    [self.structural_ids[self.TOK_PAIR_END]],
                    input_ids, labels, pos_2d, grid_mode, output_mask,
                    is_target=True  # Model predicts pair separator
                )
        
        # === FINAL TENSOR ASSEMBLY ===
        seq_len = len(input_ids)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
            "pos_1d": torch.arange(seq_len, dtype=torch.long),
            "pos_2d": torch.tensor(pos_2d, dtype=torch.long),
            "grid_mode": torch.tensor(grid_mode, dtype=torch.long),
            "output_mask": torch.tensor(output_mask, dtype=torch.long),
        }
    
    # =========================================================================
    # FIM (Fill-In-the-Middle) with Patch Masking
    # =========================================================================
    
    def _generate_patch_mask(
        self,
        height: int,
        width: int,
        target_ratio: float = 0.2,
        min_patch_size: int = 2,
        max_patch_size: int = 4,
    ) -> np.ndarray:
        """
        Generate a boolean mask for a grid with random rectangular patches.
        
        Args:
            height: Grid height
            width: Grid width
            target_ratio: Target fraction of cells to mask (~20%)
            min_patch_size: Minimum patch dimension
            max_patch_size: Maximum patch dimension
        
        Returns:
            Boolean mask array of shape (height, width), True = masked
        """
        mask = np.zeros((height, width), dtype=bool)
        total_cells = height * width
        target_masked = int(total_cells * target_ratio)
        
        # Skip if grid is too small for any patches
        if height < min_patch_size or width < min_patch_size:
            # For tiny grids, just mask random individual cells
            if target_masked > 0 and total_cells > 0:
                num_to_mask = min(target_masked, total_cells)
                indices = random.sample(range(total_cells), num_to_mask)
                for idx in indices:
                    r, c = idx // width, idx % width
                    mask[r, c] = True
            return mask
        
        # Keep adding patches until we reach target
        attempts = 0
        max_attempts = 100  # Prevent infinite loops on small grids
        
        while mask.sum() < target_masked and attempts < max_attempts:
            attempts += 1
            
            # Random patch size (constrained by grid dimensions)
            # Ensure min <= max for randint
            max_h = min(max_patch_size, height)
            max_w = min(max_patch_size, width)
            patch_h = random.randint(min_patch_size, max_h)
            patch_w = random.randint(min_patch_size, max_w)
            
            # Random position (ensure patch fits)
            if height - patch_h < 0 or width - patch_w < 0:
                continue
            start_r = random.randint(0, height - patch_h)
            start_c = random.randint(0, width - patch_w)
            
            # Apply patch
            mask[start_r:start_r + patch_h, start_c:start_c + patch_w] = True
            
            # Early exit if we've masked enough
            if mask.sum() >= target_masked:
                break
        
        return mask
    
    def _select_grids_for_masking(
        self,
        task_data: Dict[str, List[Dict]],
        target_ratio: float = 0.2,
    ) -> List[Tuple[str, int, str, np.ndarray]]:
        """
        Select which grids to mask and generate masks for them.
        
        Returns list of (pair_type, pair_idx, grid_type, mask_array) tuples.
        pair_type: 'train' or 'test'
        pair_idx: index within train/test list
        grid_type: 'input' or 'output'
        mask_array: boolean mask for that grid
        """
        # Collect all grids with their sizes
        all_grids = []
        for pair_idx, pair in enumerate(task_data.get('train', [])):
            in_grid = np.array(pair['input'])
            out_grid = np.array(pair['output'])
            all_grids.append(('train', pair_idx, 'input', in_grid))
            all_grids.append(('train', pair_idx, 'output', out_grid))
        
        for pair_idx, pair in enumerate(task_data.get('test', [])):
            in_grid = np.array(pair['input'])
            out_grid = np.array(pair['output'])
            all_grids.append(('test', pair_idx, 'input', in_grid))
            all_grids.append(('test', pair_idx, 'output', out_grid))
        
        if not all_grids:
            return []
        
        # Calculate total cells
        total_cells = sum(g[3].size for g in all_grids)
        target_masked_cells = int(total_cells * target_ratio)
        
        # Randomly select grids to mask until we hit target
        masks = []
        masked_cells = 0
        
        # Shuffle grid order for randomness
        grid_indices = list(range(len(all_grids)))
        random.shuffle(grid_indices)
        
        for idx in grid_indices:
            if masked_cells >= target_masked_cells:
                break
            
            pair_type, pair_idx, grid_type, grid = all_grids[idx]
            h, w = grid.shape
            
            # How many cells should we try to mask from this grid?
            remaining = target_masked_cells - masked_cells
            grid_target_ratio = min(0.5, remaining / grid.size)  # Cap at 50% per grid
            
            if grid_target_ratio < 0.1:
                continue  # Skip if we'd mask too little
            
            mask = self._generate_patch_mask(h, w, target_ratio=grid_target_ratio)
            if mask.sum() > 0:
                masks.append((pair_type, pair_idx, grid_type, mask))
                masked_cells += mask.sum()
        
        return masks
    
    def _serialize_grid_with_mask(
        self,
        grid: List[List[int]],
        mask: Optional[np.ndarray],
        start_row: int = 0,
        start_col: int = 0
    ) -> Tuple[Dict, List[int]]:
        """
        Serialize a grid, replacing masked cells with <|fim_hole|> tokens.
        
        Returns:
            (grid_data_dict, masked_values_list)
            - grid_data_dict: same format as serialize_grid but with holes
            - masked_values_list: the actual values that were masked (in row-major order)
        """
        grid_np = np.array(grid)
        
        if grid_np.ndim != 2 or grid_np.size == 0:
            return {"input_ids": [], "pos_2d": [], "grid_mode": []}, []
        
        rows, cols = grid_np.shape
        input_ids = []
        pos_2d = []
        grid_mode = []
        masked_values = []
        
        hole_id = self.structural_ids.get(self.TOK_FIM_HOLE)
        
        for r in range(rows):
            for c in range(cols):
                val = int(np.clip(grid_np[r, c], 0, 9))
                
                # Check if this cell is masked
                is_masked = mask is not None and mask[r, c]
                
                if is_masked:
                    # Replace with hole token
                    input_ids.append(hole_id)
                    masked_values.append(self.digit_ids[val])  # Store actual value
                else:
                    # Normal digit token
                    input_ids.append(self.digit_ids[val])
                
                pos_2d.append((start_row + r, start_col + c))
                grid_mode.append(1)  # Grid mode ON
            
            # Row separator (newline)
            input_ids.append(self.newline_id)
            pos_2d.append((start_row + r, start_col + cols))
            grid_mode.append(1)
        
        return {
            "input_ids": input_ids,
            "pos_2d": pos_2d,
            "grid_mode": grid_mode,
            "grid_height": rows,
            "grid_width": cols
        }, masked_values
    
    def build_fim_sample(
        self,
        task_data: Dict[str, List[Dict]],
        target_mask_ratio: float = 0.2,
    ) -> Dict[str, torch.Tensor]:
        """
        Build a FIM (Fill-In-the-Middle) sample with patch masking.
        
        Format:
            <|fim_prefix|> [puzzle with <|fim_hole|> markers] <|fim_suffix|> <|fim_middle|> [masked content] <|answer|>
        
        The model sees the full puzzle structure but with rectangular patches masked out.
        It must predict the masked content in order of appearance.
        
        ~20% of total grid cells are masked across both input and output grids.
        
        Args:
            task_data: Dict with 'train' and 'test' keys
            target_mask_ratio: Target fraction of cells to mask (~0.2 = 20%)
        
        Returns:
            Dictionary with input_ids, labels, attention_mask, pos_1d, pos_2d, grid_mode, output_mask
        """
        # Generate masks for grids
        grid_masks = self._select_grids_for_masking(task_data, target_mask_ratio)
        
        # Create lookup dict for masks: (pair_type, pair_idx, grid_type) -> mask
        mask_lookup = {(m[0], m[1], m[2]): m[3] for m in grid_masks}
        
        # Build the prefix (puzzle with holes)
        prefix_ids = []
        prefix_pos_2d = []
        prefix_grid_mode = []
        all_masked_values = []  # Collect masked values in order
        
        train_pairs = task_data.get('train', [])
        test_pairs = task_data.get('test', [])
        all_pairs = train_pairs + test_pairs
        
        if not all_pairs:
            raise ValueError("Task data contains no pairs")
        
        num_train = len(train_pairs)
        
        for i, pair in enumerate(all_pairs):
            is_test = i >= num_train
            pair_type = 'test' if is_test else 'train'
            pair_idx = i - num_train if is_test else i
            
            # === PAIR START + INPUT HEADER ===
            prefix_ids.extend([
                self.structural_ids[self.TOK_PAIR_START],
                self.structural_ids[self.TOK_INPUT]
            ])
            prefix_pos_2d.extend([(0, 0), (0, 0)])
            prefix_grid_mode.extend([0, 0])
            
            # === INPUT DIMENSIONS ===
            in_grid = np.array(pair['input'])
            in_h, in_w = in_grid.shape
            dim_tokens = self.encode_dimensions(in_h, in_w)
            prefix_ids.extend(dim_tokens)
            prefix_pos_2d.extend([(0, 0)] * len(dim_tokens))
            prefix_grid_mode.extend([0] * len(dim_tokens))
            
            # === INPUT GRID START ===
            prefix_ids.append(self.structural_ids[self.TOK_GRID_START])
            prefix_pos_2d.append((0, 0))
            prefix_grid_mode.append(0)
            
            # === INPUT GRID (possibly with holes) ===
            in_mask = mask_lookup.get((pair_type, pair_idx, 'input'))
            in_grid_data, in_masked = self._serialize_grid_with_mask(pair['input'], in_mask)
            prefix_ids.extend(in_grid_data["input_ids"])
            prefix_pos_2d.extend(in_grid_data["pos_2d"])
            prefix_grid_mode.extend(in_grid_data["grid_mode"])
            all_masked_values.extend(in_masked)
            
            # === INPUT GRID END ===
            prefix_ids.append(self.structural_ids[self.TOK_GRID_END])
            prefix_pos_2d.append((0, 0))
            prefix_grid_mode.append(0)
            
            # === OUTPUT HEADER ===
            prefix_ids.append(self.structural_ids[self.TOK_OUTPUT])
            prefix_pos_2d.append((0, 0))
            prefix_grid_mode.append(0)
            
            # === OUTPUT DIMENSIONS ===
            out_grid = np.array(pair['output'])
            out_h, out_w = out_grid.shape
            out_dim_tokens = self.encode_dimensions(out_h, out_w)
            prefix_ids.extend(out_dim_tokens)
            prefix_pos_2d.extend([(0, 0)] * len(out_dim_tokens))
            prefix_grid_mode.extend([0] * len(out_dim_tokens))
            
            # === OUTPUT GRID START ===
            prefix_ids.append(self.structural_ids[self.TOK_GRID_START])
            prefix_pos_2d.append((0, 0))
            prefix_grid_mode.append(0)
            
            # === OUTPUT GRID (possibly with holes) ===
            out_mask = mask_lookup.get((pair_type, pair_idx, 'output'))
            out_grid_data, out_masked = self._serialize_grid_with_mask(pair['output'], out_mask)
            prefix_ids.extend(out_grid_data["input_ids"])
            prefix_pos_2d.extend(out_grid_data["pos_2d"])
            prefix_grid_mode.extend(out_grid_data["grid_mode"])
            all_masked_values.extend(out_masked)
            
            # === OUTPUT GRID END ===
            prefix_ids.append(self.structural_ids[self.TOK_GRID_END])
            prefix_pos_2d.append((0, 0))
            prefix_grid_mode.append(0)
            
            # === PAIR END (for non-last pairs) ===
            if i < len(all_pairs) - 1:
                prefix_ids.append(self.structural_ids[self.TOK_PAIR_END])
                prefix_pos_2d.append((0, 0))
                prefix_grid_mode.append(0)
        
        # === BUILD FINAL SEQUENCE ===
        # Format: <|fim_prefix|> [prefix] <|fim_suffix|> <|fim_middle|> [masked_values] <|answer|>
        
        input_ids = []
        labels = []
        pos_2d = []
        grid_mode = []
        output_mask = []
        
        # <|fim_prefix|> - not predicted
        input_ids.append(self.structural_ids[self.TOK_FIM_PREFIX])
        labels.append(IGNORE_INDEX)
        pos_2d.append((0, 0))
        grid_mode.append(0)
        output_mask.append(0)
        
        # Prefix content (puzzle with holes) - not predicted
        for i, tid in enumerate(prefix_ids):
            input_ids.append(tid)
            labels.append(IGNORE_INDEX)
            pos_2d.append(prefix_pos_2d[i])
            grid_mode.append(prefix_grid_mode[i])
            output_mask.append(0)
        
        # <|fim_suffix|> - not predicted
        input_ids.append(self.structural_ids[self.TOK_FIM_SUFFIX])
        labels.append(IGNORE_INDEX)
        pos_2d.append((0, 0))
        grid_mode.append(0)
        output_mask.append(0)
        
        # <|fim_middle|> - predicted (marks start of answer)
        input_ids.append(self.structural_ids[self.TOK_FIM_MIDDLE])
        labels.append(self.structural_ids[self.TOK_FIM_MIDDLE])
        pos_2d.append((0, 0))
        grid_mode.append(0)
        output_mask.append(0)
        
        # Masked values - all predicted
        for tid in all_masked_values:
            input_ids.append(tid)
            labels.append(tid)
            pos_2d.append((0, 0))  # Masked values are in 1D mode (sequential)
            grid_mode.append(0)
            output_mask.append(1)  # These are the "output" we're predicting
        
        # <|answer|> - predicted (marks end)
        input_ids.append(self.structural_ids[self.TOK_ANSWER])
        labels.append(self.structural_ids[self.TOK_ANSWER])
        pos_2d.append((0, 0))
        grid_mode.append(0)
        output_mask.append(0)
        
        # === FINAL TENSOR ASSEMBLY ===
        seq_len = len(input_ids)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
            "pos_1d": torch.arange(seq_len, dtype=torch.long),
            "pos_2d": torch.tensor(pos_2d, dtype=torch.long),
            "grid_mode": torch.tensor(grid_mode, dtype=torch.long),
            "output_mask": torch.tensor(output_mask, dtype=torch.long),
        }
    
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
                elif tid == self.structural_ids.get(self.TOK_ANSWER):
                    # End of answer
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
    
    def get_answer_token_id(self) -> Optional[int]:
        """Return the <|answer|> token ID for use as eos_token_id in generation."""
        return self.structural_ids.get(self.TOK_ANSWER)
    
    def get_output_token_ids(self) -> Dict[str, int]:
        """Return token IDs needed for generation stopping/parsing."""
        return {
            "grid_start": self.structural_ids.get(self.TOK_GRID_START),
            "grid_end": self.structural_ids.get(self.TOK_GRID_END),
            "answer": self.structural_ids.get(self.TOK_ANSWER),
            "pair_end": self.structural_ids.get(self.TOK_PAIR_END),
            "newline": self.newline_id,
            "digits": self.digit_ids.copy(),
        }
