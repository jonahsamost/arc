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
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"):
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
        label_value: int = IGNORE_INDEX
    ):
        """Helper to append text tokens (1D mode, positions zeroed)."""
        for tid in token_ids:
            input_ids.append(tid)
            labels.append(label_value)
            pos_2d.append((0, 0))  # 2D coords ignored in text mode
            grid_mode.append(0)    # Text mode
    
    def _append_grid_tokens(
        self,
        grid_data: Dict,
        input_ids: List[int],
        labels: List[int],
        pos_2d: List[tuple],
        grid_mode: List[int],
        is_target: bool = False
    ):
        """Helper to append grid tokens with their 2D positions."""
        for i, tid in enumerate(grid_data["input_ids"]):
            input_ids.append(tid)
            labels.append(tid if is_target else IGNORE_INDEX)
            pos_2d.append(grid_data["pos_2d"][i])
            grid_mode.append(grid_data["grid_mode"][i])
    
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
            Dictionary with input_ids, labels, attention_mask, pos_1d, pos_2d, grid_mode
        """
        input_ids = []
        labels = []
        pos_2d = []
        grid_mode = []
        
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
                input_ids, labels, pos_2d, grid_mode
            )
            
            # === INPUT DIMENSIONS ===
            # "15x20"
            in_grid = np.array(pair['input'])
            in_h, in_w = in_grid.shape
            dim_tokens = self.encode_dimensions(in_h, in_w)
            self._append_text_tokens(dim_tokens, input_ids, labels, pos_2d, grid_mode)
            
            # === INPUT GRID START ===
            self._append_text_tokens(
                [self.structural_ids[self.TOK_GRID_START]],
                input_ids, labels, pos_2d, grid_mode
            )
            
            # === INPUT GRID PIXELS ===
            in_grid_data = self.serialize_grid(pair['input'])
            self._append_grid_tokens(
                in_grid_data, input_ids, labels, pos_2d, grid_mode,
                is_target=False
            )
            
            # === INPUT GRID END ===
            self._append_text_tokens(
                [self.structural_ids[self.TOK_GRID_END]],
                input_ids, labels, pos_2d, grid_mode
            )
            
            # === OUTPUT HEADER ===
            # "Output:"
            self._append_text_tokens(
                [self.structural_ids[self.TOK_OUTPUT]],
                input_ids, labels, pos_2d, grid_mode
            )
            
            # === INFERENCE STOP ===
            if inference_mode and is_test:
                # Stop here - model needs to predict dimensions + grid
                break
            
            # === OUTPUT DIMENSIONS (PREDICTED) ===
            out_grid = np.array(pair['output'])
            out_h, out_w = out_grid.shape
            out_dim_tokens = self.encode_dimensions(out_h, out_w)
            
            # Dimensions ARE predicted (labeled)
            for tid in out_dim_tokens:
                input_ids.append(tid)
                labels.append(tid)  # Target!
                pos_2d.append((0, 0))
                grid_mode.append(0)
            
            # === OUTPUT GRID START ===
            self._append_text_tokens(
                [self.structural_ids[self.TOK_GRID_START]],
                input_ids, labels, pos_2d, grid_mode
            )
            
            # === OUTPUT GRID PIXELS (PREDICTED) ===
            out_grid_data = self.serialize_grid(pair['output'])
            self._append_grid_tokens(
                out_grid_data, input_ids, labels, pos_2d, grid_mode,
                is_target=True  # These are prediction targets
            )
            
            # === OUTPUT GRID END (PREDICTED) ===
            grid_end_id = self.structural_ids[self.TOK_GRID_END]
            input_ids.append(grid_end_id)
            labels.append(grid_end_id)  # Target!
            pos_2d.append((0, 0))
            grid_mode.append(0)
            
            # === PAIR END ===
            pair_end_id = self.structural_ids[self.TOK_PAIR_END]
            input_ids.append(pair_end_id)
            labels.append(pair_end_id)  # Target!
            pos_2d.append((0, 0))
            grid_mode.append(0)
        
        # === FINAL TENSOR ASSEMBLY ===
        seq_len = len(input_ids)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(seq_len, dtype=torch.long),
            "pos_1d": torch.arange(seq_len, dtype=torch.long),
            "pos_2d": torch.tensor(pos_2d, dtype=torch.long),  # [seq_len, 2]
            "grid_mode": torch.tensor(grid_mode, dtype=torch.long),  # [seq_len]
        }
    
    def build_fim_sample(
        self,
        task_data: Dict[str, List[Dict]],
        mask_ratio: float = 0.3,
        mask_type: str = "rectangular"
    ) -> Dict[str, torch.Tensor]:
        """
        Build a Fill-In-Middle sample for an ARC task.
        
        Masks a portion of the output grid and creates FIM format:
        <|fim_prefix|> [context + partial output] <|fim_suffix|> [rest] <|fim_middle|> [masked]
        
        Args:
            task_data: ARC task data
            mask_ratio: Approximate fraction of output to mask
            mask_type: "rectangular", "rows", "columns", or "random"
        
        Returns:
            Dictionary with FIM-formatted tensors
        """
        # For now, build standard sample - FIM logic can be added
        # This is a placeholder for the FIM training objective
        raise NotImplementedError("FIM sample building not yet implemented")
    
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
