import torch
import numpy as np
from typing import List, Dict, Tuple, Optional

# --- 1. Token Configuration ---
# Ensure these do not collide with your Qwen vocabulary.
# Qwen vocab usually goes up to ~151936. Check `tokenizer.vocab_size`.

# Color Offset: Maps grid values 0-9 to tokens 1000-1009
OFFSET_COLOR = 1000  

# Dimension Offset: Maps sizes 1-30 to tokens 2000-2030
OFFSET_DIM = 2000

# Structural Tokens
TOK_ROW_SEP    = 151644  # \n (End of Row)
TOK_PAIR_START = 151650  # <|pair_start|>
TOK_INPUT_SEP  = 151651  # <|input|>
TOK_OUTPUT_SEP = 151652  # <|output|>
TOK_PAIR_END   = 151653  # <|pair_end|>

# Masking Value for Labels
IGNORE_INDEX = -100

class Arc2DTokenizer:
    def __init__(self):
        pass

    def serialize_grid_2d(
        self, 
        grid: List[List[int]], 
        start_row_idx: int, 
        start_col_idx: int = 0
    ) -> Dict[str, List[int]]:
        """
        Flattens a 2D grid into tokens while generating explicit 2D coordinates.
        Does NOT generate dimension tokens (handled by parent function).
        """
        grid_np = np.array(grid)
        
        # Handle empty/malformed grids gracefully
        if grid_np.ndim != 2 or grid_np.size == 0:
            return {
                "input_ids": [], "pos_y": [], "pos_x": [], 
                "final_row_idx": start_row_idx
            }

        rows, cols = grid_np.shape
        input_ids = []
        pos_y = [] 
        pos_x = [] 
        
        current_r = 0 
        
        for r in range(rows):
            current_r = start_row_idx + r
            for c in range(cols):
                # 1. Color Token
                val = grid_np[r, c]
                # Clip value just in case, though usually 0-9
                val = max(0, min(9, int(val)))
                
                input_ids.append(OFFSET_COLOR + val)
                pos_y.append(current_r)
                pos_x.append(start_col_idx + c)
            
            # 2. Row Separator (New Line)
            # Added at the end of every row
            input_ids.append(TOK_ROW_SEP)
            pos_y.append(current_r)
            # Separator sits to the right of the last pixel
            pos_x.append(start_col_idx + cols) 
            
        return {
            "input_ids": input_ids,
            "pos_y": pos_y,
            "pos_x": pos_x,
            "final_row_idx": current_r
        }

    def build_sample(
        self, 
        task_data: Dict[str, List[Dict]], 
        inference_mode: bool = False,
        
    ) -> Dict[str, torch.Tensor]:
        """
        Processes a full ARC task (Train + Test pairs) into a single 2D context.
        
        Structure:
        <PAIR_START> <INPUT> <H_TOK> <W_TOK> [Input Grid] <OUTPUT> <H_TOK> <W_TOK> [Output Grid] <PAIR_END> ...
        """
        full_input_ids = []
        full_pos_y = []
        full_pos_x = []
        full_labels = []
        
        current_row = 0
        
        # Combine train and test into one sequence
        train_pairs = task_data.get('train', [])
        test_pairs = task_data.get('test', [])
        all_pairs = train_pairs + test_pairs
        
        if not all_pairs:
            raise ValueError("Task data contains no pairs.")

        for i, pair in enumerate(all_pairs):
            is_test_pair = (i >= len(train_pairs))
            
            # ---------------------------------------------------------
            # 1. HEADER: <PAIR_START> <INPUT>
            # ---------------------------------------------------------
            full_input_ids.extend([TOK_PAIR_START, TOK_INPUT_SEP])
            full_labels.extend([IGNORE_INDEX, IGNORE_INDEX])
            # Meta-row for headers
            full_pos_y.extend([current_row, current_row])
            full_pos_x.extend([0, 1])
            
            current_row += 1 

            # ---------------------------------------------------------
            # 2. INPUT DIMENSIONS + GRID
            # ---------------------------------------------------------
            h, w = np.array(pair['input']).shape
            # Map dims (1-30) to tokens. 
            dim_tokens = [OFFSET_DIM + h, OFFSET_DIM + w]
            
            full_input_ids.extend(dim_tokens)
            full_labels.extend([IGNORE_INDEX, IGNORE_INDEX])
            # Dims sit at x=0, x=1 on the first row of the grid context
            full_pos_y.extend([current_row, current_row])
            full_pos_x.extend([0, 1])
            
            # Serialize Grid (Start pixels at col 2)
            res_in = self.serialize_grid_2d(
                pair['input'], 
                start_row_idx=current_row, 
                start_col_idx=2
            )
            
            full_input_ids.extend(res_in['input_ids'])
            full_labels.extend([IGNORE_INDEX] * len(res_in['input_ids']))
            full_pos_y.extend(res_in['pos_y'])
            full_pos_x.extend(res_in['pos_x'])
            
            current_row = res_in['final_row_idx'] + 1
            
            # ---------------------------------------------------------
            # 3. SEPARATOR: <OUTPUT>
            # ---------------------------------------------------------
            full_input_ids.append(TOK_OUTPUT_SEP)
            full_labels.append(IGNORE_INDEX)
            full_pos_y.append(current_row)
            full_pos_x.append(0)
            
            current_row += 1
            
            # ---------------------------------------------------------
            # 4. INFERENCE CHECK
            # ---------------------------------------------------------
            if inference_mode and is_test_pair:
                # In inference, we stop exactly here.
                # The model sees <OUTPUT> and must start predicting dimensions, then grid.
                break
            
            # ---------------------------------------------------------
            # 5. OUTPUT DIMENSIONS + GRID
            # ---------------------------------------------------------
            h_out, w_out = np.array(pair['output']).shape
            dim_tokens_out = [OFFSET_DIM + h_out, OFFSET_DIM + w_out]
            
            full_input_ids.extend(dim_tokens_out)
            # IMPORTANT: We PREDICT the dimensions!
            full_labels.extend(dim_tokens_out)
            
            full_pos_y.extend([current_row, current_row])
            full_pos_x.extend([0, 1])
            
            res_out = self.serialize_grid_2d(
                pair['output'], 
                start_row_idx=current_row, 
                start_col_idx=2
            )
            
            full_input_ids.extend(res_out['input_ids'])
            # IMPORTANT: We PREDICT the grid tokens!
            full_labels.extend(res_out['input_ids'])
            
            full_pos_y.extend(res_out['pos_y'])
            full_pos_x.extend(res_out['pos_x'])
            
            current_row = res_out['final_row_idx'] + 1

            # ---------------------------------------------------------
            # 6. FOOTER: <PAIR_END>
            # ---------------------------------------------------------
            full_input_ids.append(TOK_PAIR_END)
            full_labels.append(TOK_PAIR_END) # Model should learn to stop
            full_pos_y.append(current_row)
            full_pos_x.append(0)
            
            # Gap before next pair
            current_row += 2 

        # ---------------------------------------------------------
        # Final Tensor Assembly
        # ---------------------------------------------------------
        
        # 1. Global 1D Position (Standard Sequence Index)
        pos_1d = torch.arange(len(full_input_ids), dtype=torch.long)
        
        # 2. Vertical RoPE Index
        pos_y = torch.tensor(full_pos_y, dtype=torch.long)
        
        # 3. Horizontal RoPE Index
        pos_x = torch.tensor(full_pos_x, dtype=torch.long)
        
        # Stack into [3, Seq_Len]
        # Row 0: Global 1D (Causal Masking)
        # Row 1: Y (Vertical RoPE)
        # Row 2: X (Horizontal RoPE)
        position_ids = torch.stack([pos_1d, pos_y, pos_x], dim=0)

        return {
            "input_ids": torch.tensor(full_input_ids, dtype=torch.long),
            "labels": torch.tensor(full_labels, dtype=torch.long),
            "position_ids": position_ids,
            "attention_mask": torch.ones(len(full_input_ids), dtype=torch.long)
        }
