import torch
import numpy as np
from transformers import AutoTokenizer

# Standard Separators for "Grid as Text"
TOK_ROW_SEP    = "\n" 
TOK_PAIR_START = "<|pair_start|>" 
TOK_INPUT_SEP  = "<|input|>" 
TOK_OUTPUT_SEP = "<|output|>" 
TOK_PAIR_END   = "<|pair_end|>"

class ArcBaselineTokenizer:
    def __init__(self, model_id="Qwen/Qwen2-7B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        # 1. Add Special Structure Tokens
        special_tokens_dict = {
            "additional_special_tokens": [
                TOK_PAIR_START, TOK_INPUT_SEP, TOK_OUTPUT_SEP, TOK_PAIR_END
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens_dict)
        
        # 2. ROBUST ID FETCHING
        # We use encode() instead of convert_tokens_to_ids() to be safe.
        # We take [0] because encode returns a list (sometimes with start tokens).
        
        def get_id(token_str):
            # add_special_tokens=False ensures we don't get BOS/EOS added automatically
            ids = self.tokenizer.encode(token_str, add_special_tokens=False)
            if not ids:
                raise ValueError(f"Tokenizer failed to encode: '{token_str}'")
            return ids[-1] # Take the last token (usually the only one)

        # Cache IDs for Digits 0-9
        self.digit_ids = [get_id(str(i)) for i in range(10)]
        
        # Cache ID for newline (Handle \n carefully)
        # Note: Some tokenizers treat "\n" as multiple tokens or blank. 
        # If this fails, we can fallback to specific unicode bytes.
        self.newline_id = get_id("\n")
        
        # Cache IDs for structure tokens
        self.sep_ids = {
            k: get_id(k)
            for k in [TOK_PAIR_START, TOK_INPUT_SEP, TOK_OUTPUT_SEP, TOK_PAIR_END]
        }
        
        self.ignore_index = -100

    def serialize_grid_1d(self, grid: list) -> list:
        """
        Converts a 2D grid into a 1D list of token IDs.
        """
        grid_np = np.array(grid)
        if grid_np.ndim != 2: return []
        
        ids = []
        rows, cols = grid_np.shape
        
        for r in range(rows):
            for c in range(cols):
                val = grid_np[r, c]
                
                # Validation: Ensure value is a valid digit 0-9
                if val is None or not (0 <= val <= 9):
                    # Fallback or Error? For now, map to 0 or skip
                    val = 0 
                
                # Append the cached ID
                ids.append(self.digit_ids[int(val)])
            
            # End of row
            if r < rows - 1:
                ids.append(self.newline_id)
                
        return ids

    def build_sample(self, task_data: dict, inference_mode: bool = False):
        input_ids = []
        labels = []
        
        # Combine Train and Test
        train_pairs = task_data.get('train', [])
        test_pairs = task_data.get('test', [])
        all_pairs = train_pairs + test_pairs
        
        for i, pair in enumerate(all_pairs):
            is_test_pair = (i >= len(train_pairs))
            
            # --- Header ---
            # Use safe dictionary lookup
            input_ids.append(self.sep_ids[TOK_PAIR_START])
            input_ids.append(self.sep_ids[TOK_INPUT_SEP])
            labels.extend([self.ignore_index, self.ignore_index])
            
            # --- Input Grid ---
            grid_ids = self.serialize_grid_1d(pair['input'])
            input_ids.extend(grid_ids)
            labels.extend([self.ignore_index] * len(grid_ids))
            
            # --- Separator ---
            input_ids.append(self.sep_ids[TOK_OUTPUT_SEP])
            labels.append(self.ignore_index)
            
            # --- Inference Stop ---
            if inference_mode and is_test_pair:
                break
                
            # --- Output Grid ---
            out_ids = self.serialize_grid_1d(pair['output'])
            input_ids.extend(out_ids)
            # LABEL THE OUTPUT
            labels.extend(out_ids)
            
            # --- Footer ---
            input_ids.append(self.sep_ids[TOK_PAIR_END])
            labels.append(self.sep_ids[TOK_PAIR_END])

        # Validate Final List
        if any(x is None for x in input_ids):
            raise ValueError("FATAL: `input_ids` contains None! Check initialization.")

        # Return standard format
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.ones(len(input_ids), dtype=torch.long)
        }