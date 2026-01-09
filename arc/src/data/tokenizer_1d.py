import torch
import numpy as np
from transformers import AutoTokenizer

# Standard Abstract Separators
# We keep these variable names for code consistency, 
# but they will map to existing Qwen tokens internally.
TOK_PAIR_START = "<|pair_start|>"
TOK_INPUT_SEP  = "<|input|>"
TOK_OUTPUT_SEP = "<|output|>"
TOK_PAIR_END   = "<|pair_end|>"

class ArcBaselineTokenizer:
    def __init__(self, model_name="Qwen/Qwen3-4B-Thinking-2507"):
        # Load the standard tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # --- CRITICAL CHANGE: REUSE EXISTING TOKENS ---
        # Do NOT add special tokens. Do NOT resize embeddings.
        # This prevents the ValueError with device_map="auto".
        
        def get_id(token_str):
            # Encode string and take the last token ID.
            # add_special_tokens=False ensures no BOS/EOS is added.
            ids = self.tokenizer.encode(token_str, add_special_tokens=False)
            if not ids:
                # Fallback for weird edge cases, though unlikely for basic symbols
                print(f"Warning: Could not encode '{token_str}', falling back to newline.")
                return self.tokenizer.encode("\n", add_special_tokens=False)[-1]
            return ids[-1]

        # 1. Map Structural Constants to Existing Qwen Tokens
        # We choose tokens that Qwen semantically understands as "separators" or "structure".
        self.sep_ids = {
            # "###" is a common header in Markdown/Code
            TOK_PAIR_START: get_id("###"),
            
            # "\n" naturally separates headers from content
            TOK_INPUT_SEP:  get_id("\n"),
            
            # "=>" or "=" implies a transformation or result
            TOK_OUTPUT_SEP: get_id("=>"),
            
            # "\n" or ";" marks the end of a block
            TOK_PAIR_END:   get_id("\n\n") 
        }

        # 2. Cache IDs for Digits 0-9
        self.digit_ids = [get_id(str(i)) for i in range(10)]
        
        # 3. Cache ID for Row Separator (Newline)
        self.newline_id = get_id("\n")
        
        self.ignore_index = -100
        
        # Debug Print to verify IDs
        print(f"Tokenizer Initialized (Reuse Mode):")
        print(f"  '0' ID: {self.digit_ids[0]}")
        print(f"  Start (###) ID: {self.sep_ids[TOK_PAIR_START]}")
        print(f"  Output (=>) ID: {self.sep_ids[TOK_OUTPUT_SEP]}")

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
                    val = 0 
                
                # Append the cached ID
                ids.append(self.digit_ids[int(val)])
            
            # End of row (Newline)
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
            
            # --- Header: "### \n" ---
            input_ids.append(self.sep_ids[TOK_PAIR_START])
            input_ids.append(self.sep_ids[TOK_INPUT_SEP])
            labels.extend([self.ignore_index, self.ignore_index])
            
            # --- Input Grid ---
            grid_ids = self.serialize_grid_1d(pair['input'])
            input_ids.extend(grid_ids)
            labels.extend([self.ignore_index] * len(grid_ids))
            
            # --- Separator: "=>" ---
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
            
            # --- Footer: "\n\n" ---
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