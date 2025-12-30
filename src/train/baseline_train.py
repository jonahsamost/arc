import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.data.build_arc_dataset import (
    baseline_eval_all_arc_1d,
    baseline_eval_arc1_1d,
    baseline_eval_arc2_1d
)
from src.data.dataloader import ArcDataset
from src.data.tokenizer_1d import ArcBaselineTokenizer, TOK_OUTPUT_SEP, TOK_PAIR_END


def parse_grid_from_text(text: str) -> np.ndarray:
    """
    Robustly parses a grid string like "0 1 0\n1 1 1" into a numpy array.
    Returns None if parsing fails or grid is jagged.
    """
    try:
        rows = text.strip().split('\n')
        grid = []
        for r in rows:
            # clean up whitespace and split
            cols = [int(c) for c in r.strip().split() if c.isdigit()]
            if cols:
                grid.append(cols)
        
        if not grid:
            return None
            
        return np.array(grid)
    except Exception:
        return None


def calculate_metrics(pred_grid, gt_grid):
    """
    Returns (is_exact_match, pixel_accuracy_float)
    """
    if pred_grid is None:
        return 0, 0.0

    # 1. Check Dimensions
    if pred_grid.shape != gt_grid.shape:
        # If dimensions are wrong, we count it as 0 pixel accuracy 
        # (strictly speaking, you can't compare pixels if shapes differ)
        return 0, 0.0
    
    # 2. Check Pixels
    matches = (pred_grid == gt_grid)
    num_correct = np.sum(matches)
    total_pixels = pred_grid.size
    
    pixel_acc = num_correct / total_pixels
    exact_match = 1 if num_correct == total_pixels else 0
    
    return exact_match, pixel_acc


model_name = 'Qwen/Qwen2.5-Coder-7B-Instruct'
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer_wrapper = ArcBaselineTokenizer(model_name)
hf_tokenizer = tokenizer_wrapper.tokenizer

model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype="auto", device_map=device
)
model.eval()

data_path = baseline_eval_arc1_1d()
arc_dataset = ArcDataset(data_dir=data_path, baseline_1d=True)
data_loader = DataLoader(arc_dataset, batch_size=1, shuffle=False)

# 3. Eval Loop
total_tasks = 0
total_exact_matches = 0
total_pixel_acc = 0.0

# Get the ID for the separator where we want to cut the input
# We want the prompt to end exactly at "... <|output|>"
sep_token_id = tokenizer_wrapper.sep_ids[TOK_OUTPUT_SEP]
stop_token_id = tokenizer_wrapper.sep_ids[TOK_PAIR_END]

print(f"Starting Evaluation on tasks...")

with torch.no_grad():
    for batch in tqdm(data_loader):
        # Move to device
        data, filename = batch
        input_ids = data['input_ids'].to(device)
        labels = data['labels'].to(device)
        
        # --- A. Input Slicing ---
        # Find the LAST occurrence of <|output|> (151652) in the sequence.
        # This marks the start of the Test Prediction.
        # Note: torch.where returns indices. We take the last one.
        sep_indices = (input_ids[0] == sep_token_id).nonzero(as_tuple=True)[0]
        
        if len(sep_indices) == 0:
            print("Skipping malformed batch (no <|output|> token)")
            continue
            
        cut_idx = sep_indices[-1].item() + 1 # Include the separator in the prompt
        
        # Prompt: Everything up to <|output|>
        prompt_ids = input_ids[:, :cut_idx]
        
        # Ground Truth: Everything after <|output|> (excluding the trailing <|pair_end|>)
        # We look at the labels to get the true answer
        gt_ids = labels[0, cut_idx:]
        # Filter out -100 (ignore index) and stop tokens if present
        gt_ids = gt_ids[gt_ids != -100]
        # remove final stop token if it exists in GT
        if len(gt_ids) > 0 and gt_ids[-1] == stop_token_id:
            gt_ids = gt_ids[:-1]
            
        # --- B. Generation ---
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=1024, # ARC grids can be large (30x30 ~ 900 tokens)
            # do_sample=False,     # Greedy decoding for deterministic baseline
            pad_token_id=hf_tokenizer.pad_token_id,
            eos_token_id=stop_token_id
        )
        
        # Extract only the *new* tokens
        pred_ids = generated_ids[0, cut_idx:]
        
        # --- C. Decoding & Parsing ---
        pred_text = hf_tokenizer.decode(pred_ids, skip_special_tokens=True)
        gt_text = hf_tokenizer.decode(gt_ids, skip_special_tokens=True)
        
        pred_grid = parse_grid_from_text(pred_text)
        gt_grid = parse_grid_from_text(gt_text)
        
        # --- D. Scoring ---
        if gt_grid is None:
            # Should not happen if data is clean
            continue
            
        exact, pixel = calculate_metrics(pred_grid, gt_grid)
        
        total_tasks += 1
        total_exact_matches += exact
        total_pixel_acc += pixel
        
# --- Final Report ---
print("-" * 30)
print(f"Final Results on {total_tasks} Tasks")
print(f"Exact Match Accuracy: {100 * total_exact_matches / total_tasks:.2f}%")
print(f"Avg Pixel Accuracy:   {100 * total_pixel_acc / total_tasks:.2f}%")
print("-" * 30)


'''
ARC-ARC-1 results 12/30/25
------------------------------
Final Results on 400 Tasks
Exact Match Accuracy: 1.50%
Avg Pixel Accuracy:   14.10%
------------------------------
'''