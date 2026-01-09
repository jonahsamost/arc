import numpy as np

CANVAS_SIZE = 30
PUZZLE_ID_SEPARATOR = "|"

class ARCAugmenter:
    def __init__(self):
        pass

    def dihedral_transform(self, arr: np.ndarray, tid: int) -> np.ndarray:
        if tid == 0: return arr
        elif tid == 1: return np.rot90(arr, k=1)
        elif tid == 2: return np.rot90(arr, k=2)
        elif tid == 3: return np.rot90(arr, k=3)
        elif tid == 4: return np.fliplr(arr)
        elif tid == 5: return np.flipud(arr)
        elif tid == 6: return arr.T
        elif tid == 7: return np.fliplr(np.rot90(arr, k=1))
        return arr

    def generate_augmentation_params(self, augment: bool = False):
        if not augment:
            return 0, np.arange(0, 11, dtype=np.uint8) 

        trans_id = np.random.randint(0, 8)
        strategy_roll = np.random.random()
        if strategy_roll < 0.10: # identity
            mapping = np.arange(0, 11, dtype=np.uint8)
        elif strategy_roll < 0.50: # maintain closeness of colors
            k = np.random.randint(1, 9)
            mapping = np.arange(0, 10, dtype=np.uint8)
            mapping[1:] = ((np.arange(1, 10) - 1 + k) % 9) + 1
        else: # random permute
            perm = np.random.permutation(np.arange(1, 10, dtype=np.uint8))
            mapping = np.concatenate([np.arange(0, 1, dtype=np.uint8), perm])
            
        return trans_id, mapping

    def apply_canvas_jitter_legacy(self, input_grid: np.ndarray, output_grid: np.ndarray, no_jitter: bool = False):
        """
        LEGACY: Pads both grids to 30x30.
        WARNING: This inflates token count by ~30x! Use apply_transform instead.
        
        If shapes match, applies the SAME random translation (Jitter) to both.
        If shapes differ, centers them
        """
        h_in, w_in = input_grid.shape
        h_out, w_out = output_grid.shape
        
        new_input = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)
        new_output = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

        if no_jitter:
            new_input[:h_in, :w_in] = input_grid
            new_output[:h_out, :w_out] = output_grid
        
        elif (h_in == h_out) and (w_in == w_out):
            max_y = CANVAS_SIZE - h_in
            max_x = CANVAS_SIZE - w_in
            
            off_y = np.random.randint(0, max_y + 1)
            off_x = np.random.randint(0, max_x + 1)
            
            new_input[off_y : off_y+h_in, off_x : off_x+w_in] = input_grid
            new_output[off_y : off_y+h_out, off_x : off_x+w_out] = output_grid
            
        else:
            max_y_in = CANVAS_SIZE - h_in
            max_x_in = CANVAS_SIZE - w_in
            off_y_in = np.random.randint(0, max_y_in + 1)
            off_x_in = np.random.randint(0, max_x_in + 1)
            new_input[off_y_in : off_y_in+h_in, off_x_in : off_x_in+w_in] = input_grid

            max_y_out = CANVAS_SIZE - h_out
            max_x_out = CANVAS_SIZE - w_out
            off_y_out = np.random.randint(0, max_y_out + 1)
            off_x_out = np.random.randint(0, max_x_out + 1)
            new_output[off_y_out : off_y_out+h_out, off_x_out : off_x_out+w_out] = output_grid

        return new_input, new_output

    def augment_puzzle(
        self, 
        puzzle: dict, 
        num_augmentations: int = 1,
        augment: bool = True,  # Controls Color/Rotation
        jitter: bool = False   # DEPRECATED: Canvas jitter inflates tokens 30x, disabled by default
    ) -> list:
        """
        Augment a puzzle with color permutation and dihedral transforms.
        
        Args:
            puzzle: Dict with 'train' and 'test' keys containing input/output pairs
            num_augmentations: Number of augmented versions to generate
            augment: If True, apply random color permutation and rotation/flip
            jitter: DEPRECATED - If True, pads to 30x30 canvas (inflates tokens ~30x!)
        
        Returns:
            List of dicts with 'aug_id', 'puzzle', 'og_dims' keys
        """
        augmented_data = []

        for aug_idx in range(num_augmentations):
            # 1. Augmentation Params (Color/Rotation)
            # If augment=False, we force Identity (trans_id=0, mapping=0..9)
            trans_id, mapping = self.generate_augmentation_params(augment=augment)
            
            # Create ID for tracking
            mapping_str = "".join(str(x) for x in mapping) if augment else "identity"
            aug_id = f"t{trans_id}{PUZZLE_ID_SEPARATOR}{mapping_str}"
            
            aug_puzzle = {'train': [], 'test': []}
            all_pairs = puzzle['train'] + puzzle['test']
            
            task_og_dims = None 

            for i, pair in enumerate(all_pairs):
                in_arr = np.array(pair['input'], dtype=np.uint8)
                out_arr = np.array(pair['output'], dtype=np.uint8)
                
                # Capture original dims for Tokenizer/Cropping later
                h_in_raw, w_in_raw = in_arr.shape
                h_out_raw, w_out_raw = out_arr.shape
                
                # Update task dims if it's the test pair (last one)
                if i == len(all_pairs) - 1:
                    task_og_dims = [h_in_raw, w_in_raw, h_out_raw, w_out_raw]

                # --- STEP 1: Transformation ( Controlled by `augment` ) ---
                # Even if augment=False, this code runs safely because 
                # generate_augmentation_params returns identity mapping/transform.
                in_arr = mapping[in_arr]
                out_arr = mapping[out_arr]
                in_arr = self.dihedral_transform(in_arr, trans_id)
                out_arr = self.dihedral_transform(out_arr, trans_id)
                
                # --- STEP 2: Layout ( Controlled by `jitter` ) ---
                # NOTE: Canvas jitter is DISABLED by default because it inflates
                # every grid to 30x30, causing ~30x token inflation!
                # Only use for legacy compatibility.
                if jitter:
                    in_arr, out_arr = self.apply_canvas_jitter_legacy(
                        in_arr, out_arr, no_jitter=False
                    )
                # else: keep original dimensions (RECOMMENDED)
                
                new_pair = {'input': in_arr.tolist(), 'output': out_arr.tolist()}
                
                if i < len(puzzle['train']):
                    aug_puzzle['train'].append(new_pair)
                else:
                    aug_puzzle['test'].append(new_pair)

            augmented_data.append({
                "aug_id": aug_id,
                "puzzle": aug_puzzle,
                "og_dims": task_og_dims, 
            })

        return augmented_data