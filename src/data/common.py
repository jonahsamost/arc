# data augmentations
import numpy as np
import copy
import json

PUZZLE_ID_SEPARATOR = "|"
DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]

class ARCAugmenter:
    def __init__(self):
        pass

    def dihedral_transform(self, arr: np.ndarray, tid: int) -> np.ndarray:
        """
        Applies one of 8 symmetries (D4 group): rotations and flips.
        """
        if tid == 0:
            return arr  # Identity
        elif tid == 1:
            return np.rot90(arr, k=1)  # 90 deg counter-clockwise
        elif tid == 2:
            return np.rot90(arr, k=2)  # 180 deg
        elif tid == 3:
            return np.rot90(arr, k=3)  # 270 deg (90 cw)
        elif tid == 4:
            return np.fliplr(arr)      # Horizontal flip
        elif tid == 5:
            return np.flipud(arr)      # Vertical flip
        elif tid == 6:
            return arr.T               # Transpose (Main diagonal)
        elif tid == 7:
            # Reflection along anti-diagonal
            return np.fliplr(np.rot90(arr, k=1))
        else:
            return arr

    def generate_augmentation_params(self):
        trans_id = np.random.randint(0, 8)
        
        # Permute colors 1-9, keep 0 (black/background) fixed
        perm = np.random.permutation(np.arange(1, 10, dtype=np.uint8))
        mapping = np.concatenate([np.arange(0, 1, dtype=np.uint8), perm])
        
        return trans_id, mapping

    def apply_augmentation(self, puzzle: dict, trans_id: int, mapping: np.ndarray) -> dict:
        augmented_puzzle = {'train': [], 'test': []}

        # Helper to process a single grid
        def transform_grid(grid_list):
            # 1. Convert to numpy
            arr = np.array(grid_list, dtype=np.uint8)
            # 2. Apply Color Permutation
            # mapping[arr] uses numpy advanced indexing to swap colors instantly
            colored_arr = mapping[arr]
            # 3. Apply Dihedral Transformation
            transformed_arr = self.dihedral_transform(colored_arr, trans_id)
            # 4. Convert back to list for JSON compatibility
            return transformed_arr.tolist()

        # Apply to Train
        for pair in puzzle['train']:
            aug_pair = {
                'input': transform_grid(pair['input']),
                'output': transform_grid(pair['output'])
            }
            augmented_puzzle['train'].append(aug_pair)

        # Apply to Test
        for pair in puzzle['test']:
            aug_pair = {
                'input': transform_grid(pair['input']),
                'output': transform_grid(pair['output'])
            }
            augmented_puzzle['test'].append(aug_pair)

        return augmented_puzzle

    def augment_puzzle(self, puzzle: dict, num_augmentations: int = 1) -> list:
        augmented_data = []

        for _ in range(num_augmentations):
            trans_id, mapping = self.generate_augmentation_params()
            
            # Create a unique name/ID for this augmentation (optional, useful for debugging)
            # Format: t{TRANS_ID}|{COLOR_MAPPING_STRING}
            mapping_str = "".join(str(x) for x in mapping)
            aug_id = f"t{trans_id}{PUZZLE_ID_SEPARATOR}{mapping_str}"
            
            aug_puzzle = self.apply_augmentation(puzzle, trans_id, mapping)
            
            # We wrap the result to include metadata if needed, 
            # or just return the raw puzzle dict. Here we return a dict 
            # containing the aug_id and the data.
            augmented_data.append({
                "aug_id": aug_id,
                "puzzle": aug_puzzle
            })

        return augmented_data
