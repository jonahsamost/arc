import numpy as np
import random
import json
import os
import uuid
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict, Optional, Any

# ==========================================
# 0. CONFIGURATION & CONSTANTS
# ==========================================
CANVAS_MIN = 10
CANVAS_MAX = 30
COLORS = list(range(10))
BG_COLOR = 0

class ShapeType(str, Enum):
    RECT = "rect"
    HOLLOW_RECT = "hollow_rect"
    CIRCLE = "circle"      # Rough pixel circle
    LINE = "line"
    CROSS = "cross"
    L_SHAPE = "l_shape"
    PYRAMID = "pyramid"    # Little triangle
    SCATTER = "scatter"    # Cellular automata blob

class RenderStyle(str, Enum):
    ARCHITECT = "architect" # Perfect, clean
    KID = "kid"             # Wobbly, overshooting lines
    SCANNER = "scanner"     # Salt & pepper noise, missing rows
    GLITCH = "glitch"       # Block compression artifacts
    EROSION = "erosion"     # Missing chunks

@dataclass
class ARCObject:
    """
    The Noumenon (The Thing-in-Itself).
    Exists abstractly before being rendered.
    """
    color: int
    shape: ShapeType
    h: int
    w: int
    y: int = 0
    x: int = 0
    z_index: int = 0        # For Z-Buffer / Occlusion logic
    uid: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Physics Properties
    is_agent: bool = False  # Is this the "Cursor"?
    velocity: Tuple[int, int] = (0, 0)
    mass: int = 1
    
    def to_dict(self):
        return asdict(self)

@dataclass
class TaskOutput:
    """Container for the full Neuro-Symbolic data sample."""
    input_grid: List[List[int]]
    output_grid: List[List[int]]
    description: str          # The Natural Language Prompt (The Narrator)
    input_graph: List[dict]   # The Symbolic Truth (The Schema)
    output_graph: List[dict]
    category: str
    difficulty: int

# ==========================================
# 1. THE MULTI-ENGINE RENDERER (The Appearance)
# ==========================================
class GodRenderer:
    """
    Handles the projection of Abstract Objects into Pixel Space.
    Implements Z-Buffer and Style Transfer.
    """
    def __init__(self, style: RenderStyle = RenderStyle.ARCHITECT):
        self.style = style

    def render_scene(self, shape: Tuple[int, int], objects: List[ARCObject]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Renders a full scene with occlusion.
        Returns: (Visual Grid, Object ID Grid)
        """
        H, W = shape
        visual_grid = np.full((H, W), BG_COLOR, dtype=np.uint8)
        id_grid = np.full((H, W), -1, dtype=np.int32) # For internal tracking
        
        # 1. Sort by Z-Index (Painter's Algorithm)
        # Higher Z draws last (on top)
        objects.sort(key=lambda o: o.z_index)
        
        for obj in objects:
            sprite = self._generate_base_sprite(obj)
            sprite = self._apply_style(sprite, obj.color)
            
            # Blit to canvas
            sh, sw = sprite.shape
            # Clipping
            y_start = max(0, obj.y)
            y_end = min(H, obj.y + sh)
            x_start = max(0, obj.x)
            x_end = min(W, obj.x + sw)
            
            # Sprite offsets (if clipped)
            sy_start = max(0, -obj.y)
            sy_end = sy_start + (y_end - y_start)
            sx_start = max(0, -obj.x)
            sx_end = sx_start + (x_end - x_start)
            
            if sy_end > sy_start and sx_end > sx_start:
                sub_sprite = sprite[sy_start:sy_end, sx_start:sx_end]
                mask = sub_sprite > 0
                
                # Visual Blit
                target_slice = visual_grid[y_start:y_end, x_start:x_end]
                target_slice[mask] = sub_sprite[mask]
                
                # ID Blit (helps us track logic even if occlusion happens)
                id_slice = id_grid[y_start:y_end, x_start:x_end]
                # We can map UUID hash to int for ID grid if needed, 
                # here we just mark presence
                id_slice[mask] = 1 
                
        return visual_grid, id_grid

    def _generate_base_sprite(self, obj: ARCObject) -> np.ndarray:
        s = np.zeros((obj.h, obj.w), dtype=np.uint8)
        c = obj.color
        h, w = obj.h, obj.w
        
        if obj.shape == ShapeType.RECT:
            s[:] = c
        elif obj.shape == ShapeType.HOLLOW_RECT:
            s[:] = c
            if h > 2 and w > 2: s[1:-1, 1:-1] = 0
        elif obj.shape == ShapeType.LINE:
            # Decide orientation deterministically based on hash of UID to be consistent per object instance
            seed = sum(ord(char) for char in obj.uid) % 2
            if seed == 0: s[h//2, :] = c
            else: s[:, w//2] = c
        elif obj.shape == ShapeType.CROSS:
            s[h//2, :] = c
            s[:, w//2] = c
        elif obj.shape == ShapeType.PYRAMID:
            # Simple triangle
            for r in range(h):
                width_at_row = max(1, int(w * (r / h)))
                start = (w - width_at_row) // 2
                s[r, start:start+width_at_row] = c
        elif obj.shape == ShapeType.SCATTER:
            # Deterministic cellular automata
            rng = np.random.RandomState(sum(ord(x) for x in obj.uid))
            cy, cx = h//2, w//2
            s[cy, cx] = c
            for _ in range(int(h*w*0.7)):
                ys, xs = np.where(s == c)
                if len(ys) == 0: break
                idx = rng.randint(len(ys))
                ny, nx = ys[idx] + rng.randint(-1, 2), xs[idx] + rng.randint(-1, 2)
                if 0 <= ny < h and 0 <= nx < w: s[ny, nx] = c
                
        return s

    def _apply_style(self, sprite: np.ndarray, color: int) -> np.ndarray:
        """Applies the 'Glitch' or 'Kid' effects to the perfect sprite."""
        if self.style == RenderStyle.ARCHITECT:
            return sprite
            
        h, w = sprite.shape
        noisy = sprite.copy()
        mask = sprite > 0
        
        if self.style == RenderStyle.KID:
            # Wobbly lines: Shift rows/cols randomly
            for r in range(h):
                if np.random.random() < 0.3:
                    shift = np.random.randint(-1, 2)
                    noisy[r, :] = np.roll(noisy[r, :], shift)
            # Overshoot?
            if np.random.random() < 0.2:
                # Add a random pixel near edge
                ys, xs = np.where(mask)
                if len(ys) > 0:
                    idx = np.random.randint(len(ys))
                    ny, nx = ys[idx] + np.random.randint(-1, 2), xs[idx] + np.random.randint(-1, 2)
                    if 0 <= ny < h and 0 <= nx < w: noisy[ny, nx] = color
                    
        elif self.style == RenderStyle.SCANNER:
            # Salt noise
            noise_map = np.random.random((h, w))
            noisy[(noise_map < 0.05) & (noisy == 0)] = color # Add speckles
            noisy[(noise_map > 0.95) & (noisy != 0)] = 0     # Remove speckles
            
        elif self.style == RenderStyle.GLITCH:
            # Block shifts
            if h > 4:
                split = np.random.randint(1, h-1)
                noisy[split:, :] = np.roll(noisy[split:, :], shift=np.random.randint(-2, 3), axis=1)
                
        elif self.style == RenderStyle.EROSION:
            # Random dropout
            drop_mask = (np.random.random((h, w)) < 0.2) & mask
            noisy[drop_mask] = 0
            
        return noisy

# ==========================================
# 2. THE LOGIC ENGINE (The Noumenon)
# ==========================================
class GodGenerator:
    def __init__(self, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
    # --- HELPER: Object Placement ---
    def place_objects_no_overlap(self, H, W, objects: List[ARCObject]) -> bool:
        """Modifies objects in-place with x,y coordinates."""
        occupied = np.zeros((H, W), dtype=bool)
        
        # Sort large to small
        sorted_objs = sorted(objects, key=lambda o: o.h*o.w, reverse=True)
        
        for obj in sorted_objs:
            placed = False
            for _ in range(50):
                y = np.random.randint(0, max(1, H - obj.h))
                x = np.random.randint(0, max(1, W - obj.w))
                
                # Check bounds
                if occupied[y:y+obj.h, x:x+obj.w].any():
                    continue
                    
                # Place
                obj.y, obj.x = y, x
                occupied[y:y+obj.h, x:x+obj.w] = True
                placed = True
                break
            if not placed: return False
        return True

    # ==========================================
    # TASK 1: THE OCCLUDED AGENT (Apperception + Object Permanence)
    # ==========================================
    def task_occlusion_tracking(self, difficulty=1) -> TaskOutput:
        """
        Task: An agent moves from A to B. It passes BEHIND a wall.
        The model must render the path, including the 'hidden' steps (or knowing it emerges).
        Critique Addressed: Z-Buffer / Depth.
        """
        H, W = 20, 20
        style = random.choice(list(RenderStyle))
        renderer = GodRenderer(style)
        
        # 1. Setup Scene
        agent_color = 1 # Blue
        wall_color = 2  # Red
        
        # Agent starts left, moves right
        agent = ARCObject(color=agent_color, shape=ShapeType.RECT, h=2, w=2, 
                          y=10, x=2, z_index=0) # Z=0 (Behind)
        
        # Wall is in the middle
        wall = ARCObject(color=wall_color, shape=ShapeType.RECT, h=15, w=3, 
                         y=2, x=9, z_index=10) # Z=10 (In Front)
        
        # 2. Input: Agent at Start + Wall
        objs_in = [agent, wall]
        input_grid, _ = renderer.render_scene((H, W), objs_in)
        
        # 3. Logic: Simulation (Move Right)
        # The agent moves 15 steps right.
        path_pixels = []
        final_agent = ARCObject(**agent.to_dict()) # Clone
        
        steps = 15
        for _ in range(steps):
            final_agent.x += 1
            # Record "Center" for path
            cy, cx = final_agent.y + 1, final_agent.x + 1
            if 0 <= cx < W:
                path_pixels.append((cy, cx))
        
        # 4. Output: Agent at End + Path Trace
        # Crucial: The path trace should be BEHIND the wall if Z-buffered
        # But usually in ARC, a "trace" overwrites background but is occluded by foreground
        
        # Create trace object
        trace_grid = np.zeros((H, W), dtype=np.uint8)
        for r, c in path_pixels:
            trace_grid[r, c] = 4 # Yellow Path
            
        # Manually compose Output to ensure strict logic
        # Output = Wall (Top) + Agent (Bottom) + Trace (Bottom)
        # Wait, if Agent is Z=0 and Wall is Z=10, Agent is hidden.
        # Task: "Draw the path." 
        # If the path goes behind the wall, does it show?
        # Logic: "X-Ray Vision" vs "Physical Occlusion".
        # Let's do: Agent pops out other side.
        
        objs_out = [final_agent, wall]
        output_grid, _ = renderer.render_scene((H, W), objs_out)
        
        # Draw path on output, but respect wall occlusion?
        # Let's create path object with Z=-1 (very back)
        path_obj = ARCObject(color=4, shape=ShapeType.SCATTER, h=H, w=W, z_index=-1)
        # We cheat and just bake the pixels into a custom shape for the path
        # (Simplified for this generator logic)
        for r, c in path_pixels:
             if output_grid[r, c] == BG_COLOR: # Only draw if empty
                 output_grid[r, c] = 4
        
        # Re-render wall on top to be sure
        wall_sprite = renderer._generate_base_sprite(wall)
        y, x = wall.y, wall.x
        h_w, w_w = wall.h, wall.w
        output_grid[y:y+h_w, x:x+w_w] = wall_sprite

        desc = "An agent moves right, passing behind the wall. Predict its final position."
        
        return TaskOutput(
            input_grid=input_grid.tolist(),
            output_grid=output_grid.tolist(),
            description=desc,
            input_graph=[o.to_dict() for o in objs_in],
            output_graph=[o.to_dict() for o in objs_out],
            category="apperception_occlusion",
            difficulty=difficulty
        )

    # ==========================================
    # TASK 2: THE CONTINGENT SORTER (Antinomy)
    # ==========================================
    def task_contingency_sorting(self, difficulty=1) -> TaskOutput:
        """
        Task: Sort objects by Size or by Color, depending on a 'Key' object.
        Critique Addressed: Global Context / Rule Flipping.
        """
        H, W = 15, 15
        renderer = GodRenderer(RenderStyle.ARCHITECT) # Keep clean for sorting tasks
        
        # 1. The Key (The Context)
        key_color = random.choice([2, 3]) # Red or Green
        key_obj = ARCObject(color=key_color, shape=ShapeType.PYRAMID, h=3, w=3, y=0, x=0)
        
        # 2. The Data (Random Objects)
        data_objs = []
        for _ in range(5):
            c = random.randint(4, 9)
            s = random.randint(2, 4)
            obj = ARCObject(color=c, shape=ShapeType.RECT, h=s, w=s, z_index=5)
            data_objs.append(obj)
            
        # Place them
        all_objs = [key_obj] + data_objs
        if not self.place_objects_no_overlap(H, W, all_objs):
            return self.task_contingency_sorting(difficulty) # Retry
            
        input_grid, _ = renderer.render_scene((H, W), all_objs)
        
        # 3. Logic
        # Rule: If Red(2) -> Sort Vertical by Size (Small top).
        #       If Green(3) -> Sort Horizontal by Color (Hue).
        
        sorted_objs = []
        if key_color == 2: # Size Sort
            data_objs.sort(key=lambda o: o.h * o.w)
            # Stack Vertically
            curr_y = 4
            for o in data_objs:
                o.y = curr_y
                o.x = W // 2
                curr_y += o.h + 1
                sorted_objs.append(o)
            desc = "Context is Red: Stack objects vertically sorted by size."
        else: # Color Sort
            data_objs.sort(key=lambda o: o.color)
            # Stack Horizontally
            curr_x = 4
            for o in data_objs:
                o.y = H // 2
                o.x = curr_x
                curr_x += o.w + 1
                sorted_objs.append(o)
            desc = "Context is Green: Align objects horizontally sorted by color."
            
        # Key object remains
        output_objs = [key_obj] + sorted_objs
        output_grid, _ = renderer.render_scene((H, W), output_objs)
        
        return TaskOutput(
            input_grid=input_grid.tolist(),
            output_grid=output_grid.tolist(),
            description=desc,
            input_graph=[o.to_dict() for o in all_objs],
            output_graph=[o.to_dict() for o in output_objs],
            category="contingency_sorting",
            difficulty=difficulty
        )

    # ==========================================
    # TASK 3: THE GESTALT REPAIR (Quality + Negation)
    # ==========================================
    def task_gestalt_repair(self, difficulty=1) -> TaskOutput:
        """
        Task: Input is a 'Glitch' or 'Erosion' version of a shape. Output is the 'Architect' version.
        Critique Addressed: Multi-Engine Rendering / Platonic Ideals.
        """
        H, W = 12, 12
        # Input uses a messy renderer
        style_in = random.choice([RenderStyle.GLITCH, RenderStyle.EROSION, RenderStyle.SCANNER])
        ren_in = GodRenderer(style_in)
        # Output uses the perfect renderer
        ren_out = GodRenderer(RenderStyle.ARCHITECT)
        
        # 1. The Object
        shape = random.choice([ShapeType.HOLLOW_RECT, ShapeType.CROSS, ShapeType.L_SHAPE])
        color = random.randint(1, 9)
        obj = ARCObject(color=color, shape=shape, h=8, w=8, y=2, x=2)
        
        # 2. Render Input (Messy)
        # We might add 'distractors' (noise pixels) that are NOT part of the object
        input_grid, _ = ren_in.render_scene((H, W), [obj])
        
        # Add extra noise (Negation)
        noise_mask = np.random.random((H, W)) < 0.1
        input_grid[noise_mask & (input_grid == 0)] = random.randint(1, 9)
        
        # 3. Render Output (Platonic)
        # The object, perfectly centered, perfectly drawn.
        # ARC logic often involves centering the repaired object.
        obj.y = (H - obj.h) // 2
        obj.x = (W - obj.w) // 2
        output_grid, _ = ren_out.render_scene((H, W), [obj])
        
        desc = f"Denoise the grid. Extract the {shape.value} and center it, removing all artifacts."
        
        return TaskOutput(
            input_grid=input_grid.tolist(),
            output_grid=output_grid.tolist(),
            description=desc,
            input_graph=[obj.to_dict()],
            output_graph=[obj.to_dict()],
            category="gestalt_repair",
            difficulty=difficulty
        )

    # ==========================================
    # MAIN GENERATION LOOP
    # ==========================================
    def generate_batch(self, count: int, save_dir: str):
        """Generates a mix of tasks."""
        os.makedirs(save_dir, exist_ok=True)
        
        generators = [
            self.task_occlusion_tracking,
            self.task_contingency_sorting,
            self.task_gestalt_repair
        ]
        
        print(f"Initializing God Protocol Generation...")
        
        for i in range(count):
            gen_func = random.choice(generators)
            
            # Generate Train Pairs (3 shots)
            train_pairs = []
            desc = ""
            try:
                # We need consistent logic for the shots.
                # In this simplified batcher, we call the function 3 times.
                # NOTE: Real ARC generators enforce the EXACT same rule instance (e.g. Key=Red always).
                # For `task_contingency`, the rule is the conditional itself, so it's fine if keys change.
                
                # Shot 1
                t1 = gen_func()
                train_pairs.append({"input": t1.input_grid, "output": t1.output_grid})
                desc = t1.description
                
                # Shot 2
                t2 = gen_func()
                train_pairs.append({"input": t2.input_grid, "output": t2.output_grid})
                
                # Shot 3
                t3 = gen_func()
                train_pairs.append({"input": t3.input_grid, "output": t3.output_grid})
                
                # Test Pair
                tt = gen_func()
                test_pair = [{"input": tt.input_grid, "output": tt.output_grid}]
                
                # Symbolic Payload (Ground Truth for the Test input)
                symbolic_data = {
                    "train_graphs": [t1.input_graph, t2.input_graph, t3.input_graph],
                    "test_graph": tt.input_graph,
                    "test_graph_out": tt.output_graph,
                    "description": desc
                }
                
                final_task = {
                    "train": train_pairs,
                    "test": test_pair,
                    "metadata": symbolic_data,
                    "category": tt.category
                }
                
                # Save
                fname = f"god_task_{i:05d}_{tt.category}.json"
                with open(os.path.join(save_dir, fname), 'w') as f:
                    json.dump(final_task, f, indent=2)
                    
            except Exception as e:
                print(f"Failed generation {i}: {e}")
                continue

# ==========================================
# CLI
# ==========================================
# if __name__ == "__main__":
#     gen = GodGenerator(seed=42)
#     gen.generate_batch(100, "data/kantian_god_v4")
#     print("Generation Complete. Welcome to the Real World.")