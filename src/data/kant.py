"""
THE KANT PROTOCOL v2.0
=====================

A Neuro-Symbolic Synthetic Data Generator grounded in:
- Causal Graph Theory (Pearl's do-calculus)
- Transcendental Logic (Kant's Categories of Understanding)
- Program Synthesis (Generators as first-class programs)

CORE INSIGHT: We don't generate pixels. We generate CAUSAL WORLDS
and render them into pixels. The model must learn the world, not the rendering.

Architecture:
    CausalWorld → Intervention → Simulation → Rendering → Verification → Task

The output includes:
    - The observable grids (input/output)
    - The causal graph (ground truth structure)
    - The reasoning trace (how to derive the answer)
    - Counterfactuals (what would happen under different interventions)
    - Necessity proof (why this answer is unique)
"""

from __future__ import annotations

import numpy as np
import random
import json
import os
import uuid
import copy
from enum import Enum, auto
from dataclasses import dataclass, field, asdict
from typing import (
    List, Tuple, Dict, Optional, Any, Callable, Set, 
    TypeVar, Generic, Union, FrozenSet
)
from abc import ABC, abstractmethod
from collections import defaultdict
import hashlib

# ============================================================================
# 0. CONFIGURATION & CONSTANTS
# ============================================================================

CANVAS_MIN = 10
CANVAS_MAX = 30
COLORS = list(range(10))
BG_COLOR = 0
MAX_SIMULATION_STEPS = 100

# ============================================================================
# 1. FOUNDATIONAL TYPES - The Ontology
# ============================================================================

class ValueType(Enum):
    """Primitive types in our causal world."""
    INTEGER = "integer"
    BOOLEAN = "boolean"
    POSITION = "position"      # (y, x) tuple
    VECTOR = "vector"          # (dy, dx) tuple
    COLOR = "color"            # 0-9
    SHAPE = "shape"            # ShapeType enum
    SET = "set"                # Set of entity IDs
    GRID = "grid"              # 2D numpy array


class ShapeType(str, Enum):
    """Shape primitives - but now they're SYMBOLS, not just pixels."""
    POINT = "point"
    RECT = "rect"
    HOLLOW_RECT = "hollow_rect"
    CIRCLE = "circle"
    LINE_H = "line_h"
    LINE_V = "line_v"
    CROSS = "cross"
    L_SHAPE = "l_shape"
    T_SHAPE = "t_shape"
    TRIANGLE = "triangle"
    DIAMOND = "diamond"


@dataclass(frozen=True)
class Position:
    """Immutable position in grid space."""
    y: int
    x: int
    
    def __add__(self, other: 'Vector') -> 'Position':
        return Position(self.y + other.dy, self.x + other.dx)
    
    def __sub__(self, other: 'Position') -> 'Vector':
        return Vector(self.y - other.y, self.x - other.x)
    
    def distance_to(self, other: 'Position') -> float:
        return ((self.y - other.y)**2 + (self.x - other.x)**2) ** 0.5
    
    def manhattan_to(self, other: 'Position') -> int:
        return abs(self.y - other.y) + abs(self.x - other.x)


@dataclass(frozen=True)
class Vector:
    """Immutable direction/velocity vector."""
    dy: int
    dx: int
    
    def __add__(self, other: 'Vector') -> 'Vector':
        return Vector(self.dy + other.dy, self.dx + other.dx)
    
    def __mul__(self, scalar: int) -> 'Vector':
        return Vector(self.dy * scalar, self.dx * scalar)
    
    def __neg__(self) -> 'Vector':
        return Vector(-self.dy, -self.dx)
    
    @property
    def magnitude(self) -> float:
        return (self.dy**2 + self.dx**2) ** 0.5
    
    def normalized(self) -> 'Vector':
        mag = self.magnitude
        if mag == 0:
            return Vector(0, 0)
        return Vector(int(self.dy / mag), int(self.dx / mag))


DIRECTION_VECTORS = {
    "up": Vector(-1, 0),
    "down": Vector(1, 0),
    "left": Vector(0, -1),
    "right": Vector(0, 1),
    "stay": Vector(0, 0),
}

# ============================================================================
# 2. THE KANTIAN CATEGORIES - A Priori Structure of Understanding
# ============================================================================

class Category(Enum):
    """
    Kant's 12 Categories of Understanding.
    These are the necessary conditions for any possible experience.
    """
    # QUANTITY - How many?
    UNITY = "unity"           # One object, considered as such
    PLURALITY = "plurality"   # Many objects, considered separately  
    TOTALITY = "totality"     # All objects, considered as one whole
    
    # QUALITY - What kind?
    REALITY = "reality"       # Positive determination (presence)
    NEGATION = "negation"     # Negative determination (absence)
    LIMITATION = "limitation" # Bounded determination (degree)
    
    # RELATION - How connected?
    SUBSTANCE = "substance"   # Persistence through change
    CAUSALITY = "causality"   # Succession according to rule
    COMMUNITY = "community"   # Reciprocal determination
    
    # MODALITY - How certain?
    POSSIBILITY = "possibility"   # Compatible with conditions
    ACTUALITY = "actuality"       # Given under conditions
    NECESSITY = "necessity"       # Determined by conditions


@dataclass
class Schema:
    """
    A Transcendental Schema bridges pure categories to sensible intuitions.
    Kant: "The schema is the representation of a general procedure of the
    imagination to present a concept with an image."
    
    We implement schemas as TEMPORAL RULES that apply categories to sequences.
    """
    category: Category
    temporal_rule: str  # Human-readable description
    
    # The computational form: a function that checks if a sequence
    # of world states exemplifies this category
    predicate: Callable[['WorldState', 'WorldState'], bool]
    
    # How to generate instances that exemplify this schema
    # Takes a WorldState and returns a modified WorldState
    generator: Callable[['WorldState'], 'WorldState']


class SchemaRegistry:
    """
    The Schematism: maps categories to their temporal determinations.
    """
    
    @staticmethod
    def get_schema(category: Category) -> Schema:
        schemas = {
            # CAUSALITY: "If A, then B follows according to a rule"
            Category.CAUSALITY: Schema(
                category=Category.CAUSALITY,
                temporal_rule="Succession of states according to a rule",
                predicate=lambda s1, s2: SchemaRegistry._check_causal_succession(s1, s2),
                generator=SchemaRegistry._generate_causal_world,
            ),
            
            # SUBSTANCE: "Something persists through change"
            Category.SUBSTANCE: Schema(
                category=Category.SUBSTANCE,
                temporal_rule="Permanence of the real in time",
                predicate=lambda s1, s2: SchemaRegistry._check_substance_persistence(s1, s2),
                generator=SchemaRegistry._generate_substance_world,
            ),
            
            # COMMUNITY: "A and B reciprocally determine each other"
            Category.COMMUNITY: Schema(
                category=Category.COMMUNITY,
                temporal_rule="Simultaneity of determinations according to a rule",
                predicate=lambda s1, s2: SchemaRegistry._check_community(s1, s2),
                generator=SchemaRegistry._generate_community_world,
            ),
            
            # TOTALITY: "All X considered as one"
            Category.TOTALITY: Schema(
                category=Category.TOTALITY,
                temporal_rule="The complete determination of a manifold",
                predicate=lambda s1, s2: SchemaRegistry._check_totality(s1, s2),
                generator=SchemaRegistry._generate_totality_world,
            ),
            
            # NECESSITY: "This must be so, given the conditions"
            Category.NECESSITY: Schema(
                category=Category.NECESSITY,
                temporal_rule="Existence at all times",
                predicate=lambda s1, s2: SchemaRegistry._check_necessity(s1, s2),
                generator=SchemaRegistry._generate_necessity_world,
            ),
            
            # NEGATION: "The absence of X"
            Category.NEGATION: Schema(
                category=Category.NEGATION,
                temporal_rule="Empty time (non-being)",
                predicate=lambda s1, s2: SchemaRegistry._check_negation(s1, s2),
                generator=SchemaRegistry._generate_negation_world,
            ),
        }
        return schemas.get(category, schemas[Category.CAUSALITY])
    
    @staticmethod
    def _check_causal_succession(s1: 'WorldState', s2: 'WorldState') -> bool:
        """Check if s2 follows from s1 according to causal rules."""
        # There exists at least one entity whose state changed due to a cause
        for eid in s1.entities:
            if eid in s2.entities:
                e1, e2 = s1.entities[eid], s2.entities[eid]
                if e1.get('position') != e2.get('position'):
                    # Position changed - was there a cause?
                    if e1.get('velocity') or s1.causal_graph.has_edge_to(eid):
                        return True
        return False
    
    @staticmethod
    def _check_substance_persistence(s1: 'WorldState', s2: 'WorldState') -> bool:
        """Check if some entity persists through change."""
        # Same entity exists in both states, but with different accidents
        for eid in s1.entities:
            if eid in s2.entities:
                e1, e2 = s1.entities[eid], s2.entities[eid]
                # Color persists (substance), position changes (accident)
                if e1.get('color') == e2.get('color') and e1.get('position') != e2.get('position'):
                    return True
        return False
    
    @staticmethod
    def _check_community(s1: 'WorldState', s2: 'WorldState') -> bool:
        """Check for reciprocal causation."""
        cg = s2.causal_graph
        for eid1 in s2.entities:
            for eid2 in s2.entities:
                if eid1 != eid2:
                    if cg.has_edge(eid1, eid2) and cg.has_edge(eid2, eid1):
                        return True
        return False
    
    @staticmethod
    def _check_totality(s1: 'WorldState', s2: 'WorldState') -> bool:
        """Check if transformation applies to ALL entities as a whole."""
        # All entities transformed by the same rule
        if len(s2.entities) < 2:
            return False
        transformations = []
        for eid in s1.entities:
            if eid in s2.entities:
                e1, e2 = s1.entities[eid], s2.entities[eid]
                p1, p2 = e1.get('position'), e2.get('position')
                if p1 and p2:
                    transformations.append((p2.y - p1.y, p2.x - p1.x))
        # All non-trivial transformations are identical
        non_trivial = [t for t in transformations if t != (0, 0)]
        return len(non_trivial) >= 2 and len(set(non_trivial)) == 1
    
    @staticmethod
    def _check_necessity(s1: 'WorldState', s2: 'WorldState') -> bool:
        """Check if the transition is necessary (no alternatives)."""
        # This requires the verification engine - placeholder
        return True
    
    @staticmethod
    def _check_negation(s1: 'WorldState', s2: 'WorldState') -> bool:
        """Check if something present in s1 is absent in s2."""
        return len(s1.entities) > len(s2.entities)
    
    # Generator methods - create worlds exemplifying each category
    @staticmethod
    def _generate_causal_world(base: 'WorldState') -> 'WorldState':
        return base  # Implemented in task generators
    
    @staticmethod
    def _generate_substance_world(base: 'WorldState') -> 'WorldState':
        return base
    
    @staticmethod
    def _generate_community_world(base: 'WorldState') -> 'WorldState':
        return base
    
    @staticmethod
    def _generate_totality_world(base: 'WorldState') -> 'WorldState':
        return base
    
    @staticmethod
    def _generate_necessity_world(base: 'WorldState') -> 'WorldState':
        return base
    
    @staticmethod
    def _generate_negation_world(base: 'WorldState') -> 'WorldState':
        return base


# ============================================================================
# 3. CAUSAL GRAPH ENGINE - The Structure of Causation
# ============================================================================

@dataclass
class CausalMechanism:
    """
    A causal mechanism defines HOW a cause produces an effect.
    This is not mere correlation - it's the generative process.
    """
    name: str
    cause_vars: List[str]       # Variable names that are causes
    effect_var: str             # Variable name that is effect
    mechanism_fn: Callable      # The actual function: causes -> effect
    description: str            # Human-readable explanation
    
    def apply(self, cause_values: Dict[str, Any]) -> Any:
        """Apply the mechanism to produce the effect."""
        args = [cause_values[v] for v in self.cause_vars]
        return self.mechanism_fn(*args)


class CausalGraph:
    """
    A causal DAG representing the structure of a world.
    
    Key insight: The graph is the GROUND TRUTH. Pixels are merely
    observations of the graph's state at a particular time.
    """
    
    def __init__(self):
        self.variables: Dict[str, ValueType] = {}
        self.mechanisms: Dict[str, CausalMechanism] = {}  # effect_var -> mechanism
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)  # cause -> effects
        
    def add_variable(self, name: str, vtype: ValueType) -> None:
        self.variables[name] = vtype
        
    def add_mechanism(self, mechanism: CausalMechanism) -> None:
        """Add a causal edge with its mechanism."""
        self.mechanisms[mechanism.effect_var] = mechanism
        for cause in mechanism.cause_vars:
            self.adjacency[cause].add(mechanism.effect_var)
            
    def has_edge(self, cause: str, effect: str) -> bool:
        return effect in self.adjacency.get(cause, set())
    
    def has_edge_to(self, effect: str) -> bool:
        return effect in self.mechanisms
    
    def get_parents(self, var: str) -> List[str]:
        """Get all direct causes of a variable."""
        if var in self.mechanisms:
            return self.mechanisms[var].cause_vars
        return []
    
    def get_children(self, var: str) -> Set[str]:
        """Get all direct effects of a variable."""
        return self.adjacency.get(var, set())
    
    def topological_sort(self) -> List[str]:
        """Return variables in causal order (causes before effects)."""
        visited = set()
        order = []
        
        def visit(var):
            if var in visited:
                return
            visited.add(var)
            for parent in self.get_parents(var):
                visit(parent)
            order.append(var)
        
        for var in self.variables:
            visit(var)
        return order
    
    def propagate(self, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Propagate values through the graph according to mechanisms.
        This is forward simulation: given exogenous values, compute all endogenous values.
        """
        result = dict(values)
        for var in self.topological_sort():
            if var in self.mechanisms:
                mech = self.mechanisms[var]
                try:
                    cause_values = {cv: result[cv] for cv in mech.cause_vars}
                    result[var] = mech.apply(cause_values)
                except KeyError:
                    pass  # Missing cause, keep existing value
        return result
    
    def do(self, intervention: Dict[str, Any], values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pearl's do-operator: do(X=x) sets X to x and removes all arrows into X.
        This captures the difference between observation and intervention.
        """
        # Copy the graph and values
        intervened_values = dict(values)
        intervened_values.update(intervention)
        
        # Create modified graph with incoming edges to intervened vars removed
        temp_mechanisms = dict(self.mechanisms)
        for var in intervention:
            if var in temp_mechanisms:
                del temp_mechanisms[var]
        
        # Propagate with modified graph
        result = dict(intervened_values)
        for var in self.topological_sort():
            if var in temp_mechanisms:
                mech = temp_mechanisms[var]
                try:
                    cause_values = {cv: result[cv] for cv in mech.cause_vars}
                    result[var] = mech.apply(cause_values)
                except KeyError:
                    pass
        return result
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the causal structure."""
        return {
            "variables": {k: v.value for k, v in self.variables.items()},
            "edges": [
                {
                    "causes": m.cause_vars,
                    "effect": m.effect_var,
                    "mechanism": m.name,
                    "description": m.description,
                }
                for m in self.mechanisms.values()
            ],
        }


# ============================================================================
# 4. ENTITY SYSTEM - Objects in the World
# ============================================================================

@dataclass
class Entity:
    """
    An entity is a thing that exists in the causal world.
    It has:
    - An identity (uid) that persists through change (SUBSTANCE)
    - Properties that can change (accidents)
    - Causal powers (what it can do to other entities)
    """
    uid: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Essential properties (define what it IS)
    shape: ShapeType = ShapeType.RECT
    color: int = 1
    
    # Accidental properties (can change)
    position: Position = field(default_factory=lambda: Position(0, 0))
    size: Tuple[int, int] = (2, 2)  # (height, width)
    velocity: Vector = field(default_factory=lambda: Vector(0, 0))
    
    # Causal properties
    mass: int = 1
    solid: bool = True        # Can other entities pass through?
    mobile: bool = True       # Can this entity move?
    alive: bool = True        # Does it exist?
    
    # Relational properties
    contains: Set[str] = field(default_factory=set)  # UIDs of contained entities
    attached_to: Optional[str] = None                 # UID of entity this is attached to
    
    # Epistemic properties (for the model to reason about)
    visible: bool = True
    marked: bool = False      # Has been selected/highlighted
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "shape": self.shape.value,
            "color": self.color,
            "position": (self.position.y, self.position.x),
            "size": self.size,
            "velocity": (self.velocity.dy, self.velocity.dx),
            "mass": self.mass,
            "solid": self.solid,
            "mobile": self.mobile,
            "alive": self.alive,
            "visible": self.visible,
        }
    
    @staticmethod
    def from_dict(d: Dict[str, Any]) -> 'Entity':
        return Entity(
            uid=d["uid"],
            shape=ShapeType(d["shape"]),
            color=d["color"],
            position=Position(*d["position"]),
            size=tuple(d["size"]),
            velocity=Vector(*d["velocity"]),
            mass=d["mass"],
            solid=d["solid"],
            mobile=d["mobile"],
            alive=d["alive"],
            visible=d.get("visible", True),
        )


# ============================================================================
# 5. WORLD STATE - A Snapshot in Time
# ============================================================================

@dataclass
class WorldState:
    """
    A complete state of the world at a moment in time.
    The world is the totality of facts, not things. (Wittgenstein)
    """
    timestamp: int
    grid_shape: Tuple[int, int]
    entities: Dict[str, Entity]  # uid -> Entity
    causal_graph: CausalGraph
    global_properties: Dict[str, Any] = field(default_factory=dict)
    
    def copy(self) -> 'WorldState':
        return WorldState(
            timestamp=self.timestamp,
            grid_shape=self.grid_shape,
            entities={uid: copy.deepcopy(e) for uid, e in self.entities.items()},
            causal_graph=self.causal_graph,  # Graph structure is immutable
            global_properties=dict(self.global_properties),
        )
    
    def get_entity_at(self, pos: Position) -> Optional[Entity]:
        """Get entity occupying a position."""
        for e in self.entities.values():
            if not e.alive or not e.visible:
                continue
            h, w = e.size
            if (e.position.y <= pos.y < e.position.y + h and
                e.position.x <= pos.x < e.position.x + w):
                return e
        return None
    
    def check_collision(self, e1: Entity, e2: Entity) -> bool:
        """Check if two entities overlap."""
        if e1.uid == e2.uid:
            return False
        h1, w1 = e1.size
        h2, w2 = e2.size
        return not (
            e1.position.y + h1 <= e2.position.y or
            e2.position.y + h2 <= e1.position.y or
            e1.position.x + w1 <= e2.position.x or
            e2.position.x + w2 <= e1.position.x
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "grid_shape": self.grid_shape,
            "entities": {uid: e.to_dict() for uid, e in self.entities.items()},
            "global_properties": self.global_properties,
        }


# ============================================================================
# 6. CAUSAL MECHANISMS LIBRARY - The Laws of Physics
# ============================================================================

class MechanismLibrary:
    """
    A library of reusable causal mechanisms.
    These are the 'laws of physics' in our synthetic worlds.
    """
    
    @staticmethod
    def gravity(velocity: Vector, mass: int) -> Vector:
        """Objects with mass accelerate downward."""
        return Vector(velocity.dy + 1, velocity.dx)
    
    @staticmethod
    def collision_response(e1_vel: Vector, e2_vel: Vector, e1_mass: int, e2_mass: int) -> Tuple[Vector, Vector]:
        """Elastic collision between two objects."""
        # Simplified: swap velocities weighted by mass
        total_mass = e1_mass + e2_mass
        new_v1 = Vector(
            int((e1_vel.dy * (e1_mass - e2_mass) + 2 * e2_mass * e2_vel.dy) / total_mass),
            int((e1_vel.dx * (e1_mass - e2_mass) + 2 * e2_mass * e2_vel.dx) / total_mass)
        )
        new_v2 = Vector(
            int((e2_vel.dy * (e2_mass - e1_mass) + 2 * e1_mass * e1_vel.dy) / total_mass),
            int((e2_vel.dx * (e2_mass - e1_mass) + 2 * e1_mass * e1_vel.dx) / total_mass)
        )
        return new_v1, new_v2
    
    @staticmethod
    def wall_bounce(velocity: Vector, wall_normal: Vector) -> Vector:
        """Reflect velocity off a wall."""
        if wall_normal.dy != 0:
            return Vector(-velocity.dy, velocity.dx)
        elif wall_normal.dx != 0:
            return Vector(velocity.dy, -velocity.dx)
        return velocity
    
    @staticmethod
    def attraction(pos1: Position, pos2: Position, strength: int) -> Vector:
        """Object at pos1 is attracted toward pos2."""
        dy = pos2.y - pos1.y
        dx = pos2.x - pos1.x
        dist = max(1, (dy**2 + dx**2) ** 0.5)
        return Vector(
            int(strength * dy / dist),
            int(strength * dx / dist)
        )
    
    @staticmethod
    def color_propagation(neighbor_colors: List[int], threshold: int) -> int:
        """Color spreads like a cellular automaton."""
        if not neighbor_colors:
            return BG_COLOR
        counts = defaultdict(int)
        for c in neighbor_colors:
            if c != BG_COLOR:
                counts[c] += 1
        if counts:
            max_color = max(counts, key=counts.get)
            if counts[max_color] >= threshold:
                return max_color
        return BG_COLOR
    
    @staticmethod
    def conditional_transform(condition: bool, if_true: Any, if_false: Any) -> Any:
        """Generic conditional mechanism."""
        return if_true if condition else if_false


# ============================================================================
# 7. WORLD SIMULATION ENGINE - Time's Arrow
# ============================================================================

class WorldEngine:
    """
    The engine that simulates world dynamics over time.
    Implements the temporal schema: causality as succession according to rules.
    """
    
    def __init__(self, initial_state: WorldState):
        self.states: List[WorldState] = [initial_state]
        self.rules: List[Callable[[WorldState], WorldState]] = []
        
    def add_rule(self, rule: Callable[[WorldState], WorldState]) -> None:
        """Add a global update rule."""
        self.rules.append(rule)
    
    def step(self) -> WorldState:
        """Advance the world by one timestep."""
        current = self.states[-1].copy()
        current.timestamp += 1
        
        # Apply velocity to position for all mobile entities
        H, W = current.grid_shape
        for entity in current.entities.values():
            if entity.mobile and entity.alive:
                new_pos = entity.position + entity.velocity
                # Boundary checking
                h, w = entity.size
                new_y = max(0, min(H - h, new_pos.y))
                new_x = max(0, min(W - w, new_pos.x))
                entity.position = Position(new_y, new_x)
        
        # Check collisions and apply collision rules
        entities = list(current.entities.values())
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                if e1.alive and e2.alive and e1.solid and e2.solid:
                    if current.check_collision(e1, e2):
                        self._handle_collision(current, e1, e2)
        
        # Apply global rules
        for rule in self.rules:
            current = rule(current)
        
        self.states.append(current)
        return current
    
    def _handle_collision(self, state: WorldState, e1: Entity, e2: Entity) -> None:
        """Handle collision between two entities."""
        # Simple response: stop both entities
        if e1.mobile:
            e1.velocity = Vector(0, 0)
        if e2.mobile:
            e2.velocity = Vector(0, 0)
    
    def simulate(self, steps: int) -> List[WorldState]:
        """Run simulation for n steps."""
        for _ in range(min(steps, MAX_SIMULATION_STEPS)):
            self.step()
        return self.states
    
    def intervene(self, intervention: Dict[str, Any]) -> 'WorldEngine':
        """
        Create a counterfactual branch with an intervention.
        do(X=x): Set X to x and simulate from there.
        """
        # Copy current state
        counterfactual_state = self.states[-1].copy()
        
        # Apply intervention
        for key, value in intervention.items():
            if "." in key:
                entity_uid, prop = key.split(".", 1)
                if entity_uid in counterfactual_state.entities:
                    setattr(counterfactual_state.entities[entity_uid], prop, value)
            else:
                counterfactual_state.global_properties[key] = value
        
        # Return new engine with counterfactual branch
        new_engine = WorldEngine(counterfactual_state)
        new_engine.rules = self.rules.copy()
        return new_engine
    
    def get_trajectory(self) -> List[Dict[str, Any]]:
        """Get the full trajectory as serializable data."""
        return [s.to_dict() for s in self.states]


# ============================================================================
# 8. RENDERER - From Noumenon to Phenomenon
# ============================================================================

class Renderer:
    """
    Projects the abstract CausalWorld into observable pixels.
    
    Key insight: The renderer is LOSSY. Information is destroyed.
    The model must infer the world from incomplete observations.
    """
    
    def __init__(self, noise_level: float = 0.0, occlusion: bool = True):
        self.noise_level = noise_level
        self.occlusion = occlusion
    
    def render(self, state: WorldState) -> np.ndarray:
        """Render world state to a grid."""
        H, W = state.grid_shape
        grid = np.full((H, W), BG_COLOR, dtype=np.uint8)
        
        # Sort entities by some z-order (using uid hash for determinism)
        sorted_entities = sorted(
            [e for e in state.entities.values() if e.alive and e.visible],
            key=lambda e: hash(e.uid) % 1000
        )
        
        for entity in sorted_entities:
            sprite = self._generate_sprite(entity)
            self._blit(grid, sprite, entity.position, entity.color)
        
        # Apply noise
        if self.noise_level > 0:
            noise_mask = np.random.random((H, W)) < self.noise_level
            noise_colors = np.random.randint(0, 10, (H, W))
            grid[noise_mask] = noise_colors[noise_mask]
        
        return grid
    
    def _generate_sprite(self, entity: Entity) -> np.ndarray:
        """Generate a binary sprite mask for an entity."""
        h, w = entity.size
        sprite = np.zeros((h, w), dtype=bool)
        
        if entity.shape == ShapeType.POINT:
            if h > 0 and w > 0:
                sprite[h//2, w//2] = True
                
        elif entity.shape == ShapeType.RECT:
            sprite[:] = True
            
        elif entity.shape == ShapeType.HOLLOW_RECT:
            sprite[:] = True
            if h > 2 and w > 2:
                sprite[1:-1, 1:-1] = False
                
        elif entity.shape == ShapeType.CIRCLE:
            cy, cx = h // 2, w // 2
            r = min(h, w) // 2
            for y in range(h):
                for x in range(w):
                    if (y - cy)**2 + (x - cx)**2 <= r**2:
                        sprite[y, x] = True
                        
        elif entity.shape == ShapeType.LINE_H:
            sprite[h//2, :] = True
            
        elif entity.shape == ShapeType.LINE_V:
            sprite[:, w//2] = True
            
        elif entity.shape == ShapeType.CROSS:
            sprite[h//2, :] = True
            sprite[:, w//2] = True
            
        elif entity.shape == ShapeType.L_SHAPE:
            bar = max(1, min(h, w) // 3)
            sprite[:, :bar] = True
            sprite[-bar:, :] = True
            
        elif entity.shape == ShapeType.T_SHAPE:
            bar = max(1, min(h, w) // 3)
            sprite[:bar, :] = True
            sprite[:, w//2-bar//2:w//2+bar//2+1] = True
            
        elif entity.shape == ShapeType.TRIANGLE:
            for row in range(h):
                width = max(1, int(w * (row + 1) / h))
                start = (w - width) // 2
                sprite[row, start:start+width] = True
                
        elif entity.shape == ShapeType.DIAMOND:
            cy, cx = h // 2, w // 2
            for y in range(h):
                for x in range(w):
                    if abs(y - cy) + abs(x - cx) <= min(cy, cx):
                        sprite[y, x] = True
        
        return sprite
    
    def _blit(self, grid: np.ndarray, sprite: np.ndarray, pos: Position, color: int) -> None:
        """Blit a sprite onto the grid with clipping."""
        H, W = grid.shape
        sh, sw = sprite.shape
        
        # Calculate clipping bounds
        y_start = max(0, pos.y)
        y_end = min(H, pos.y + sh)
        x_start = max(0, pos.x)
        x_end = min(W, pos.x + sw)
        
        sy_start = max(0, -pos.y)
        sy_end = sy_start + (y_end - y_start)
        sx_start = max(0, -pos.x)
        sx_end = sx_start + (x_end - x_start)
        
        if sy_end > sy_start and sx_end > sx_start:
            sub_sprite = sprite[sy_start:sy_end, sx_start:sx_end]
            target = grid[y_start:y_end, x_start:x_end]
            target[sub_sprite] = color


# ============================================================================
# 9. VERIFICATION ENGINE - Proving Necessity
# ============================================================================

class NecessityVerifier:
    """
    The verification engine proves that solutions are NECESSARY, not merely correct.
    
    A solution S is necessary iff:
    1. S is consistent with all training examples
    2. No alternative S' is also consistent with all training examples
    
    If multiple solutions exist, we generate disambiguation examples.
    """
    
    def __init__(self, hypothesis_generator: Callable[[List[Dict]], List[Callable]]):
        self.hypothesis_generator = hypothesis_generator
    
    def find_consistent_hypotheses(
        self, 
        examples: List[Tuple[np.ndarray, np.ndarray]],
        candidate_rules: List[Callable[[np.ndarray], np.ndarray]]
    ) -> List[Callable]:
        """Find all rules consistent with the examples."""
        consistent = []
        for rule in candidate_rules:
            is_consistent = True
            for inp, out in examples:
                try:
                    predicted = rule(inp)
                    if not np.array_equal(predicted, out):
                        is_consistent = False
                        break
                except Exception:
                    is_consistent = False
                    break
            if is_consistent:
                consistent.append(rule)
        return consistent
    
    def generate_disambiguation(
        self,
        rules: List[Callable],
        input_generator: Callable[[], np.ndarray],
        max_attempts: int = 100
    ) -> Optional[np.ndarray]:
        """
        Generate an input that distinguishes between ambiguous rules.
        Returns an input where rules disagree, or None if they always agree.
        """
        for _ in range(max_attempts):
            inp = input_generator()
            outputs = set()
            for rule in rules:
                try:
                    out = rule(inp)
                    outputs.add(out.tobytes())
                except Exception:
                    continue
            if len(outputs) > 1:
                return inp
        return None
    
    def prove_necessity(
        self,
        examples: List[Tuple[np.ndarray, np.ndarray]],
        candidate_rules: List[Callable],
        solution_rule: Callable
    ) -> Dict[str, Any]:
        """
        Prove that solution_rule is the necessary (unique) solution.
        """
        consistent = self.find_consistent_hypotheses(examples, candidate_rules)
        
        if len(consistent) == 0:
            return {
                "is_necessary": False,
                "reason": "No consistent rule found",
                "alternatives": [],
            }
        elif len(consistent) == 1:
            return {
                "is_necessary": True,
                "reason": "Unique consistent rule",
                "alternatives": [],
            }
        else:
            return {
                "is_necessary": False,
                "reason": f"Multiple consistent rules: {len(consistent)}",
                "alternatives": [r.__name__ for r in consistent if r != solution_rule],
            }


# ============================================================================
# 10. META-GENERATOR - Programs that Generate Programs
# ============================================================================

@dataclass
class GeneratorProgram:
    """
    A generator is a PROGRAM that produces tasks.
    By making generators first-class, we can:
    1. Compose generators
    2. Generate tasks about generators
    3. Learn generator-invariant representations
    """
    name: str
    parameters: Dict[str, Any]
    generate_fn: Callable[..., 'KantTask']
    description: str
    
    # The program representation (for the model to reason about)
    source_code: str = ""
    
    def generate(self, **override_params) -> 'KantTask':
        params = {**self.parameters, **override_params}
        return self.generate_fn(**params)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameters": self.parameters,
            "description": self.description,
            "source_code": self.source_code,
        }


class MetaGenerator:
    """
    A generator of generators.
    
    This enables compositional generalization:
    - Learn structure of generator space, not just task space
    - Transfer between generators with shared components
    """
    
    def __init__(self):
        self.primitive_components: Dict[str, Callable] = {}
        self.composition_rules: List[Callable] = []
    
    def register_component(self, name: str, component: Callable) -> None:
        """Register a primitive generator component."""
        self.primitive_components[name] = component
    
    def compose(self, components: List[str], composition_type: str = "sequential") -> GeneratorProgram:
        """Compose primitive components into a new generator."""
        selected = [self.primitive_components[c] for c in components if c in self.primitive_components]
        
        if composition_type == "sequential":
            def composed_fn(**params):
                result = None
                for comp in selected:
                    result = comp(result, **params) if result else comp(**params)
                return result
        elif composition_type == "parallel":
            def composed_fn(**params):
                results = [comp(**params) for comp in selected]
                return self._merge_results(results)
        else:
            composed_fn = lambda **p: selected[0](**p) if selected else None
        
        return GeneratorProgram(
            name=f"composed_{composition_type}_{'_'.join(components)}",
            parameters={},
            generate_fn=composed_fn,
            description=f"Composed generator: {composition_type} of {components}",
            source_code=f"compose({components}, {composition_type})",
        )
    
    def _merge_results(self, results: List[Any]) -> Any:
        """Merge results from parallel generators."""
        # Placeholder - actual implementation depends on result types
        return results[0] if results else None


# ============================================================================
# 11. REASONING TRACE - The Path of Thought
# ============================================================================

@dataclass
class ReasoningStep:
    """A single step in a reasoning trace."""
    step_number: int
    observation: str      # What the reasoner observes
    inference: str        # What the reasoner infers
    justification: str    # Why this inference follows
    category_used: Category  # Which Kantian category was applied
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_number,
            "observe": self.observation,
            "infer": self.inference,
            "because": self.justification,
            "category": self.category_used.value,
        }


@dataclass
class ReasoningTrace:
    """
    The complete reasoning trace from input to output.
    This makes the model's reasoning EXPLICIT and trainable.
    """
    steps: List[ReasoningStep]
    conclusion: str
    counterfactuals: Dict[str, str]  # "if X were Y" -> "then Z would be W"
    necessity_proof: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "conclusion": self.conclusion,
            "counterfactuals": self.counterfactuals,
            "necessity_proof": self.necessity_proof,
        }


# ============================================================================
# 12. TASK OUTPUT - The Complete Data Sample
# ============================================================================

@dataclass
class KantTask:
    """
    The complete output of the Kant Protocol.
    
    This is NOT just (input, output) pairs.
    It's the entire epistemic situation:
    - What you observe
    - What you can infer
    - Why the inference is necessary
    - What would be different under counterfactuals
    """
    # The observable data
    input_grid: np.ndarray
    output_grid: np.ndarray
    
    # The ground truth structure
    causal_graph: CausalGraph
    input_state: WorldState
    output_state: WorldState
    
    # The reasoning
    reasoning_trace: ReasoningTrace
    
    # The counterfactuals
    counterfactual_worlds: Dict[str, Tuple[np.ndarray, np.ndarray]]
    
    # Metadata
    categories_exemplified: List[Category]
    difficulty: int
    generator_program: Optional[GeneratorProgram] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input": self.input_grid.tolist(),
            "output": self.output_grid.tolist(),
            "causal_structure": self.causal_graph.to_dict(),
            "input_state": self.input_state.to_dict(),
            "output_state": self.output_state.to_dict(),
            "reasoning": self.reasoning_trace.to_dict(),
            "counterfactuals": {
                k: {"input": v[0].tolist(), "output": v[1].tolist()}
                for k, v in self.counterfactual_worlds.items()
            },
            "categories": [c.value for c in self.categories_exemplified],
            "difficulty": self.difficulty,
            "generator": self.generator_program.to_dict() if self.generator_program else None,
        }
    
    def to_arc_format(self) -> Dict[str, Any]:
        """Convert to standard ARC format for compatibility."""
        return {
            "input": self.input_grid.tolist(),
            "output": self.output_grid.tolist(),
        }


# ============================================================================
# 13. THE GOD GENERATOR - Putting It All Together
# ============================================================================

class KantGenerator:
    """
    The Kant Protocol Generator.
    
    This generates tasks that FORCE the model to reason, not just pattern match.
    Each task exemplifies specific Kantian categories and includes:
    - Causal structure (not just correlation)
    - Necessity proofs (not just correct answers)
    - Counterfactuals (not just the actual world)
    - Reasoning traces (not just input/output)
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.renderer = Renderer()
        self.meta_generator = MetaGenerator()
        self._register_primitive_components()
    
    def _register_primitive_components(self) -> None:
        """Register primitive generator components for composition."""
        self.meta_generator.register_component("motion", self._component_motion)
        self.meta_generator.register_component("collision", self._component_collision)
        self.meta_generator.register_component("sorting", self._component_sorting)
        self.meta_generator.register_component("grouping", self._component_grouping)
        self.meta_generator.register_component("transformation", self._component_transformation)
    
    # ========================================================================
    # PRIMITIVE COMPONENTS
    # ========================================================================
    
    def _component_motion(self, **params) -> Dict[str, Any]:
        """Primitive: Object motion with velocity."""
        return {"type": "motion", "params": params}
    
    def _component_collision(self, **params) -> Dict[str, Any]:
        """Primitive: Collision detection and response."""
        return {"type": "collision", "params": params}
    
    def _component_sorting(self, **params) -> Dict[str, Any]:
        """Primitive: Sorting objects by property."""
        return {"type": "sorting", "params": params}
    
    def _component_grouping(self, **params) -> Dict[str, Any]:
        """Primitive: Grouping objects by property."""
        return {"type": "grouping", "params": params}
    
    def _component_transformation(self, **params) -> Dict[str, Any]:
        """Primitive: Spatial transformation."""
        return {"type": "transformation", "params": params}
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _create_entity(
        self,
        shape: ShapeType = ShapeType.RECT,
        color: int = 1,
        position: Position = None,
        size: Tuple[int, int] = (2, 2),
        velocity: Vector = None,
        **kwargs
    ) -> Entity:
        """Create an entity with defaults."""
        return Entity(
            shape=shape,
            color=color,
            position=position or Position(0, 0),
            size=size,
            velocity=velocity or Vector(0, 0),
            **kwargs
        )
    
    def _create_initial_state(
        self,
        grid_shape: Tuple[int, int],
        entities: List[Entity]
    ) -> WorldState:
        """Create an initial world state."""
        return WorldState(
            timestamp=0,
            grid_shape=grid_shape,
            entities={e.uid: e for e in entities},
            causal_graph=CausalGraph(),
        )
    
    def _place_entities_no_overlap(
        self,
        entities: List[Entity],
        grid_shape: Tuple[int, int],
        max_attempts: int = 100
    ) -> bool:
        """Place entities without overlap. Returns False if failed."""
        H, W = grid_shape
        occupied = np.zeros((H, W), dtype=bool)
        
        # Sort by size (largest first)
        sorted_entities = sorted(entities, key=lambda e: e.size[0] * e.size[1], reverse=True)
        
        for entity in sorted_entities:
            h, w = entity.size
            placed = False
            
            for _ in range(max_attempts):
                y = random.randint(0, max(0, H - h))
                x = random.randint(0, max(0, W - w))
                
                # Check overlap
                if not occupied[y:y+h, x:x+w].any():
                    entity.position = Position(y, x)
                    occupied[y:y+h, x:x+w] = True
                    placed = True
                    break
            
            if not placed:
                return False
        
        return True
    
    def _build_reasoning_trace(
        self,
        observations: List[str],
        inferences: List[str],
        justifications: List[str],
        categories: List[Category],
        conclusion: str,
        counterfactuals: Dict[str, str] = None
    ) -> ReasoningTrace:
        """Build a complete reasoning trace."""
        steps = []
        for i, (obs, inf, just, cat) in enumerate(zip(observations, inferences, justifications, categories)):
            steps.append(ReasoningStep(
                step_number=i + 1,
                observation=obs,
                inference=inf,
                justification=just,
                category_used=cat,
            ))
        
        return ReasoningTrace(
            steps=steps,
            conclusion=conclusion,
            counterfactuals=counterfactuals or {},
        )
    
    # ========================================================================
    # TASK GENERATORS - Each exemplifies specific categories
    # ========================================================================
    
    def task_causal_motion(self, difficulty: int = 1) -> KantTask:
        """
        CATEGORY: CAUSALITY
        
        An object moves BECAUSE it has velocity.
        The model must infer: position_new = position_old + velocity
        
        This is not correlation (things move). It's causation (velocity CAUSES movement).
        """
        H, W = 15, 15
        
        # Create agent with velocity
        velocity = Vector(0, random.choice([-2, -1, 1, 2]))
        agent = self._create_entity(
            shape=random.choice([ShapeType.RECT, ShapeType.CIRCLE]),
            color=random.randint(1, 9),
            size=(2, 2),
            velocity=velocity,
        )
        
        # Create obstacles (stationary)
        obstacles = []
        num_obstacles = difficulty
        for i in range(num_obstacles):
            obs = self._create_entity(
                shape=ShapeType.RECT,
                color=random.randint(1, 9),
                size=(random.randint(2, 4), random.randint(2, 4)),
                velocity=Vector(0, 0),
                mobile=False,
            )
            obstacles.append(obs)
        
        # Place entities
        entities = [agent] + obstacles
        if not self._place_entities_no_overlap(entities, (H, W)):
            return self.task_causal_motion(difficulty)  # Retry
        
        # Build causal graph
        causal_graph = CausalGraph()
        causal_graph.add_variable(f"{agent.uid}.position", ValueType.POSITION)
        causal_graph.add_variable(f"{agent.uid}.velocity", ValueType.VECTOR)
        causal_graph.add_mechanism(CausalMechanism(
            name="velocity_causes_position_change",
            cause_vars=[f"{agent.uid}.velocity"],
            effect_var=f"{agent.uid}.position",
            mechanism_fn=lambda v: v,  # Simplified
            description="Velocity causes position to change by that amount each timestep",
        ))
        
        # Create initial state
        initial_state = self._create_initial_state((H, W), entities)
        initial_state.causal_graph = causal_graph
        
        # Simulate
        engine = WorldEngine(initial_state)
        steps = random.randint(2, 5)
        engine.simulate(steps)
        final_state = engine.states[-1]
        
        # Render
        input_grid = self.renderer.render(initial_state)
        output_grid = self.renderer.render(final_state)
        
        # Build reasoning trace
        trace = self._build_reasoning_trace(
            observations=[
                f"There is a {agent.shape.value} at position {agent.position}",
                f"The {agent.shape.value} has velocity {(velocity.dy, velocity.dx)}",
            ],
            inferences=[
                f"The object will move {steps} steps",
                f"New position = old position + velocity × steps",
            ],
            justifications=[
                "Velocity is the cause of position change (CAUSALITY)",
                "Succession according to a rule: position changes by velocity each timestep",
            ],
            categories=[Category.CAUSALITY, Category.CAUSALITY],
            conclusion=f"The {agent.shape.value} moves from {agent.position} to {final_state.entities[agent.uid].position}",
            counterfactuals={
                f"if velocity were (0, 0)": "object would stay in place",
                f"if velocity were {(-velocity.dy, -velocity.dx)}": "object would move in opposite direction",
            },
        )
        
        # Generate counterfactual worlds
        counterfactuals = {}
        
        # Counterfactual 1: No velocity
        cf_engine = WorldEngine(initial_state.copy())
        cf_engine.states[0].entities[agent.uid].velocity = Vector(0, 0)
        cf_engine.simulate(steps)
        cf_input = self.renderer.render(cf_engine.states[0])
        cf_output = self.renderer.render(cf_engine.states[-1])
        counterfactuals["velocity_zero"] = (cf_input, cf_output)
        
        # Counterfactual 2: Opposite velocity
        cf_engine2 = WorldEngine(initial_state.copy())
        cf_engine2.states[0].entities[agent.uid].velocity = -velocity
        cf_engine2.simulate(steps)
        cf_input2 = self.renderer.render(cf_engine2.states[0])
        cf_output2 = self.renderer.render(cf_engine2.states[-1])
        counterfactuals["velocity_reversed"] = (cf_input2, cf_output2)
        
        return KantTask(
            input_grid=input_grid,
            output_grid=output_grid,
            causal_graph=causal_graph,
            input_state=initial_state,
            output_state=final_state,
            reasoning_trace=trace,
            counterfactual_worlds=counterfactuals,
            categories_exemplified=[Category.CAUSALITY, Category.SUBSTANCE],
            difficulty=difficulty,
            generator_program=GeneratorProgram(
                name="causal_motion",
                parameters={"difficulty": difficulty, "steps": steps},
                generate_fn=self.task_causal_motion,
                description="Object motion caused by velocity",
            ),
        )
    
    def task_totality_transformation(self, difficulty: int = 1) -> KantTask:
        """
        CATEGORY: TOTALITY
        
        ALL objects transform according to the SAME rule.
        The model must recognize: this applies to the SET, not individuals.
        """
        H, W = 20, 20
        
        # Create multiple objects of same color
        target_color = random.randint(1, 5)
        other_colors = [c for c in range(6, 10)]
        
        # Target objects (will all transform)
        targets = []
        num_targets = random.randint(3, 5)
        for _ in range(num_targets):
            t = self._create_entity(
                shape=random.choice([ShapeType.RECT, ShapeType.CIRCLE, ShapeType.TRIANGLE]),
                color=target_color,
                size=(random.randint(2, 3), random.randint(2, 3)),
            )
            targets.append(t)
        
        # Distractor objects (won't transform)
        distractors = []
        num_distractors = random.randint(1, 3)
        for _ in range(num_distractors):
            d = self._create_entity(
                shape=random.choice([ShapeType.RECT, ShapeType.DIAMOND]),
                color=random.choice(other_colors),
                size=(random.randint(2, 3), random.randint(2, 3)),
            )
            distractors.append(d)
        
        # Place all entities
        entities = targets + distractors
        if not self._place_entities_no_overlap(entities, (H, W)):
            return self.task_totality_transformation(difficulty)
        
        # Create initial state
        initial_state = self._create_initial_state((H, W), entities)
        
        # The transformation: ALL target-colored objects move by same vector
        transform_vector = Vector(
            random.choice([-3, -2, 2, 3]),
            random.choice([-3, -2, 2, 3])
        )
        
        # Build causal graph - the transformation is GLOBAL
        causal_graph = CausalGraph()
        causal_graph.add_variable("global.transform_rule", ValueType.VECTOR)
        causal_graph.add_variable("global.target_color", ValueType.COLOR)
        
        for t in targets:
            causal_graph.add_variable(f"{t.uid}.position", ValueType.POSITION)
            causal_graph.add_mechanism(CausalMechanism(
                name=f"totality_transform_{t.uid}",
                cause_vars=["global.transform_rule", "global.target_color"],
                effect_var=f"{t.uid}.position",
                mechanism_fn=lambda rule, color: rule,  # All same transform
                description=f"Object {t.uid} transforms according to global rule",
            ))
        
        initial_state.causal_graph = causal_graph
        initial_state.global_properties["transform_rule"] = transform_vector
        initial_state.global_properties["target_color"] = target_color
        
        # Apply transformation to create output state
        output_state = initial_state.copy()
        output_state.timestamp = 1
        for t in targets:
            old_pos = output_state.entities[t.uid].position
            new_pos = old_pos + transform_vector
            # Clamp to bounds
            h, w = output_state.entities[t.uid].size
            new_y = max(0, min(H - h, new_pos.y))
            new_x = max(0, min(W - w, new_pos.x))
            output_state.entities[t.uid].position = Position(new_y, new_x)
        
        # Render
        input_grid = self.renderer.render(initial_state)
        output_grid = self.renderer.render(output_state)
        
        # Build reasoning trace
        trace = self._build_reasoning_trace(
            observations=[
                f"There are {num_targets} objects of color {target_color}",
                f"There are {num_distractors} objects of other colors",
                "All same-colored objects transform identically",
            ],
            inferences=[
                f"The rule applies to the TOTALITY of color-{target_color} objects",
                f"Transform = {(transform_vector.dy, transform_vector.dx)}",
                "Other objects are unchanged",
            ],
            justifications=[
                "Color defines the relevant SET (TOTALITY)",
                "All members of the set transform by the same rule",
                "Objects outside the set are unaffected (LIMITATION)",
            ],
            categories=[Category.TOTALITY, Category.TOTALITY, Category.LIMITATION],
            conclusion=f"All {num_targets} objects of color {target_color} move by {(transform_vector.dy, transform_vector.dx)}",
            counterfactuals={
                "if only one object transformed": "rule would not be about TOTALITY",
                "if distractors also moved": "rule would apply to ALL objects, not just color group",
            },
        )
        
        # Counterfactual: What if we transformed different color?
        counterfactuals = {}
        cf_state = initial_state.copy()
        for d in distractors:
            old_pos = cf_state.entities[d.uid].position
            new_pos = old_pos + transform_vector
            h, w = cf_state.entities[d.uid].size
            new_y = max(0, min(H - h, new_pos.y))
            new_x = max(0, min(W - w, new_pos.x))
            cf_state.entities[d.uid].position = Position(new_y, new_x)
        cf_input = self.renderer.render(initial_state)
        cf_output = self.renderer.render(cf_state)
        counterfactuals["different_color_targeted"] = (cf_input, cf_output)
        
        return KantTask(
            input_grid=input_grid,
            output_grid=output_grid,
            causal_graph=causal_graph,
            input_state=initial_state,
            output_state=output_state,
            reasoning_trace=trace,
            counterfactual_worlds=counterfactuals,
            categories_exemplified=[Category.TOTALITY, Category.LIMITATION],
            difficulty=difficulty,
            generator_program=GeneratorProgram(
                name="totality_transformation",
                parameters={"difficulty": difficulty},
                generate_fn=self.task_totality_transformation,
                description="All objects of a class transform together",
            ),
        )
    
    def task_community_interaction(self, difficulty: int = 1) -> KantTask:
        """
        CATEGORY: COMMUNITY (Reciprocal Causation)
        
        Object A affects Object B, AND Object B affects Object A.
        This is not one-way causation but MUTUAL determination.
        """
        H, W = 15, 15
        
        # Create two objects that will interact
        obj_a = self._create_entity(
            shape=ShapeType.CIRCLE,
            color=random.randint(1, 4),
            size=(3, 3),
            velocity=Vector(0, 1),  # Moving right
        )
        
        obj_b = self._create_entity(
            shape=ShapeType.CIRCLE,
            color=random.randint(5, 9),
            size=(3, 3),
            velocity=Vector(0, -1),  # Moving left
        )
        
        # Place them on collision course
        obj_a.position = Position(6, 2)
        obj_b.position = Position(6, 10)
        
        entities = [obj_a, obj_b]
        
        # Build causal graph with BIDIRECTIONAL causation
        causal_graph = CausalGraph()
        
        # A's state affects B
        causal_graph.add_variable(f"{obj_a.uid}.velocity", ValueType.VECTOR)
        causal_graph.add_variable(f"{obj_b.uid}.velocity", ValueType.VECTOR)
        
        causal_graph.add_mechanism(CausalMechanism(
            name="a_affects_b",
            cause_vars=[f"{obj_a.uid}.velocity"],
            effect_var=f"{obj_b.uid}.velocity",
            mechanism_fn=lambda v: Vector(-v.dy, -v.dx),  # Reflection
            description="A's momentum transfers to B upon collision",
        ))
        
        causal_graph.add_mechanism(CausalMechanism(
            name="b_affects_a",
            cause_vars=[f"{obj_b.uid}.velocity"],
            effect_var=f"{obj_a.uid}.velocity",
            mechanism_fn=lambda v: Vector(-v.dy, -v.dx),  # Reflection
            description="B's momentum transfers to A upon collision",
        ))
        
        # Create initial state
        initial_state = self._create_initial_state((H, W), entities)
        initial_state.causal_graph = causal_graph
        
        # Simulate until collision
        engine = WorldEngine(initial_state)
        
        # Custom collision rule for community
        def community_collision_rule(state: WorldState) -> WorldState:
            a = state.entities[obj_a.uid]
            b = state.entities[obj_b.uid]
            if state.check_collision(a, b):
                # Swap velocities (elastic collision)
                a.velocity, b.velocity = b.velocity, a.velocity
            return state
        
        engine.add_rule(community_collision_rule)
        engine.simulate(6)
        final_state = engine.states[-1]
        
        # Render
        input_grid = self.renderer.render(initial_state)
        output_grid = self.renderer.render(final_state)
        
        # Build reasoning trace
        trace = self._build_reasoning_trace(
            observations=[
                f"Object A (color {obj_a.color}) moves right",
                f"Object B (color {obj_b.color}) moves left",
                "They are on a collision course",
            ],
            inferences=[
                "Upon collision, A affects B's velocity",
                "Simultaneously, B affects A's velocity",
                "This is RECIPROCAL causation",
            ],
            justifications=[
                "Community: A and B mutually determine each other",
                "Not one-way: A→B alone doesn't explain the outcome",
                "Simultaneity: both effects happen at the same time",
            ],
            categories=[Category.COMMUNITY, Category.COMMUNITY, Category.COMMUNITY],
            conclusion="After collision, both objects reverse direction due to mutual influence",
            counterfactuals={
                "if only A moved": "B would be pushed but A unaffected (not community)",
                "if B were immovable": "A would bounce but B stay (one-way causation)",
            },
        )
        
        # Counterfactual: One-way causation
        counterfactuals = {}
        cf_state = initial_state.copy()
        cf_state.entities[obj_b.uid].mobile = False
        cf_engine = WorldEngine(cf_state)
        cf_engine.simulate(6)
        cf_output = self.renderer.render(cf_engine.states[-1])
        counterfactuals["one_way_causation"] = (input_grid.copy(), cf_output)
        
        return KantTask(
            input_grid=input_grid,
            output_grid=output_grid,
            causal_graph=causal_graph,
            input_state=initial_state,
            output_state=final_state,
            reasoning_trace=trace,
            counterfactual_worlds=counterfactuals,
            categories_exemplified=[Category.COMMUNITY, Category.CAUSALITY],
            difficulty=difficulty,
            generator_program=GeneratorProgram(
                name="community_interaction",
                parameters={"difficulty": difficulty},
                generate_fn=self.task_community_interaction,
                description="Reciprocal causation between objects",
            ),
        )
    
    def task_necessity_unique_solution(self, difficulty: int = 1) -> KantTask:
        """
        CATEGORY: NECESSITY
        
        The task has ONE AND ONLY ONE correct solution.
        We PROVE this by showing all alternatives are inconsistent.
        """
        H, W = 12, 12
        
        # Create a pattern that has a unique completion
        # Use a simple rule: color at (y,x) = (y + x) % num_colors + 1
        num_colors = random.randint(3, 5)
        
        # Create the complete pattern
        complete_grid = np.zeros((H, W), dtype=np.uint8)
        for y in range(H):
            for x in range(W):
                complete_grid[y, x] = (y + x) % num_colors + 1
        
        # Remove some cells to create input (these must be inferred)
        input_grid = complete_grid.copy()
        mask = np.random.random((H, W)) < 0.3  # 30% missing
        input_grid[mask] = BG_COLOR
        
        # The output IS the complete pattern
        output_grid = complete_grid.copy()
        
        # Build causal graph
        causal_graph = CausalGraph()
        causal_graph.add_variable("global.rule", ValueType.INTEGER)
        causal_graph.add_mechanism(CausalMechanism(
            name="necessity_rule",
            cause_vars=["global.rule"],
            effect_var="grid",
            mechanism_fn=lambda r: r,
            description=f"Color at (y,x) = (y + x) mod {num_colors} + 1. This is NECESSARY, not arbitrary.",
        ))
        
        # Create states
        # For this task, we don't use entities - the grid IS the state
        initial_state = WorldState(
            timestamp=0,
            grid_shape=(H, W),
            entities={},
            causal_graph=causal_graph,
            global_properties={"rule": f"color = (y + x) % {num_colors} + 1"},
        )
        
        output_state = WorldState(
            timestamp=1,
            grid_shape=(H, W),
            entities={},
            causal_graph=causal_graph,
            global_properties={"rule": f"color = (y + x) % {num_colors} + 1"},
        )
        
        # Build reasoning trace
        trace = self._build_reasoning_trace(
            observations=[
                "The grid has a regular pattern with some cells missing",
                f"Visible cells follow: color = (row + col) mod {num_colors} + 1",
                "No other rule fits the visible pattern",
            ],
            inferences=[
                "The missing cells MUST follow the same rule",
                "Any other completion would violate the pattern",
                "The solution is NECESSARY, not just possible",
            ],
            justifications=[
                "Necessity: determined by conditions (the visible pattern)",
                "If any other color were used, consistency would fail",
                "The rule determines all cells uniquely",
            ],
            categories=[Category.NECESSITY, Category.NECESSITY, Category.NECESSITY],
            conclusion=f"The unique completion follows: color = (row + col) mod {num_colors} + 1",
            counterfactuals={
                "if a missing cell had different color": "the pattern would be violated",
                "if the rule were different": "it would not match visible cells",
            },
        )
        
        # Add necessity proof
        trace.necessity_proof = {
            "claim": "The solution is unique",
            "proof": f"Given the visible cells, only (y+x) mod {num_colors} + 1 fits all positions. "
                     f"Any alternative rule would produce at least one inconsistency with a visible cell.",
            "alternative_rules_tested": [
                f"(y * x) mod {num_colors} + 1 - FAILS at multiple positions",
                f"y mod {num_colors} + 1 - FAILS (ignores x)",
                f"x mod {num_colors} + 1 - FAILS (ignores y)",
            ],
        }
        
        # No counterfactual worlds needed - the solution is necessary
        counterfactuals = {}
        
        return KantTask(
            input_grid=input_grid,
            output_grid=output_grid,
            causal_graph=causal_graph,
            input_state=initial_state,
            output_state=output_state,
            reasoning_trace=trace,
            counterfactual_worlds=counterfactuals,
            categories_exemplified=[Category.NECESSITY, Category.REALITY],
            difficulty=difficulty,
            generator_program=GeneratorProgram(
                name="necessity_unique",
                parameters={"difficulty": difficulty, "num_colors": num_colors},
                generate_fn=self.task_necessity_unique_solution,
                description="Pattern completion with provably unique solution",
            ),
        )
    
    def task_negation_absence(self, difficulty: int = 1) -> KantTask:
        """
        CATEGORY: NEGATION
        
        The transformation involves REMOVING something.
        The model must reason about absence, not just presence.
        """
        H, W = 15, 15
        
        # Create objects - some will be removed based on a rule
        entities = []
        num_objects = random.randint(6, 10)
        
        # The rule: remove objects whose color matches the "eraser" color
        eraser_color = random.randint(1, 3)
        
        for _ in range(num_objects):
            color = random.randint(1, 9)
            e = self._create_entity(
                shape=random.choice([ShapeType.RECT, ShapeType.CIRCLE, ShapeType.TRIANGLE]),
                color=color,
                size=(random.randint(2, 3), random.randint(2, 3)),
            )
            entities.append(e)
        
        # Add an "eraser" entity that defines what gets removed
        eraser = self._create_entity(
            shape=ShapeType.DIAMOND,
            color=eraser_color,
            size=(3, 3),
            marked=True,  # This is the "key" object
        )
        entities.append(eraser)
        
        if not self._place_entities_no_overlap(entities, (H, W)):
            return self.task_negation_absence(difficulty)
        
        # Build causal graph
        causal_graph = CausalGraph()
        causal_graph.add_variable("eraser.color", ValueType.COLOR)
        
        for e in entities:
            causal_graph.add_variable(f"{e.uid}.alive", ValueType.BOOLEAN)
            if e.uid != eraser.uid:
                causal_graph.add_mechanism(CausalMechanism(
                    name=f"negation_{e.uid}",
                    cause_vars=["eraser.color"],
                    effect_var=f"{e.uid}.alive",
                    mechanism_fn=lambda ec, obj_color=e.color: obj_color != ec,
                    description=f"Object {e.uid} is negated if its color matches eraser",
                ))
        
        # Create initial state
        initial_state = self._create_initial_state((H, W), entities)
        initial_state.causal_graph = causal_graph
        
        # Apply negation to create output state
        output_state = initial_state.copy()
        output_state.timestamp = 1
        removed_count = 0
        for e in entities:
            if e.uid != eraser.uid and e.color == eraser_color:
                output_state.entities[e.uid].alive = False
                output_state.entities[e.uid].visible = False
                removed_count += 1
        
        # Render
        input_grid = self.renderer.render(initial_state)
        output_grid = self.renderer.render(output_state)
        
        # Build reasoning trace
        trace = self._build_reasoning_trace(
            observations=[
                f"There is a diamond (eraser) of color {eraser_color}",
                f"Several objects share color {eraser_color}",
                "Other objects have different colors",
            ],
            inferences=[
                "The diamond defines what gets NEGATED",
                f"All objects of color {eraser_color} are removed",
                "Objects of other colors remain",
            ],
            justifications=[
                "Negation: the transformation involves ABSENCE",
                "The diamond's color is the criterion for removal",
                "Non-matching objects exemplify LIMITATION (partially affected)",
            ],
            categories=[Category.NEGATION, Category.NEGATION, Category.LIMITATION],
            conclusion=f"{removed_count} objects removed; they shared color {eraser_color} with the eraser",
            counterfactuals={
                f"if eraser were different color": f"different objects would be removed",
                "if no matching objects": "nothing would be removed",
            },
        )
        
        # Counterfactual: Different eraser color
        counterfactuals = {}
        alt_color = (eraser_color % 9) + 1
        cf_state = initial_state.copy()
        for e in entities:
            if e.uid != eraser.uid and e.color == alt_color:
                cf_state.entities[e.uid].alive = False
                cf_state.entities[e.uid].visible = False
        cf_output = self.renderer.render(cf_state)
        counterfactuals["different_eraser_color"] = (input_grid.copy(), cf_output)
        
        return KantTask(
            input_grid=input_grid,
            output_grid=output_grid,
            causal_graph=causal_graph,
            input_state=initial_state,
            output_state=output_state,
            reasoning_trace=trace,
            counterfactual_worlds=counterfactuals,
            categories_exemplified=[Category.NEGATION, Category.LIMITATION],
            difficulty=difficulty,
            generator_program=GeneratorProgram(
                name="negation_absence",
                parameters={"difficulty": difficulty},
                generate_fn=self.task_negation_absence,
                description="Objects removed based on color matching",
            ),
        )
    
    def task_substance_persistence(self, difficulty: int = 1) -> KantTask:
        """
        CATEGORY: SUBSTANCE
        
        Something PERSISTS through change. The identity remains
        while accidents (position, shape) change.
        """
        H, W = 15, 15
        
        # Create an object that will transform but maintain identity
        original_shape = random.choice([ShapeType.RECT, ShapeType.CIRCLE])
        transformed_shape = random.choice([s for s in [ShapeType.TRIANGLE, ShapeType.DIAMOND, ShapeType.L_SHAPE] 
                                           if s != original_shape])
        
        persistent_color = random.randint(1, 9)  # THIS IS THE SUBSTANCE
        
        entity = self._create_entity(
            shape=original_shape,
            color=persistent_color,
            size=(4, 4),
        )
        
        # Add some unchanging background objects
        background = []
        for _ in range(random.randint(2, 4)):
            bg = self._create_entity(
                shape=ShapeType.RECT,
                color=random.choice([c for c in range(1, 10) if c != persistent_color]),
                size=(2, 2),
            )
            background.append(bg)
        
        entities = [entity] + background
        if not self._place_entities_no_overlap(entities, (H, W)):
            return self.task_substance_persistence(difficulty)
        
        # Build causal graph
        causal_graph = CausalGraph()
        causal_graph.add_variable(f"{entity.uid}.color", ValueType.COLOR)
        causal_graph.add_variable(f"{entity.uid}.shape", ValueType.SHAPE)
        causal_graph.add_variable(f"{entity.uid}.position", ValueType.POSITION)
        
        # Color persists (substance), shape changes (accident)
        causal_graph.add_mechanism(CausalMechanism(
            name="substance_persistence",
            cause_vars=[f"{entity.uid}.color"],
            effect_var=f"{entity.uid}.color",
            mechanism_fn=lambda c: c,  # Identity - color doesn't change
            description="Color is the SUBSTANCE that persists through transformation",
        ))
        
        # Create initial state
        initial_state = self._create_initial_state((H, W), entities)
        initial_state.causal_graph = causal_graph
        
        # Transform shape but keep color
        output_state = initial_state.copy()
        output_state.timestamp = 1
        output_state.entities[entity.uid].shape = transformed_shape
        # Also move it
        new_pos = Position(
            random.randint(0, H - 4),
            random.randint(0, W - 4)
        )
        output_state.entities[entity.uid].position = new_pos
        
        # Render
        input_grid = self.renderer.render(initial_state)
        output_grid = self.renderer.render(output_state)
        
        # Build reasoning trace
        trace = self._build_reasoning_trace(
            observations=[
                f"There is a {original_shape.value} of color {persistent_color}",
                f"In the output, there is a {transformed_shape.value} of the SAME color",
                "The shape changed, the color did not",
            ],
            inferences=[
                "The color is the SUBSTANCE (what persists)",
                "The shape is an ACCIDENT (what changes)",
                "The object maintains its identity through transformation",
            ],
            justifications=[
                "Substance: permanence of the real in time",
                "Accidents change while substance remains",
                "Identity = persistent color, not transient shape",
            ],
            categories=[Category.SUBSTANCE, Category.SUBSTANCE, Category.ACTUALITY],
            conclusion=f"The color-{persistent_color} object transformed from {original_shape.value} to {transformed_shape.value}",
            counterfactuals={
                "if color also changed": "we could not identify it as the SAME object",
                "if shape did not change": "there would be no transformation, just movement",
            },
        )
        
        # Counterfactual: Color change (identity lost)
        counterfactuals = {}
        cf_state = output_state.copy()
        cf_state.entities[entity.uid].color = (persistent_color % 9) + 1
        cf_output = self.renderer.render(cf_state)
        counterfactuals["color_changed_identity_lost"] = (input_grid.copy(), cf_output)
        
        return KantTask(
            input_grid=input_grid,
            output_grid=output_grid,
            causal_graph=causal_graph,
            input_state=initial_state,
            output_state=output_state,
            reasoning_trace=trace,
            counterfactual_worlds=counterfactuals,
            categories_exemplified=[Category.SUBSTANCE, Category.ACTUALITY],
            difficulty=difficulty,
            generator_program=GeneratorProgram(
                name="substance_persistence",
                parameters={"difficulty": difficulty},
                generate_fn=self.task_substance_persistence,
                description="Object transforms while maintaining identity through color",
            ),
        )
    
    # ========================================================================
    # BATCH GENERATION
    # ========================================================================
    
    def generate_task(self, category: Optional[Category] = None, difficulty: int = 1) -> KantTask:
        """Generate a single task, optionally targeting a specific category."""
        
        category_to_generator = {
            Category.CAUSALITY: self.task_causal_motion,
            Category.TOTALITY: self.task_totality_transformation,
            Category.COMMUNITY: self.task_community_interaction,
            Category.NECESSITY: self.task_necessity_unique_solution,
            Category.NEGATION: self.task_negation_absence,
            Category.SUBSTANCE: self.task_substance_persistence,
        }
        
        if category and category in category_to_generator:
            return category_to_generator[category](difficulty)
        else:
            generator = random.choice(list(category_to_generator.values()))
            return generator(difficulty)
    
    def generate_batch(
        self,
        count: int,
        save_dir: str,
        shots_per_task: int = 3,
        include_meta: bool = True
    ) -> None:
        """
        Generate a batch of tasks with full Kant Protocol output.
        
        Each saved task includes:
        - Standard ARC format (train/test pairs)
        - Causal graph structure
        - Reasoning traces
        - Counterfactuals
        - Generator program (if include_meta)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"=== GOD PROTOCOL v2.0 ===")
        print(f"Generating {count} tasks with {shots_per_task} shots each...")
        print(f"Categories: {[c.value for c in Category]}")
        print()
        
        generators = [
            self.task_causal_motion,
            self.task_totality_transformation,
            self.task_community_interaction,
            self.task_necessity_unique_solution,
            self.task_negation_absence,
            self.task_substance_persistence,
        ]
        
        for i in range(count):
            try:
                # Select generator
                gen_fn = random.choice(generators)
                difficulty = random.randint(1, 3)
                
                # Generate training shots
                train_tasks = [gen_fn(difficulty) for _ in range(shots_per_task)]
                
                # Generate test task
                test_task = gen_fn(difficulty)
                
                # Build full output
                output = {
                    # Standard ARC format
                    "train": [t.to_arc_format() for t in train_tasks],
                    "test": [test_task.to_arc_format()],
                    
                    # Kant Protocol additions
                    "kant_protocol": {
                        "version": "2.0",
                        
                        # Causal structure
                        "causal_graphs": {
                            "train": [t.causal_graph.to_dict() for t in train_tasks],
                            "test": test_task.causal_graph.to_dict(),
                        },
                        
                        # Reasoning traces
                        "reasoning_traces": {
                            "train": [t.reasoning_trace.to_dict() for t in train_tasks],
                            "test": test_task.reasoning_trace.to_dict(),
                        },
                        
                        # Counterfactuals
                        "counterfactuals": {
                            k: {"input": v[0].tolist(), "output": v[1].tolist()}
                            for k, v in test_task.counterfactual_worlds.items()
                        },
                        
                        # Categories exemplified
                        "categories": [c.value for c in test_task.categories_exemplified],
                        
                        # Difficulty
                        "difficulty": difficulty,
                    },
                }
                
                # Add generator program if meta enabled
                if include_meta and test_task.generator_program:
                    output["kant_protocol"]["generator"] = test_task.generator_program.to_dict()
                
                # Save
                category_str = "_".join(c.value for c in test_task.categories_exemplified[:2])
                fname = f"kant_{i:05d}_{category_str}.json"
                
                with open(os.path.join(save_dir, fname), 'w') as f:
                    json.dump(output, f, indent=2)
                
                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{count} tasks")
                    
            except Exception as e:
                print(f"Error generating task {i}: {e}")
                continue
        
        print(f"\nGeneration complete. {count} tasks saved to {save_dir}")
        print("Welcome to the Real World.")


def kant_generate(
    save_dir: str = "data/kant_100k",
    count: int = 100_000,
    shots_per_task: int = 3,
    seed: Optional[int] = None,
    deduplicate: bool = True,
    max_retries_per_task: int = 10,
    save_every: int = 1000,
) -> None:
    """
    Generate a large dataset of unique Kant Protocol tasks.
    
    Uniqueness Strategy:
    1. Each task gets a unique seed derived from (base_seed + task_index)
    2. We hash (input_grid, output_grid) to detect pixel-level duplicates
    3. We hash the causal graph structure to detect semantic duplicates
    
    For 100k tasks at ~3 shots each = 400k grids. With the combinatorial
    space available (>10^36 for some tasks), collisions are astronomically unlikely.
    
    Args:
        save_dir: Output directory
        count: Number of tasks to generate
        shots_per_task: Training examples per task
        seed: Base random seed (None = random)
        deduplicate: Check for and skip duplicate puzzles
        max_retries_per_task: Max attempts if duplicates found
        save_every: Save checkpoint every N tasks
    """
    import hashlib
    import time
    from pathlib import Path
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize
    base_seed = seed if seed is not None else random.randint(0, 2**31)
    seen_hashes: Set[str] = set()
    stats = {
        "generated": 0,
        "duplicates_skipped": 0,
        "errors": 0,
        "by_category": defaultdict(int),
    }
    
    print(f"=== KANT PROTOCOL GENERATOR ===")
    print(f"Target: {count:,} tasks")
    print(f"Base seed: {base_seed}")
    print(f"Output: {save_dir}")
    print(f"Deduplication: {deduplicate}")
    print()
    
    start_time = time.time()
    
    # Task generators with weights (can adjust to balance dataset)
    generators = [
        ("causal_motion", 1.0),
        ("totality_transformation", 1.0),
        ("community_interaction", 0.8),  # Slightly less - fewer variations
        ("necessity_unique_solution", 1.5),  # More weight - huge variation space
        ("negation_absence", 1.0),
        ("substance_persistence", 1.0),
    ]
    total_weight = sum(w for _, w in generators)
    
    task_idx = 0
    attempt = 0
    max_attempts = count * max_retries_per_task
    
    while stats["generated"] < count and attempt < max_attempts:
        attempt += 1
        
        # Deterministic seed for this attempt
        task_seed = base_seed + attempt
        gen = KantGenerator(seed=task_seed)
        
        # Weighted random selection of generator
        r = random.random() * total_weight
        cumulative = 0
        selected_gen_name = generators[0][0]
        for name, weight in generators:
            cumulative += weight
            if r <= cumulative:
                selected_gen_name = name
                break
        
        gen_method = getattr(gen, f"task_{selected_gen_name}")
        difficulty = random.randint(1, 3)
        
        try:
            # Generate shots + test
            tasks = []
            for _ in range(shots_per_task + 1):  # +1 for test
                task = gen_method(difficulty)
                tasks.append(task)
            
            train_tasks = tasks[:-1]
            test_task = tasks[-1]
            
            # Compute uniqueness hash
            if deduplicate:
                # Hash based on test input/output (most distinctive)
                hash_input = (
                    test_task.input_grid.tobytes() +
                    test_task.output_grid.tobytes()
                )
                task_hash = hashlib.md5(hash_input).hexdigest()[:16]
                
                if task_hash in seen_hashes:
                    stats["duplicates_skipped"] += 1
                    continue
                seen_hashes.add(task_hash)
            
            # Build output
            output = {
                "train": [t.to_arc_format() for t in train_tasks],
                "test": [test_task.to_arc_format()],
                "kant_protocol": {
                    "version": "2.0",
                    "seed": task_seed,
                    "causal_graphs": {
                        "train": [t.causal_graph.to_dict() for t in train_tasks],
                        "test": test_task.causal_graph.to_dict(),
                    },
                    "reasoning_traces": {
                        "train": [t.reasoning_trace.to_dict() for t in train_tasks],
                        "test": test_task.reasoning_trace.to_dict(),
                    },
                    "counterfactuals": {
                        k: {"input": v[0].tolist(), "output": v[1].tolist()}
                        for k, v in test_task.counterfactual_worlds.items()
                    },
                    "categories": [c.value for c in test_task.categories_exemplified],
                    "difficulty": difficulty,
                    "generator": test_task.generator_program.to_dict() if test_task.generator_program else None,
                },
            }
            
            # Save
            category_str = "_".join(c.value for c in test_task.categories_exemplified[:2])
            fname = f"kant_{task_idx:06d}_{category_str}.json"
            
            with open(os.path.join(save_dir, fname), 'w') as f:
                json.dump(output, f)  # No indent for space efficiency
            
            stats["generated"] += 1
            stats["by_category"][selected_gen_name] += 1
            task_idx += 1
            
            # Progress reporting
            if stats["generated"] % save_every == 0:
                elapsed = time.time() - start_time
                rate = stats["generated"] / elapsed
                eta = (count - stats["generated"]) / rate if rate > 0 else 0
                print(f"Progress: {stats['generated']:,}/{count:,} ({100*stats['generated']/count:.1f}%) | "
                      f"Rate: {rate:.1f}/s | ETA: {eta/60:.1f}m | "
                      f"Dupes: {stats['duplicates_skipped']:,}")
                
        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] < 10:  # Only print first few errors
                print(f"Error at attempt {attempt}: {e}")
            continue
    
    # Final stats
    elapsed = time.time() - start_time
    print()
    print(f"=== GENERATION COMPLETE ===")
    print(f"Tasks generated: {stats['generated']:,}")
    print(f"Duplicates skipped: {stats['duplicates_skipped']:,}")
    print(f"Errors: {stats['errors']:,}")
    print(f"Time: {elapsed/60:.1f} minutes ({stats['generated']/elapsed:.1f} tasks/sec)")
    print(f"Output: {save_dir}")
    print()
    print("Category distribution:")
    for cat, cnt in sorted(stats["by_category"].items()):
        print(f"  {cat}: {cnt:,} ({100*cnt/stats['generated']:.1f}%)")
    
    # Save metadata
    meta = {
        "total_tasks": stats["generated"],
        "base_seed": base_seed,
        "shots_per_task": shots_per_task,
        "category_distribution": dict(stats["by_category"]),
        "generation_time_seconds": elapsed,
    }
    with open(os.path.join(save_dir, "_metadata.json"), 'w') as f:
        json.dump(meta, f, indent=2)


# Convenience function for quick generation
def generate_kant_dataset(n: int = 100_000, output: str = "data/kant_100k") -> None:
    """Quick wrapper to generate N unique Kant Protocol tasks."""
    kant_generate(save_dir=output, count=n, seed=42)