"""
Plan C custom Simple Spread environment — FIXED version.
============================================================================
Key fix:
  - Target landmark order in the observation is randomly shuffled every episode.
  - This prevents agents from learning a private fixed rule based on landmark index.
  - Communication becomes necessary for proper coordination.

Other features remain the same:
  - Fewer landmarks than agents
  - Obstacles
  - Partial observability (Ni)
============================================================================
"""
from __future__ import annotations
import numpy as np

from mpe2._mpe_utils.core import Agent, Landmark, World
from mpe2._mpe_utils.partial_observability import (
    padded_comms,
    padded_relative_positions,
)
from mpe2._mpe_utils.scenario import BaseScenario
from mpe2._mpe_utils.simple_env import SimpleEnv, make_env
from mpe2.simple_spread.simple_spread import ExtendedWorld
from gymnasium.utils import EzPickle
from pettingzoo.utils.conversions import parallel_wrapper_fn


class PlanCScenario(BaseScenario):
    """Simple Spread with randomised landmark order + obstacles."""

    OCCUPY_THRESHOLD = 0.1

    def __init__(self, num_landmarks=4, num_obstacles=2,
                 num_agent_neighbors=None, num_landmark_neighbors=None):
        self.num_landmarks = num_landmarks
        self.num_obstacles = num_obstacles
        self.num_agent_neighbors = num_agent_neighbors
        self.num_landmark_neighbors = num_landmark_neighbors
        self._landmark_perm = None   # will be set every reset

    def make_world(self, N: int = 6) -> ExtendedWorld:
        world = ExtendedWorld()
        world.dim_c = 2
        world.collaborative = True

        # agents
        world.agents = [Agent() for _ in range(N)]
        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.15

        # landmarks (targets + obstacles)
        self._n_targets = self.num_landmarks
        self._n_obstacles = self.num_obstacles
        total = self.num_landmarks + self.num_obstacles
        world.landmarks = [Landmark() for _ in range(total)]
        for i, lm in enumerate(world.landmarks):
            if i < self.num_landmarks:
                lm.name = f"landmark_{i}"
                lm.collide = False
                lm.movable = False
                lm.is_obstacle = False
            else:
                lm.name = f"obstacle_{i - self.num_landmarks}"
                lm.collide = True
                lm.movable = False
                lm.is_obstacle = True
                lm.size = 0.18
        return world

    def reset_world(self, world: ExtendedWorld, np_random) -> None:
        for agent in world.agents:
            agent.color = np.array([0.35, 0.35, 0.85])
        for lm in world.landmarks:
            if getattr(lm, "is_obstacle", False):
                lm.color = np.array([0.85, 0.25, 0.25])
            else:
                lm.color = np.array([0.25, 0.25, 0.25])

        # random agent positions
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # random landmark + obstacle positions
        for lm in world.landmarks:
            lm.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            lm.state.p_vel = np.zeros(world.dim_p)

        # === CRITICAL FIX: shuffle target landmark order every episode ===
        targets = [lm for lm in world.landmarks if not getattr(lm, "is_obstacle", False)]
        self._landmark_perm = np_random.permutation(len(targets))

    def is_collision(self, a, b) -> bool:
        delta = a.state.p_pos - b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta)))
        return dist < (a.size + b.size)

    def reward(self, agent: Agent, world: ExtendedWorld) -> float:
        rew = 0.0
        if agent.collide:
            for other in world.agents:
                if other is not agent and self.is_collision(other, agent):
                    rew -= 1.0
            for lm in world.landmarks:
                if getattr(lm, "is_obstacle", False) and self.is_collision(lm, agent):
                    rew -= 1.0
        return rew

    def global_reward(self, world: ExtendedWorld) -> float:
        rew = 0.0
        for lm in world.landmarks:
            if getattr(lm, "is_obstacle", False):
                continue
            dists = [
                np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                for a in world.agents
            ]
            rew -= min(dists)
        return rew

    def observation(self, agent: Agent, world: ExtendedWorld) -> np.ndarray:
        others = [o for o in world.agents if o is not agent]

        targets = [lm for lm in world.landmarks if not getattr(lm, "is_obstacle", False)]
        obstacles = [lm for lm in world.landmarks if getattr(lm, "is_obstacle", False)]

        # Apply the random permutation decided at reset
        if self._landmark_perm is not None:
            targets = [targets[i] for i in self._landmark_perm]

        # target landmark positions
        if self.num_landmark_neighbors is None:
            entity_pos = [e.state.p_pos - agent.state.p_pos for e in targets]
        else:
            entity_pos = padded_relative_positions(
                agent, targets, self.num_landmark_neighbors
            )

        # obstacles always fully visible
        obstacle_pos = [o.state.p_pos - agent.state.p_pos for o in obstacles]

        # other agents + communication
        if self.num_agent_neighbors is None:
            other_pos = [o.state.p_pos - agent.state.p_pos for o in others]
            comm = [o.state.c for o in others]
        else:
            other_pos = padded_relative_positions(
                agent, others, self.num_agent_neighbors
            )
            comm = padded_comms(agent, others, self.num_agent_neighbors, world.dim_c)

        return np.concatenate(
            [agent.state.p_vel] + [agent.state.p_pos]
            + entity_pos + obstacle_pos + other_pos + comm
        )


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N: int = 6,
        num_landmarks: int = 4,
        num_obstacles: int = 2,
        local_ratio: float = 0.5,
        max_cycles: int = 50,
        continuous_actions: bool = False,
        render_mode: str | None = None,
        num_agent_neighbors: int | None = None,
        num_landmark_neighbors: int | None = None,
    ) -> None:
        assert 0.0 <= local_ratio <= 1.0, "local_ratio must be between 0 and 1."
        EzPickle.__init__(
            self, N=N, num_landmarks=num_landmarks, num_obstacles=num_obstacles,
            local_ratio=local_ratio, max_cycles=max_cycles,
            continuous_actions=continuous_actions, render_mode=render_mode,
            num_agent_neighbors=num_agent_neighbors,
            num_landmark_neighbors=num_landmark_neighbors,
        )
        scenario = PlanCScenario(
            num_landmarks=num_landmarks, num_obstacles=num_obstacles,
            num_agent_neighbors=num_agent_neighbors,
            num_landmark_neighbors=num_landmark_neighbors,
        )
        world = scenario.make_world(N)
        SimpleEnv.__init__(
            self, scenario=scenario, world=world,
            render_mode=render_mode, max_cycles=max_cycles,
            continuous_actions=continuous_actions, local_ratio=local_ratio,
        )
        self.metadata["name"] = "simple_spread_planC"


env = make_env(raw_env)
planC_parallel_env = parallel_wrapper_fn(env)