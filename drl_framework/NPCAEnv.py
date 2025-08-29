"""
NPCAEnv (Step 1)
-----------------
A minimal Gymnasium environment that wraps your slot-based CSMA/CA + NPCA simulator
(random_access.py) to learn a binary decision *only when the agent STA is in PRIMARY_BACKOFF
and the **primary channel is busy due to OBSS**: stay on primary (freeze) vs switch to NPCA backoff.

Design goals for Step 1
- Keep the simulator logic intact. No edits to random_access.py are required.
- Use a small hack to honor the RL action: for action=0 (stay primary), we temporarily
  set `agent.npca_enabled=False` for that slot so `_handle_primary_backoff()` takes the
  PRIMARY_FROZEN branch; for action=1 (switch), we leave `npca_enabled=True` and the original
  NPCA path is taken.
- Provide a clean Gymnasium API: reset(), step(), observation_space, action_space.
- Return an observation at each *decision point* (the next time the agent meets the OBSS-on-primary condition).
- Basic reward shaping included (configurable): +1 for a successful transmission, -1 for a collided
  transmission, small time penalty per elapsed slot, small cost for switching.

Next steps (future):
- Replace the npca_enabled hack with a proper hook in STA._handle_primary_backoff.
- Expand the action space (e.g., choose NPCA max duration, defer, etc.).
- Add wrappers for vectorized envs and normalization.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Import your simulator primitives
from random_access import Channel, STA, Simulator, SLOTTIME, CONTENTION_WINDOW


# -----------------------------
# Config dataclass
# -----------------------------
@dataclass
class NPCAEnvConfig:
    num_slots: int = 50_000              # episode horizon in slots
    n_stas: int = 6                      # total STAs (including the agent)
    agent_sta_idx: int = 0               # index of the agent STA
    ppdu_duration: int = 30              # slots per PPDU (for all STAs in step 1)
    primary_channel_id: int = 0
    npca_channel_id: int = 1
    obss_rate_primary: float = 0.10      # OBSS generation probability per free slot on primary
    obss_rate_npca: float = 0.05         # OBSS generation probability per free slot on npca ch
    obss_duration_range: Tuple[int, int] = (20, 60)
    seed: Optional[int] = None

    # Reward shaping
    success_reward: float = 1.0
    collision_penalty: float = -1.0
    time_penalty: float = -0.001         # per elapsed slot until next decision
    switch_penalty: float = -0.01        # cost when action==1 (switch)

    # Normalization constants
    max_backoff: int = CONTENTION_WINDOW[-1]  # 1023
    max_ppdu_slots: int = 200            # for normalizing occupied/obss remains


# -----------------------------
# Environment
# -----------------------------
class NPCAEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, config: NPCAEnvConfig):
        super().__init__()
        self.cfg = config
        self.rng = np.random.default_rng(self.cfg.seed)

        # Action: 0 = stay on primary (freeze), 1 = switch to NPCA backoff
        self.action_space = spaces.Discrete(2)

        # Observation (dict, normalized where applicable)
        # Keep the style similar to your custom_env.py
        self.observation_space = spaces.Dict({
            "primary_backoff": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "cw_index": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "npca_enabled": spaces.Discrete(2),
            "primary_busy_intra": spaces.Discrete(2),
            "primary_busy_obss": spaces.Discrete(2),
            "npca_busy_intra": spaces.Discrete(2),
            "npca_busy_obss": spaces.Discrete(2),
            "primary_occupied_rem": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "primary_obss_rem": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "npca_occupied_rem": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "npca_obss_rem": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
            "ppdu_norm": spaces.Box(0.0, 1.0, (1,), dtype=np.float32),
        })

        # Runtime state
        self.channels: List[Channel] = []
        self.stas: List[STA] = []
        self.agent: Optional[STA] = None
        self.current_slot: int = 0
        self._prev_state_name: Optional[str] = None
        self._prev_tx_remaining: int = 0
        self._last_action: Optional[int] = None

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        # Build channels
        self.channels = [
            Channel(self.cfg.primary_channel_id, self.cfg.obss_rate_primary, self.cfg.obss_duration_range),
            Channel(self.cfg.npca_channel_id, self.cfg.obss_rate_npca, self.cfg.obss_duration_range),
        ]

        # Build STAs: agent on primary by default; others split across channels
        self.stas = []
        for i in range(self.cfg.n_stas):
            ch_id = self.cfg.primary_channel_id if i % 2 == 0 else self.cfg.npca_channel_id
            sta = STA(
                sta_id=i,
                channel_id=ch_id,
                primary_channel=self.channels[self.cfg.primary_channel_id],
                npca_channel=self.channels[self.cfg.npca_channel_id],
                npca_enabled=True,  # all STAs support NPCA in step 1
                ppdu_duration=self.cfg.ppdu_duration,
            )
            self.stas.append(sta)
        self.agent = self.stas[self.cfg.agent_sta_idx]

        # Reset slot clock and book-keeping
        self.current_slot = 0
        self._prev_state_name = self.agent.state.name if self.agent else None
        self._prev_tx_remaining = getattr(self.agent, "tx_remaining", 0)
        self._last_action = None

        # Advance until the first decision point
        obs = self._advance_until_decision_or_end()
        info = {"slot": self.current_slot}
        return obs, info

    def step(self, action: int):
        assert self.agent is not None
        reward = 0.0
        terminated = False
        truncated = False

        # Cost for switching
        if action == 1:
            reward += self.cfg.switch_penalty

        # Apply action for exactly one slot via the npca_enabled hack
        original_flag = self.agent.npca_enabled
        if action == 0:
            self.agent.npca_enabled = False  # forces PRIMARY_FROZEN path under primary OBSS
        self._simulate_one_slot(self.current_slot)
        self.agent.npca_enabled = original_flag
        self.current_slot += 1

        # Time penalty for that elapsed slot
        reward += self.cfg.time_penalty

        # Continue simulation until next decision or episode end, accumulating rewards
        obs = self._advance_until_decision_or_end(reward_accumulator=reward)
        reward = obs.pop("_accumulated_reward")  # retrieve accumulated reward from obs builder

        terminated = (self.current_slot >= self.cfg.num_slots)
        info = {"slot": self.current_slot, "last_action": int(action)}
        return obs, float(reward), terminated, truncated, info

    # -------------------------
    # Core helpers
    # -------------------------
    def _decision_condition(self) -> bool:
        """Return True when the agent should act: PRIMARY_BACKOFF with primary OBSS present."""
        a = self.agent
        prim = self.channels[self.cfg.primary_channel_id]
        return (
            (a.state.name == "PRIMARY_BACKOFF") and
            (prim.obss_remain > 0) and  # primary busy due to OBSS
            (a.npca_enabled is True)
        )

    def _advance_until_decision_or_end(self, reward_accumulator: float = 0.0) -> Dict[str, np.ndarray]:
        """Run slot-by-slot until we hit the next decision point or the episode ends.
        Accumulate rewards for transmissions that *finish* during this rollout.
        Returns the observation dict at the decision point (or terminal obs if ended).
        """
        while (self.current_slot < self.cfg.num_slots) and (not self._decision_condition()):
            # Before stepping, remember if agent is mid-TX and about to finish
            reward_accumulator += self._simulate_one_slot(self.current_slot)
            self.current_slot += 1
            reward_accumulator += self.cfg.time_penalty

        obs = self._get_obs_dict()
        obs["_accumulated_reward"] = reward_accumulator
        return obs

    def _simulate_one_slot(self, slot: int) -> float:
        """A single-slot version of Simulator.run() main loop. Returns reward delta for any
        agent TX that *finishes* this slot (success/collision)."""
        # ① Update channels (expire busy periods, refresh remain counters)
        for ch in self.channels:
            ch.update(slot)

        # Track pre-step snapshot for reward calc
        was_tx = (self.agent.state.name in ("PRIMARY_TX", "NPCA_TX"))
        prev_tx_remaining = int(getattr(self.agent, "tx_remaining", 0))

        # ② Each STA picks/updates state
        for sta in self.stas:
            sta.occupy_request = None
            sta.step(slot)

        # ③ Stochastic OBSS
        obss_reqs: List[Tuple[Optional[STA], object]] = []
        for ch in self.channels:
            obss_req = ch.generate_obss(slot)
            if obss_req:
                obss_reqs.append((None, obss_req))

        # ④ Collect STA occupy requests
        sta_reqs = [(sta, sta.occupy_request) for sta in self.stas if sta.occupy_request is not None]

        # ⑤ Merge all requests (STA + OBSS)
        all_reqs = sta_reqs + obss_reqs

        # ⑥ Bucket by channel
        from collections import defaultdict
        channel_requests: Dict[int, List[Tuple[Optional[STA], object]]] = defaultdict(list)
        for sta, req in all_reqs:
            channel_requests[req.channel_id].append((sta, req))

        # ⑦ Resolve per channel
        for ch_id, reqs in channel_requests.items():
            if len(reqs) == 1:
                sta, req = reqs[0]
                if req.is_obss:
                    self.channels[ch_id].add_obss_traffic(req, slot)
                else:
                    self.channels[ch_id].occupy(slot, req.duration, sta.sta_id)
                if sta is not None:
                    sta.tx_success = True
            else:
                for sta, req in reqs:
                    if sta is not None:
                        if req.is_obss:
                            self.channels[ch_id].add_obss_traffic(req, slot)
                        else:
                            self.channels[ch_id].occupy(slot, req.duration, sta.sta_id)
                        sta.tx_success = False
                        # collisions are recorded here; exponential backoff will be applied inside STA on next step

        # ⑧ Commit next_state
        for sta in self.stas:
            sta.state = sta.next_state

        # Reward calc for agent when transmission *finishes* this slot
        reward_delta = 0.0
        now_tx = (self.agent.state.name in ("PRIMARY_TX", "NPCA_TX"))
        now_tx_remaining = int(getattr(self.agent, "tx_remaining", 0))

        # If previously mid-TX and now remaining became 0 inside STA.step -> TX finished this slot
        if was_tx and (now_tx_remaining == 0) and (prev_tx_remaining > 0):
            if getattr(self.agent, "tx_success", False):
                reward_delta += self.cfg.success_reward
            else:
                reward_delta += self.cfg.collision_penalty

        return reward_delta

    # -------------------------
    # Observation builder
    # -------------------------
    def _get_obs_dict(self) -> Dict[str, np.ndarray]:
        a = self.agent
        prim = self.channels[self.cfg.primary_channel_id]
        npca = self.channels[self.cfg.npca_channel_id]

        def norm(x: float, maxv: float) -> float:
            return float(np.clip(x / maxv, 0.0, 1.0)) if maxv > 0 else 0.0

        obs = {
            "primary_backoff": np.array([norm(float(a.backoff), self.cfg.max_backoff)], dtype=np.float32),
            "cw_index": np.array([norm(float(getattr(a, "cw_index", 0)), len(CONTENTION_WINDOW) - 1)], dtype=np.float32),
            "npca_enabled": int(bool(a.npca_enabled)),
            "primary_busy_intra": int(prim.occupied_remain > 0),
            "primary_busy_obss": int(prim.obss_remain > 0),
            "npca_busy_intra": int(npca.occupied_remain > 0),
            "npca_busy_obss": int(npca.obss_remain > 0),
            "primary_occupied_rem": np.array([norm(float(prim.occupied_remain), self.cfg.max_ppdu_slots)], dtype=np.float32),
            "primary_obss_rem": np.array([norm(float(prim.obss_remain), self.cfg.max_ppdu_slots)], dtype=np.float32),
            "npca_occupied_rem": np.array([norm(float(npca.occupied_remain), self.cfg.max_ppdu_slots)], dtype=np.float32),
            "npca_obss_rem": np.array([norm(float(npca.obss_remain), self.cfg.max_ppdu_slots)], dtype=np.float32),
            "ppdu_norm": np.array([norm(float(a.ppdu_duration), self.cfg.max_ppdu_slots)], dtype=np.float32),
        }
        return obs

    # -------------------------
    # Render/Close (placeholders)
    # -------------------------
    def render(self):
        return None

    def close(self):
        return None


# Factory (keeps parity with your other envs)
def make_env(**kwargs):
    cfg = NPCAEnvConfig(**kwargs)
    return partial(NPCAEnv, cfg)
