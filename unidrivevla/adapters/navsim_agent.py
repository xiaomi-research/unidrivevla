"""A policy adapter as a NAVSIM agent, for the official entries of either devkit.

cache_mode (run_dataset_caching.py): only the canonical NAVSIM builders, no network. Otherwise (run_pdm_score*.py):
compute_trajectory builds the canonical features of the AgentInput, runs the policy's worker transform and eval forward,
and returns its ego-frame poses.
"""

import inspect
from typing import Any, Optional

import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, Trajectory

from unidrivevla.adapters import initialize_for_inference
from unidrivevla.data.navsim import SampleFeatureBuilder, SampleTargetBuilder, sensor_config
from unidrivevla.data.registry import POSE_INTERVAL_S

AGENT_TAKES_SAMPLING = "trajectory_sampling" in inspect.signature(AbstractAgent.__init__).parameters  # NAVSIM 2.0


class UniDriveVLANavsimAgent(AbstractAgent):
    def __init__(self, policy: Optional[Any] = None, cache_mode: bool = False):
        if not cache_mode and policy is None:
            raise ValueError("a policy is required unless cache_mode=True")
        policy = None if cache_mode else policy
        if policy is not None and callable(getattr(policy, "policy_for_source", None)):
            policy = policy.policy_for_source("navsim")
        poses = policy.trajectory_sampling.num_poses if policy is not None else 8
        sampling = TrajectorySampling(num_poses=poses, interval_length=POSE_INTERVAL_S)
        if AGENT_TAKES_SAMPLING:
            super().__init__(sampling)
        else:
            super().__init__()
        self.policy, self._sampling = policy, sampling
        self._features = SampleFeatureBuilder()
        self._transform = self.policy.worker_transform() if self.policy is not None else None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def name(self) -> str:
        return "UniDriveVLANavsimAgent"

    def initialize(self) -> None:
        if self.policy is None:
            return
        initialize_for_inference(self.policy)
        self.policy.to(self._device).eval()

    def get_sensor_config(self):
        return sensor_config()

    def get_feature_builders(self):
        return [self._features]

    def get_target_builders(self):
        return [SampleTargetBuilder()]

    @torch.no_grad()
    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        features, _ = self._transform(self._features.compute_features(agent_input), {})
        batch = {k: v.unsqueeze(0).to(self._device) if torch.is_tensor(v) else [v] for k, v in features.items()}
        poses = self.policy(batch)["pred_traj"][0].float().cpu().numpy()
        return Trajectory(poses, self._sampling)
