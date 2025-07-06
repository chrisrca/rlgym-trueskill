# TESTING FILE NOT FOR RELEASE!!!

from rlgym_trueskill.trueskillworker import TrueSkillWorker
from rlgym_sim.utils.action_parsers.continuous_act import ContinuousAction
from rlgym_sim.utils.obs_builders.advanced_obs import AdvancedObs
import numpy as np
import torch

class ModelPolicy:
    def __init__(self, model):
        self.model = model

    def get_actions(self, state, device):
        # Test
        print("ModelPolicy get_action called")
        return np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32) # Throttle

if __name__ == "__main__":
    worker = TrueSkillWorker(
        ModelPolicy=ModelPolicy,
        tick_skip=4,
        past_version_prob=0.75,
        timeout_seconds=500,
        action_parser=ContinuousAction(),
        obs_builder=AdvancedObs(),
        render=False,
        wandb=None,
        model_folder="C:/Users/goali/Downloads/checkpoints",
        device="cpu",
        gamemodes={'1v1s': True, '2v2s': True, '3v3s': True}
    ).run()