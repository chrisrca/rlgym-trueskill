from rlgym_trueskill.trueskillworker import TrueSkillWorker
from rlgym_sim.utils.action_parsers.continuous_act import ContinuousAction
from rlgym_sim.utils.obs_builders.advanced_obs import AdvancedObs
import numpy as np

if __name__ == "__main__":
    tick_skip=4
    action_parser = ContinuousAction()
    obs_builder = AdvancedObs()

    worker = TrueSkillWorker(
        tick_skip=tick_skip,
        past_version_prob=0.5,
        timeout_seconds=500,
        action_parser=action_parser,
        obs_builder=obs_builder,
        render=False,
        wandb=None,
        model_folder=None,
        device="cpu",
        gamemodes={'1v1s': True, '2v2s': True, '3v3s': True}
    ).run()