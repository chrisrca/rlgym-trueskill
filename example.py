from rlgym_trueskill.trueskillworker import TrueSkillWorker
from rlgym.rocket_league.action_parsers import RepeatAction
from rlgym_tools.rocket_league.action_parsers.advanced_lookup_table_action import AdvancedLookupTableAction
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league import common_values
import numpy as np

if __name__ == "__main__":
    tick_skip=4
    action_parser = RepeatAction(AdvancedLookupTableAction(3,3,3,16,True), repeats=tick_skip)

    obs_builder = DefaultObs(
        zero_padding=3,
        pos_coef=np.asarray([1 / common_values.SIDE_WALL_X, 1 / common_values.BACK_NET_Y, 1 / common_values.CEILING_Z]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL
    )

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