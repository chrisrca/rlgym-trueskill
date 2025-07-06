from rlgym.api import RLGym
from rlgym_sim.gym import Gym
from trueskill import rate
from rlgym_trueskill.matchmaking.v1.utils.dynamicgmsetter import DynamicGMSetter
from rlgym_trueskill.matchmaking.v1.utils.generate_episode import generate_episode
from rlgym_trueskill.matchmaking.trueskill.probability import probability_NvsM
import numpy as np
import os
import torch

class Matchmaker:
    def __init__(self, ModelPolicy, match, past_version_prob=0.75, sigma_target=2, 
                 dynamic_gm=True, gamemode_weights=None, gamemode=None, render=True):

        self.ModelPolicy = ModelPolicy
        self.match=match
        self.past_version_prob=past_version_prob
        self.sigma_target=2
        self.sigma_target = sigma_target
        self.dynamic_gm = dynamic_gm
        self.gamemode_weights = gamemode_weights
        self.gamemode = gamemode
        self.render = render
        if self.gamemode_weights is not None:
            assert np.isclose(sum(self.gamemode_weights.values()), 1), "gamemode_weights must sum to 1"
        state_setter = DynamicGMSetter(match._state_setter)
        self.set_team_size = state_setter.set_team_size
        match._state_setter = state_setter
        self.env = Gym(
            match=self.match,
            tick_skip=4,
            dodge_deadzone=0.5,
            boost_consumption=1,
            copy_gamestate_every_step=True,
            gravity=1
        )

    def run(self):
        pass