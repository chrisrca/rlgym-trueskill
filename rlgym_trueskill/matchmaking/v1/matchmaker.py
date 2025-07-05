from rlgym.api import RLGym
from rlgym_ppo.util import RLGymV2GymWrapper
import os
import time

class Matchmaker:
    def __init__(self, team_size, match: RLGym):
        self.team_size = team_size
        self.match = match

    def run(self):
        print(f"\nStarting matchmaker for team size {self.team_size}")
        print(f"Match: {self.match}")
        print(f"Process ID: {os.getpid()}")
