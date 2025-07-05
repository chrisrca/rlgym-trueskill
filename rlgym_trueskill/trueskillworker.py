from rlgym_trueskill.rewards.dummy_reward import DummyReward
from multiprocessing import Process
import inspect

# Auto-detect if using rlgym_sim (v1) or rlgym (v2)
def is_rlgym_sim(obj):
    if obj is None:
        return False
    module = inspect.getmodule(obj.__class__)
    return module is not None and module.__name__.startswith('rlgym_sim')

class TrueSkillWorker:
    def __init__(
            self, 
            tick_skip=4, 
            past_version_prob=0.5, 
            timeout_seconds=500, 
            action_parser=None, 
            obs_builder=None, 
            render=False, 
            wandb=None, 
            model_folder=None, 
            device="cpu",
            gamemodes={'1v1s': True, '2v2s': True, '3v3s': True}
    ):
        # Auto detect version using module names
        self.v1 = is_rlgym_sim(action_parser) or is_rlgym_sim(obs_builder)
        print(f"Detected {'RLGym Sim' if self.v1 else 'RLGym v2'}.")

        self.tick_skip = tick_skip
        self.fps = 120 // self.tick_skip
        self.past_version_prob = past_version_prob
        self.render = render
        self.wandb = wandb
        self.model_folder = model_folder
        self.action_parser = action_parser
        self.obs_builder = obs_builder
        self.device = device
        self.gamemodes = gamemodes
        self.timeout_seconds = timeout_seconds

    def run(self):
        # Import necessary modules based on version
        if self.v1:
            from rlgym_sim.utils.action_parsers.continuous_act import ContinuousAction
            from rlgym_sim.utils.gamestates.game_state import GameState, PlayerData
            from rlgym_sim.envs import Match
            from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition, NoTouchTimeoutCondition
            from rlgym_sim.utils.state_setters.default_state import DefaultState
            from rlgym_trueskill.matchmaking.v1.matchmaker import Matchmaker
        else:
            from rlgym.api import RLGym
            from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition
            from rlgym.rocket_league.rlviser import RLViserRenderer
            from rlgym.rocket_league.sim import RocketSimEngine
            from rlgym.rocket_league.state_mutators import KickoffMutator, MutatorSequence
            from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator
            from rlgym_trueskill.matchmaking.v2.matchmaker import Matchmaker

        print("Starting TrueSkillWorkers...")

        # Start processes for each enabled gamemode
        processes = []
        for mode, enabled in self.gamemodes.items():
            if enabled:
                team_size = int(mode[0])
                # V1 Matchmaker
                if self.v1:
                    match = Match(
                        state_setter=DefaultState(),
                        obs_builder=self.obs_builder,
                        action_parser=self.action_parser,
                        reward_function=DummyReward(),
                        terminal_conditions=[TimeoutCondition(self.fps * 500), GoalScoredCondition()],
                        spawn_opponents=True,
                        team_size=team_size,
                    )
                    matchmaker = Matchmaker(team_size=team_size, match=match)
                    processes.append(Process(target=matchmaker.run))
                # V2 Matchmaker
                else:
                    match = RLGym(
                        state_mutator=MutatorSequence(
                            FixedTeamSizeMutator(team_size, team_size),
                            KickoffMutator(),
                        ),
                        obs_builder=self.obs_builder,
                        action_parser=self.action_parser,
                        reward_fn=DummyReward(),
                        termination_cond=GoalCondition(),
                        truncation_cond=NoTouchTimeoutCondition(timeout_seconds=self.timeout_seconds),
                        transition_engine=RocketSimEngine(),
                        renderer=RLViserRenderer(),
                    )
                    matchmaker = Matchmaker(team_size=team_size, match=match)
                    processes.append(Process(target=matchmaker.run))

        for p in processes:
            p.start()
        for p in processes:
            p.join()
