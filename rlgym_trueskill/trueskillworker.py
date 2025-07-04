from rlgym.api import RLGym
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.rlviser import RLViserRenderer
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import KickoffMutator, MutatorSequence
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator
from rlgym_trueskill.matchmaking.matchmaker import Matchmaker
from rlgym_trueskill.rewards.dummy_reward import DummyReward
from multiprocessing import Process

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
        print("Starting TrueSkillWorkers...")

        # Start processes for each enabled gamemode
        processes = []
        for mode, enabled in self.gamemodes.items():
            if enabled:
                team_size = int(mode[0])
                # Create the RLGym instance
                match = RLGym(
                    state_mutator=MutatorSequence(
                        FixedTeamSizeMutator(team_size,team_size),
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
                
                # Pass the ready-made match to matchmaker
                matchmaker = Matchmaker(team_size=team_size, match=match)
                processes.append(Process(target=matchmaker.run))

        for p in processes:
            p.start()
        for p in processes:
            p.join()

