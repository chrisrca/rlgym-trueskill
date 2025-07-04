from rlgym.api import RLGym
from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition
from rlgym.rocket_league.rlviser import RLViserRenderer
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import KickoffMutator
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
        self.device = device
        self.gamemodes = gamemodes

        if action_parser is None:
            raise ValueError("No action parser provided.")
        else:
            self.action_parser = action_parser

        if obs_builder is None:
            raise ValueError("No observation builder provided.")
        else:
            self.obs_builder = obs_builder

        self.match = RLGym(
            state_mutator=KickoffMutator(), # Use KickoffMutator to set up the initial state      
            obs_builder=obs_builder,    # User provided observation builder 
            action_parser=action_parser,    # User provided action parser
            reward_fn=DummyReward(),    # We don't need rewards
            termination_cond=GoalCondition(),   # Terminate when a goal is scored   
            truncation_cond=NoTouchTimeoutCondition(timeout_seconds=timeout_seconds),   # Terminate after a timeout
            transition_engine=RocketSimEngine(),    # Use RocketSimEngine for the simulation
            renderer=RLViserRenderer(), # Use RLViserRenderer for rendering
        )

    def run(self):
        print("Starting TrueSkillWorkers...")

        # Start processes for each enabled gamemode
        processes = []
        for mode, enabled in self.gamemodes.items():
            if enabled:
                team_size = int(mode[0])
                matchmaker = Matchmaker(team_size=team_size, match=self.match)
                processes.append(Process(target=matchmaker.run))

        for p in processes:
            p.start()
        for p in processes:
            p.join()

