from rlgym_trueskill.matchmaking.v1.rewards.dummy_reward import DummyReward
from multiprocessing import Process
import inspect

# Auto-detect if using rlgym_sim (v1) or rlgym (v2)
def is_rlgym_sim(obj):
    if obj is None:
        return False
    module = inspect.getmodule(obj.__class__)
    return module is not None and module.__name__.startswith('rlgym_sim')

class TrueSkillWorker: 
    def __init__( # No default values for parameters, they must be provided. (render is an exception as it doesn't affect how the worker runs)
            self, 
            ModelPolicy=None,
            tick_skip=None, 
            past_version_prob=None, 
            timeout_seconds=None, 
            action_parser=None, 
            obs_builder=None, 
            render=False, 
            wandb=None, 
            model_folder=None, 
            device=None,
            gamemodes={'1v1s': True, '2v2s': True, '3v3s': True}
    ):
        # Auto detect version using module names
        self.v1 = is_rlgym_sim(action_parser) or is_rlgym_sim(obs_builder)
        print(f"Detected {'RLGym Sim' if self.v1 else 'RLGym v2'}.")

        # Safety checks for required parameters (missing parameters will cause errors downstream that we need to catch early)
        self.ModelPolicy = ModelPolicy
        if self.ModelPolicy is None:
            raise ValueError("ModelPolicy must be provided.")

        self.tick_skip = tick_skip
        if self.tick_skip is None:
            print("No tick skip specified, defaulting to 4.")
            self.tick_skip = 4
        self.fps = 120 // self.tick_skip
        
        self.past_version_prob = past_version_prob
        if self.past_version_prob is None:
            print("No past version probability specified, defaulting to 0.75.")
            self.past_version_prob = 0.75

        self.timeout_seconds = timeout_seconds
        if self.timeout_seconds is None:
            print("No timeout specified, defaulting to 500 seconds.")
            self.timeout_seconds = 500

        self.wandb = wandb
        # if self.wandb is None:
        #     raise ValueError("wandb instance must be provided.")
        
        self.model_folder = model_folder
        if self.model_folder is None:
            raise ValueError("model_folder must be provided.")
        
        self.action_parser = action_parser
        if self.action_parser is None:
            raise ValueError("action_parser must be provided.")
        
        self.obs_builder = obs_builder
        if self.obs_builder is None:
            raise ValueError("obs_builder must be provided.")
        
        self.device = device
        if self.device is None:
            print("No inference device specified, defaulting to 'cpu'.")
            self.device = "cpu"

        self.render = render
        self.gamemodes = gamemodes

    def _v1_worker(self, team_size):
        from rlgym_sim.envs import Match
        from rlgym_sim.utils.terminal_conditions.common_conditions import GoalScoredCondition, TimeoutCondition, NoTouchTimeoutCondition
        from rlgym_sim.utils.state_setters.default_state import DefaultState
        from rlgym_trueskill.matchmaking.v1.matchmaker import Matchmaker
    
        match = Match(
            state_setter=DefaultState(),
            obs_builder=self.obs_builder,
            action_parser=self.action_parser,
            reward_function=DummyReward(),
            terminal_conditions=[TimeoutCondition(self.fps * self.timeout_seconds), GoalScoredCondition()],
            spawn_opponents=True,
            team_size=team_size,
        )
        Matchmaker(ModelPolicy=self.ModelPolicy, match=match, device=self.device).run()

    def run(self):
        print("Starting TrueSkillWorkers...")

        # Start processes for each enabled gamemode
        processes = []
        for mode, enabled in self.gamemodes.items():
            if enabled:
                team_size = int(mode[0])
                # V1 Matchmaker
                if self.v1:
                    team_size = int(mode[0])
                    processes.append(Process(target=self._v1_worker, args=(team_size,)))

        for p in processes:
            p.start()
        for p in processes:
            p.join()
