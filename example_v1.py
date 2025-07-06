from rlgym_trueskill.trueskillworker import TrueSkillWorker
from rlgym_sim.utils.action_parsers.continuous_act import ContinuousAction
from rlgym_sim.utils.obs_builders.advanced_obs import AdvancedObs
import numpy as np
import torch

# !!! YOU MUST IMPLEMENT THIS CLASS !!!
# This class should define how your model interacts with the environment.
class ModelPolicy:
    # This method takes a model's path as input and utilizes it to produce actions.
    # You should modify this method to fit your model if it does not already.
    def __init__(self, model):
        self.model = model

    # This method takes a state as input, processes it through the model, and returns the action.
    # You should modify this method to fit your model's action selection needs.
    def get_action(self, state, device):
        # Convert state to tensor and move to device
        state = torch.tensor(state, dtype=torch.float32).to(device)
        
        # Perform model inference
        with torch.no_grad():
            action = self.model(state)
        
        # Ensure action is on CPU, converted to numpy, and reshaped to (1, 8)
        return action.cpu().numpy().reshape(1, 8)

if __name__ == "__main__":
    # The TrueSkillWorker will manage all matchmaking, model loading, and logging. 
    worker = TrueSkillWorker(
        ModelPolicy=ModelPolicy, # Your custom policy above (this is required!)
        tick_skip=4, # Your model's tick skip.
        past_version_prob=0.75, # Probability of selecting a past version of the model.
        timeout_seconds=500, # Timeout for each match if a goal is not scored.
        action_parser=ContinuousAction(), # Action parser to use for matches.
        obs_builder=AdvancedObs(), # Obs builder to use for matches.
        render=False, # Render in RLViser if available.
        wandb=None, # Your wandb instance to log the graph to.
        model_folder=None, # Path to the folder containing your models. Model checkpoints should be named as [<name>]-[<version>].pt e.g. "model-1.pt", "model-2.pt", etc. 
        device="cpu", # Device to run the model on, e.g. "cpu" or "cuda:0".
        gamemodes={'1v1s': True, '2v2s': True, '3v3s': True} # Gamemodes to use for matchmaking, e.g. {'1v1s': True, '2v2s': True, '3v3s': True}.
    ).run()