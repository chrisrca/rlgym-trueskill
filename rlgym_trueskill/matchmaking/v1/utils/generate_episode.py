import numpy as np
from rlgym_sim.gym import Gym
from tqdm import tqdm

def generate_episode(ModelPolicy, env: Gym, policies, evaluate=False, progress=False, render=False):
    """
    Run a single match in the environment with the given policies and return the result.

    :param env: RLGym environment
    :param policies: List of policies (ModelPolicy instances) for each agent
    :param evaluate: If True, run in evaluation mode (returns only result)
    :param progress: If True, show a progress bar with in-game time
    :param render: If True, render the environment using rlviser
    :return: Match result (int) for evaluation mode, or (None, result) for compatibility
    """
    if progress:
        progress_bar = tqdm(unit="steps")
    else:
        progress_bar = None

    observations, info = env.reset(return_info=True)
    result = 0
    index = 0

    # Ensure policies is a list
    if not isinstance(policies, list):
        policies = [policies]

    while True:
        all_actions = []

        if not isinstance(observations, list):
            observations = [observations]

        print(ModelPolicy.get_actions(observations[0], env.device))  # Debugging line to check actions

        # Zip policies and observations correctly
        for policy, obs in zip(policies, observations):
            if isinstance(policy, ModelPolicy):
                actions = policy.get_actions(obs)
                all_actions.append(actions)
            else:
                raise ValueError(f"Unsupported policy type: {type(policy)}")

        all_actions = np.concatenate(all_actions, axis=0)
        observations, rewards, done, info = env.step(all_actions)

        if render:
            env.render()  # Render the environment after each step

        if progress_bar is not None:
            progress_bar.update()
            igt = progress_bar.n * 4 / 120
            progress_bar.set_postfix_str(f"{igt // 60:02.0f}:{igt % 60:02.0f} IGT")
        
        index += 1

        if done:
            result += info.get("result", 0)
            break

    if progress_bar is not None:
        progress_bar.close()

    if evaluate:
        return result
    return None, result