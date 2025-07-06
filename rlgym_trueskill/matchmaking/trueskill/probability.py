import numpy as np
from trueskill import global_env

def probability_NvsM(team1_ratings, team2_ratings, env=None):
    """
    Calculate the win probability of team1 over team2 using TrueSkill ratings.

    :param team1_ratings: List of TrueSkill ratings for team1
    :param team2_ratings: List of TrueSkill ratings for team2
    :param env: TrueSkill environment (defaults to global environment)
    :return: Probability of team1 winning
    """
    if env is None:
        env = global_env()

    team1_mu = sum(r.mu for r in team1_ratings)
    team1_sigma = sum((env.beta ** 2 + r.sigma ** 2) for r in team1_ratings)
    team2_mu = sum(r.mu for r in team2_ratings)
    team2_sigma = sum((env.beta ** 2 + r.sigma ** 2) for r in team2_ratings)

    x = (team1_mu - team2_mu) / np.sqrt(team1_sigma + team2_sigma)
    probability_win_team1 = env.cdf(x)
    return probability_win_team1