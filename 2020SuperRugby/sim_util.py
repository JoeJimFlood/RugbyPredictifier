'''
Utility functions for SportPredictifier Simulation
'''
import numpy as np
from numpy.random import poisson, binomial, negative_binomial

def sim_poisson(mean, n_sim):
    '''
    Simulates Poisson for when mean is equal to variance
    '''
    return poisson(mean, n_sim)

def sim_negative_binomial(mean, var, n_sim):
    '''
    Simulates negative binomial for when mean is less than variance
    '''
    p = mean / var
    n = mean * p / (1-p)
    try:
        return negative_binomial(n, p, n_sim)
    except ValueError:
        print(mean, var)
        print(n, p)
        raise Exception

def sim_binomial(mean, var, n_sim):
    '''
    Simulates binomial for when mean is greater than variance
    '''
    p = 1 - (var/mean)
    n = (mean / p)
    floor_n = int(n)
    high_prob = n - floor_n
    ns = floor_n + binomial(1, high_prob, n_sim)
    try:
        return binomial(ns, p)
    except ValueError:
        print(mean, var, p)

def sim(mean, var, n_sim):
    #Check if there's a negative mean or variance. If so, set one to the other so a Poisson distribution can be used.
    if mean < 0:
        mean = var
    if var < 0:
        var = mean

    if mean > var:
        return sim_binomial(mean, var, n_sim)
    elif mean < var:
        return sim_negative_binomial(mean, var, n_sim)
    else:
        return sim_poisson(mean, n_sim)

#############################################################################################################################
def calculate_score(scores, score_array):
    #assert len(scores) == len(score_array), 'Score array and scores must have same length'
    score_array = np.array(score_array)
    return np.dot(np.vstack(scores).T, score_array)

def eval_results(team_1_scores, team_2_scores, knockout = False):
    team_1_wins = (team_1_scores > team_2_scores).astype(float)
    team_2_wins = (team_1_scores < team_2_scores).astype(float)
    draws = (team_1_scores == team_2_scores).astype(float)
    if knockout:
        team_1_wins += (0.5*draws)
        team_2_wins += (0.5*draws)
        draws = np.zeros_like(team_1_wins)
        return team_1_wins, team_2_wins, draws
    else:
        return team_1_wins, team_2_wins, draws

def eval_try_bonus(team_1_tries, team_2_tries, req_diff):
    team_1_bp = (team_1_tries - team_2_tries >= req_diff).astype(int)
    team_2_bp = (team_2_tries - team_1_tries >= req_diff).astype(int)
    return team_1_bp, team_2_bp

def eval_losing_bonus(team_1_score, team_2_score, req_diff):
    diff = team_1_score - team_2_score
    team_1_bp = ((diff < 0)*(diff >= -req_diff)).astype(int)
    team_2_bp = ((diff > 0)*(diff <= req_diff)).astype(int)
    return team_1_bp, team_2_bp