import os
import sys
import pandas as pd
import numpy as np
from numpy.random import poisson, uniform
from numpy import mean
import time
import math
import pdb
po = True

teamsheetpath = sys.path[0] + '/teamcsvs/'

compstat = {'TF': 'TA', 'TA': 'TF', #Dictionary to use to compare team stats with opponent stats
            'CF': 'CA', 'CA': 'CF',
            'CON%F': 'CON%A', 'CON%A': 'CON%F',
            'PF': 'PA', 'PA': 'PF',
            'DGF': 'DGA', 'DGA': 'DGF'}

def get_opponent_stats(opponent): #Gets summaries of statistics for opponent each week
    opponent_stats = {}
    global teamsheetpath
    opp_stats = pd.DataFrame.from_csv(teamsheetpath + opponent + '.csv')
    for stat in opp_stats.columns:
        try:
            opponent_stats.update({stat: opp_stats[stat].mean()})
        except TypeError:
            continue
    opponent_stats.update({'CON%F': float(opp_stats['CF'].sum())/opp_stats['TF'].sum()})
    opponent_stats.update({'CON%A': float(opp_stats['CA'].sum())/opp_stats['TA'].sum()})
    return opponent_stats

def get_residual_performance(team): #Get how each team has done compared to the average performance of their opponents
    global teamsheetpath
    score_df = pd.DataFrame.from_csv(teamsheetpath + team + '.csv')
    residual_stats = {}
    score_df['CON%F'] = np.nan
    score_df['CON%A'] = np.nan
    for week in score_df.index:
        opponent_stats = get_opponent_stats(score_df['OPP'][week])
        for stat in opponent_stats:
            if week == score_df.index.tolist()[0]:
                score_df['OPP_' + stat] = np.nan       
            score_df['OPP_' + stat][week] = opponent_stats[stat]
        score_df['CON%F'][week] = float(score_df['CF'][week]) / score_df['TF'][week]
        score_df['CON%A'][week] = float(score_df['CA'][week]) / score_df['TA'][week]
    #print opponent_stats
    for stat in opponent_stats:
        
        score_df['R_' + stat] = score_df[stat] - score_df['OPP_' + compstat[stat]]
        if stat in ['TF', 'PF', 'DGF', 'TA', 'PA', 'DGA']:
            residual_stats.update({stat: score_df['R_' + stat].mean()})
        elif stat == 'CON%F':
            residual_stats.update({stat: (score_df['R_CON%F'].multiply(score_df['TF'])).sum() / score_df['TF'].sum()})
        elif stat == 'CON%A':
            residual_stats.update({stat: (score_df['R_CON%A'].multiply(score_df['TA'])).sum() / score_df['TA'].sum()})
    return residual_stats

def get_score(expected_scores): #Get the score for a team based on expected scores
    score = 0
    if expected_scores['T'] > 0:
        tries = poisson(expected_scores['T'])
    else:
        tries = poisson(0.01)
    score = score + 6 * tries
    if expected_scores['P'] > 0:
        fgs = poisson(expected_scores['P'])
    else:
        fgs = poisson(0.01)
    score = score + 3 * fgs
    if expected_scores['DG'] > 0:
        sfs = poisson(expected_scores['DG'])
    else:
        sfs = poisson(0.01)
    score = score + 2 * sfs
    for t in range(tries):
        successful_con_determinant = uniform(0, 1)
        if successful_con_determinant <= expected_scores['CONPROB']:
            score += 2
        else:
            continue
    if tries >= 4:
        bp = True
    else:
        bp = False
    return (score, bp)

def game(team_1, team_2,
         expected_scores_1, expected_scores_2,
         playoff = False): #Get two scores and determine a winner
    (score_1, bp1) = get_score(expected_scores_1)
    (score_2, bp2) = get_score(expected_scores_2)
    if score_1 > score_2:
        win_1 = 1
        win_2 = 0
        draw_1 = 0
        draw_2 = 0
        if bp1:
            bpw1 = 1
        else:
            bpw1 = 0
        if bp2:
            bpl2 = 1
        else:
            bpl2 = 0
        bpl1 = 0
        bpw2 = 0
        bpd1 = 0
        bpd2 = 0
        lbp1 = 0
        if score_1 - score_2 <= 7:
            lbp2 = 1
        else:
            lbp2 = 0

    elif score_2 > score_1:
        win_1 = 0
        win_2 = 1
        draw_1 = 0
        draw_2 = 0
        if bp1:
            bpl1 = 1
        else:
            bpl1 = 0
        if bp2:
            bpw2 = 1
        else:
            bpw2 = 0
        bpw1 = 0
        bpl2 = 0
        bpd1 = 0
        bpd2 = 0
        lbp2 = 0
        if score_2 - score_1 <= 7:
            lbp1 = 1
        else:
            lbp1 = 0
    else:
        if playoff:
            win_1 = 0.5
            win_2 = 0.5
            draw_1 = 0
            draw_2 = 0
            bpw1 = 0
            bpw2 = 0
            bpd1 = 0
            bpd2 = 0
            bpl1 = 0
            bpl2 = 0
            lbp1 = 0
            lbp2 = 0
        else:
            win_1 = 0
            win_2 = 0
            draw_1 = 1
            draw_2 = 1
            bpw1 = 0
            bpw2 = 0
            bpl1 = 0
            bpl2 = 0
            lbp1 = 0
            lbp2 = 0
        if bp1:
            bpd1 = 1
        else:
            bpd1 = 0
        if bp2:
            bpd2 = 1
        else:
            bpd2 = 0
    summary = {team_1: [win_1, draw_1, score_1, bpw1, bpd1, bpl1, lbp1]}
    summary.update({team_2: [win_2, draw_2, score_2, bpw2, bpd2, bpl2, lbp2]})
    return summary

def get_expected_scores(team_1_stats, team_2_stats, team_1_df, team_2_df): #Get the expected scores for a matchup based on the previous teams' performances
    expected_scores = {}
    for stat in team_1_stats:
        expected_scores.update({'T': mean([team_1_stats['TF'] + team_2_df['TA'].mean(),
                                           team_2_stats['TA'] + team_1_df['TF'].mean()])})
        expected_scores.update({'P': mean([team_1_stats['PF'] + team_2_df['PA'].mean(),
                                           team_2_stats['PA'] + team_1_df['PF'].mean()])})
        expected_scores.update({'DG': mean([team_1_stats['DGF'] + team_2_df['DGA'].mean(),
                                            team_2_stats['DGA'] + team_1_df['DGF'].mean()])})
        #print mean([team_1_stats['PAT1%F'] + team_2_df['PAT1AS'].astype('float').sum() / team_2_df['PAT1AA'].sum(),
        #       team_2_stats['PAT1%A'] + team_1_df['PAT1FS'].astype('float').sum() / team_1_df['PAT1FA'].sum()])
        conprob = mean([team_1_stats['CON%F'] + team_2_df['CA'].astype('float').sum() / team_2_df['TA'].sum(),
                        team_2_stats['CON%A'] + team_1_df['CF'].astype('float').sum() / team_1_df['TF'].sum()])
        if not math.isnan(conprob):
            expected_scores.update({'CONPROB': conprob})
        else:
            expected_scores.update({'CONPROB': 0.75})
        #print(expected_scores['PAT1PROB'])
    #print(expected_scores)
    return expected_scores
            
def matchup(team_1, team_2):
    ts = time.time()
    team_1_season = pd.DataFrame.from_csv(teamsheetpath + team_1 + '.csv')
    team_2_season = pd.DataFrame.from_csv(teamsheetpath + team_2 + '.csv')
    stats_1 = get_residual_performance(team_1)
    stats_2 = get_residual_performance(team_2)
    expected_scores_1 = get_expected_scores(stats_1, stats_2, team_1_season, team_2_season)
    expected_scores_2 = get_expected_scores(stats_2, stats_1, team_2_season, team_1_season)
    team_1_wins = 0
    team_2_wins = 0
    team_1_draws = 0
    team_2_draws = 0
    team_1_bpw = 0
    team_2_bpw = 0
    team_1_bpd = 0
    team_2_bpd = 0
    team_1_bpl = 0
    team_2_bpl = 0
    team_1_lbp = 0
    team_2_lbp = 0
    team_1_scores = []
    team_2_scores = []
    i = 0
    error = 1
    while error > 0.000001 or i < 5000000: #Run until convergence after 5 million iterations
        summary = game(team_1, team_2,
                       expected_scores_1, expected_scores_2,
                       playoff = po)
        team_1_prev_wins = team_1_wins
        team_1_wins += summary[team_1][0]
        team_2_wins += summary[team_2][0]
        team_1_draws += summary[team_1][1]
        team_2_draws += summary[team_2][1]
        team_1_scores.append(summary[team_1][2])
        team_2_scores.append(summary[team_2][2])
        team_1_bpw += summary[team_1][3]
        team_2_bpw += summary[team_2][3]
        team_1_bpd += summary[team_1][4]
        team_2_bpd += summary[team_2][4]
        team_1_bpl += summary[team_1][5]
        team_2_bpl += summary[team_2][5]
        team_1_lbp += summary[team_1][6]
        team_2_lbp += summary[team_2][6]
        team_1_prob = float(team_1_wins) / len(team_1_scores)
        team_2_prob = float(team_2_wins) / len(team_2_scores)
        team_1_bpw_prob = float(team_1_bpw) / len(team_1_scores)
        team_2_bpw_prob = float(team_2_bpw) / len(team_2_scores)
        team_1_bpd_prob = float(team_1_bpd) / len(team_1_scores)
        team_2_bpd_prob = float(team_2_bpd) / len(team_2_scores)
        team_1_bpl_prob = float(team_1_bpl) / len(team_1_scores)
        team_2_bpl_prob = float(team_2_bpl) / len(team_2_scores)
        team_1_lbp_prob = float(team_1_lbp) / len(team_1_scores)
        team_2_lbp_prob = float(team_2_lbp) / len(team_2_scores)
        if i > 0:
            team_1_prev_prob = float(team_1_prev_wins) / i
            error = team_1_prob - team_1_prev_prob
        i = i + 1
    if i == 5000000:
        print('Probability converged within 5 million iterations')
    else:
        print('Probability converged after ' + str(i) + ' iterations')
    games = pd.DataFrame.from_items([(team_1, team_1_scores), (team_2, team_2_scores)])
    summaries = games.describe(percentiles = np.linspace(0.05, 0.95, 19))
    summaries = summaries.reset_index()
    for item in summaries.index:
        try:
            summaries['index'][item] = str(int(float(summaries['index'][item][:-1]))) + '%'
        except ValueError:
            continue
    bonus_points = pd.DataFrame(index = ['4-Try Bonus Point with Win',
                                         '4-Try Bonus Point with Draw',
                                         '4-Try Bonus Point with Loss',
                                         'Losing Bonus Point'])
    bonus_points[team_1] = [team_1_bpw_prob, team_1_bpd_prob, team_1_bpl_prob, team_1_lbp_prob]
    bonus_points[team_2] = [team_2_bpw_prob, team_2_bpd_prob, team_2_bpl_prob, team_2_lbp_prob]
    summaries = summaries.set_index('index')
    summaries = summaries.groupby(level = 0).last()
    output = {'ProbWin': {team_1: team_1_prob, team_2: team_2_prob}, 'Scores': summaries, 'Bonus Points': bonus_points}

    print(team_1 + '/' + team_2 + ' score distributions computed in ' + str(round(time.time() - ts, 1)) + ' seconds')

    return output