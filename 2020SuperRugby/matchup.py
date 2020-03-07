import os
os.chdir(os.path.dirname(__file__))

import sim_util
import sys
import pandas as pd
import numpy as np
from numpy.random import poisson, uniform
from numpy import mean
import time
import math

po = False

team_homes = pd.read_csv(os.path.join(os.path.split(__file__)[0], 'TeamHomes.csv'), header = None, index_col = 0)
stadium_locs = pd.read_csv(os.path.join(os.path.split(__file__)[0], 'StadiumLocs.csv'), index_col = 0)

teamsheetpath = os.path.join(os.path.split(__file__)[0], 'Score Tables')

compstat = {'TF': 'TA', 'TA': 'TF', #Dictionary to use to compare team stats with opponent stats
            'CF': 'CA', 'CA': 'CF',
            'CON%F': 'CON%A', 'CON%A': 'CON%F',
            'PF': 'PA', 'PA': 'PF',
            'DGF': 'DGA', 'DGA': 'DGF'}

def weighted_variance(data, weights):
    assert len(data) == len(weights), 'Data and weights must be same length'
    weighted_average = np.average(data, weights = weights)
    v1 = weights.sum()
    v2 = np.square(weights).sum()
    return (weights*np.square(data - weighted_average)).sum() / (v1 - (v2/v1))

def get_opponent_stats(opponent, venue): #Gets summaries of statistics for opponent each week
    opponent_stats = {}
    global teamsheetpath, stadium_locs, team_homes
    opp_stats = pd.DataFrame.from_csv(os.path.join(teamsheetpath, opponent + '.csv'))

    opponent_home = team_homes[1][opponent]
    (venue_lat, venue_lng) = stadium_locs.loc[venue, ['Lat', 'Long']]
    (opponent_home_lat, opponent_home_lng) = stadium_locs.loc[opponent_home, ['Lat', 'Long']]
    opponent_reference_distance = geodesic_distance(opponent_home_lat, opponent_home_lng, venue_lat, venue_lng)

    def get_opponent_weight(location):
        return get_travel_weight(location, opponent_home_lat, opponent_home_lng, opponent_reference_distance)

    opp_stats['Weight'] = opp_stats['VENUE'].apply(get_opponent_weight)

    for stat in opp_stats.columns:
        if stat != 'VENUE':
            if stat != 'OPP':
                opponent_stats.update({stat: np.average(opp_stats[stat], weights = opp_stats['Weight'])})
    try:
        opponent_stats.update({'CON%F': float((opp_stats['CF']*opp_stats['Weight']).sum())/(opp_stats['TF']*opp_stats['Weight']).sum()})
    except ZeroDivisionError:
        opponent_stats.update({'CON%F': 0.75})
    try:
        opponent_stats.update({'CON%A': float((opp_stats['CA']*opp_stats['Weight']).sum())/(opp_stats['TA']*opp_stats['Weight']).sum()})
    except ZeroDivisionError:
        opponent_stats.update({'CON%A': 0.75})
    return opponent_stats

def get_residual_performance(score_df): #Get how each team has done compared to the average performance of their opponents
    global teamsheetpath, team_homes, stadium_locs
    #score_df = pd.DataFrame.from_csv(os.path.join(teamsheetpath, team + '.csv'))
    residual_stats = {}
    residual_variances = {}

    score_df['CON%F'] = np.nan
    score_df['CON%A'] = np.nan
    for week in score_df.index:
        opponent_stats = get_opponent_stats(score_df['OPP'][week], score_df['VENUE'][week])
        for stat in opponent_stats:
            if week == score_df.index.tolist()[0]:
                score_df['OPP_' + stat] = np.nan       
            score_df['OPP_' + stat][week] = opponent_stats[stat]
        score_df['CON%F'][week] = float(score_df['CF'][week]) / score_df['TF'][week]
        score_df['CON%A'][week] = float(score_df['CA'][week]) / score_df['TA'][week]

    for stat in opponent_stats:
        
        if stat == 'Weight':
            continue

        score_df['R_' + stat] = score_df[stat] - score_df['OPP_' + compstat[stat]]
        if stat in ['TF', 'PF', 'DGF', 'TA', 'PA', 'DGA']:
            residual_stats.update({stat: np.average(score_df['R_' + stat], weights = score_df['Weight'])})
            residual_variances[stat] = weighted_variance(score_df['R_' + stat], score_df['Weight'])
        elif stat == 'CON%F':
            try:
                residual_stats.update({stat: (score_df['R_CON%F'].multiply(score_df['TF'])*score_df['Weight']).sum() / (score_df['TF']*score_df['Weight']).sum()})
            except ZeroDivisionError:
                residual_stats.update({stat: 0})
        elif stat == 'CON%A':
            try:
                residual_stats.update({stat: (score_df['R_CON%A'].multiply(score_df['TA'])*score_df['Weight']).sum() / (score_df['TA']*score_df['Weight']).sum()})
            except ZeroDisivionError:
                residual_stats.update({stat: 0})
    return residual_stats, pd.Series(residual_variances)

#def get_score(expected_scores): #Get the score for a team based on expected scores
#    score = 0
#    if expected_scores['T'] > 0:
#        tries = poisson(expected_scores['T'])
#    else:
#        tries = poisson(0.01)
#    score = score + 6 * tries
#    if expected_scores['P'] > 0:
#        fgs = poisson(expected_scores['P'])
#    else:
#        fgs = poisson(0.01)
#    score = score + 3 * fgs
#    if expected_scores['DG'] > 0:
#        sfs = poisson(expected_scores['DG'])
#    else:
#        sfs = poisson(0.01)
#    score = score + 2 * sfs
#    for t in range(tries):
#        successful_con_determinant = uniform(0, 1)
#        if successful_con_determinant <= expected_scores['CONPROB']:
#            score += 2
#        else:
#            continue
#    #if tries >= 4:
#    #    bp = True
#    #else:
#    #    bp = False
#    return (score, tries)

#def game(team_1, team_2,
#         expected_scores_1, expected_scores_2,
#         playoff = False): #Get two scores and determine a winner
#    (score_1, tries_1) = get_score(expected_scores_1)
#    (score_2, tries_2) = get_score(expected_scores_2)

#    if tries_1 - tries_2 >= 3:
#        bp1 = True
#        bp2 = False
#    elif tries_2 - tries_1 >= 3:
#        bp1 = False
#        bp2 = True
#    else:
#        bp1 = False
#        bp2 = False

#    if score_1 > score_2:
#        win_1 = 1
#        win_2 = 0
#        draw_1 = 0
#        draw_2 = 0
#        if bp1:
#            bpw1 = 1
#        else:
#            bpw1 = 0
#        if bp2:
#            bpl2 = 1
#        else:
#            bpl2 = 0
#        bpl1 = 0
#        bpw2 = 0
#        bpd1 = 0
#        bpd2 = 0
#        lbp1 = 0
#        if score_1 - score_2 <= 7:
#            lbp2 = 1
#        else:
#            lbp2 = 0

#    elif score_2 > score_1:
#        win_1 = 0
#        win_2 = 1
#        draw_1 = 0
#        draw_2 = 0
#        if bp1:
#            bpl1 = 1
#        else:
#            bpl1 = 0
#        if bp2:
#            bpw2 = 1
#        else:
#            bpw2 = 0
#        bpw1 = 0
#        bpl2 = 0
#        bpd1 = 0
#        bpd2 = 0
#        lbp2 = 0
#        if score_2 - score_1 <= 7:
#            lbp1 = 1
#        else:
#            lbp1 = 0
#    else:
#        if playoff:
#            win_1 = 0.5
#            win_2 = 0.5
#            draw_1 = 0
#            draw_2 = 0
#            bpw1 = 0
#            bpw2 = 0
#            bpd1 = 0
#            bpd2 = 0
#            bpl1 = 0
#            bpl2 = 0
#            lbp1 = 0
#            lbp2 = 0
#        else:
#            win_1 = 0
#            win_2 = 0
#            draw_1 = 1
#            draw_2 = 1
#            bpw1 = 0
#            bpw2 = 0
#            bpl1 = 0
#            bpl2 = 0
#            lbp1 = 0
#            lbp2 = 0
#        if bp1:
#            bpd1 = 1
#        else:
#            bpd1 = 0
#        if bp2:
#            bpd2 = 1
#        else:
#            bpd2 = 0
#    summary = {team_1: [win_1, draw_1, score_1, bpw1, bpd1, bpl1, lbp1]}
#    summary.update({team_2: [win_2, draw_2, score_2, bpw2, bpd2, bpl2, lbp2]})
#    return summary

def get_expected_scores(team_1_stats, team_2_stats, team_1_df, team_2_df): #Get the expected scores for a matchup based on the previous teams' performances
    expected_scores = {}
    #print('\n')
    #print('Residual Stats')
    #print(team_1_stats)
    #print(team_2_stats)
    #print('\n')
    for stat in team_1_stats:
        expected_scores.update({'T': mean([team_1_stats['TF'] + np.average(team_2_df['TA'], weights = team_2_df['Weight']),
                                           team_2_stats['TA'] + np.average(team_1_df['TF'], weights = team_1_df['Weight'])])})
        expected_scores.update({'P': mean([team_1_stats['PF'] + np.average(team_2_df['PA'], weights = team_2_df['Weight']),
                                           team_2_stats['PA'] + np.average(team_1_df['PF'], weights = team_1_df['Weight'])])})
        expected_scores.update({'DG': mean([team_1_stats['DGF'] + np.average(team_2_df['DGA'], weights = team_2_df['Weight']),
                                            team_2_stats['DGA'] + np.average(team_1_df['DGF'], weights = team_1_df['Weight'])])})
        #expected_scores['T'] = max(expected_scores['T'], 0)
        #expected_scores['P'] = max(expected_scores['P'], 0)
        expected_scores['DG'] = max(expected_scores['DG'], 0)
        #print mean([team_1_stats['PAT1%F'] + team_2_df['PAT1AS'].astype('float').sum() / team_2_df['PAT1AA'].sum(),
        #       team_2_stats['PAT1%A'] + team_1_df['PAT1FS'].astype('float').sum() / team_1_df['PAT1FA'].sum()])
        try:
            conprob = mean([team_1_stats['CON%F'] + (team_2_df['CA']*team_2_df['Weight']).sum() / (team_2_df['TA']*team_2_df['Weight']).sum(),
                            team_2_stats['CON%A'] + (team_1_df['CF']*team_1_df['Weight']).sum() / (team_1_df['TF']*team_1_df['Weight']).sum()])
        except ZeroDivisionError:
            conprob = 0.75
        if not math.isnan(conprob):
            conprob = min(max(conprob, 0.01), 0.99)
            expected_scores.update({'CONPROB': conprob})
        else:
            expected_scores.update({'CONPROB': 0.75})
        #print(expected_scores['PAT1PROB'])
    #print(expected_scores)
    return expected_scores

def geodesic_distance(olat, olng, dlat, dlng):
    '''
    Returns geodesic distance in percentage of half the earth's circumference between two points on the earth's surface
    '''
    scale = math.tau/360
    olat *= scale
    olng *= scale
    dlat *= scale
    dlng *= scale

    delta_lat = (dlat - olat)
    delta_lng = (dlng - olng)

    a = math.sin(delta_lat/2)**2 + math.cos(olat)*math.cos(dlat)*math.sin(delta_lng/2)**2
    return 4*math.atan2(math.sqrt(a), math.sqrt(1-a))/math.tau

def get_travel_weight(venue, home_lat, home_lng, reference_distance):
    '''
    Gets the travel weight based on a venue, a team's home lat/long coordinates, and a reference distance
    '''
    global stadium_locs
    (venue_lat, venue_lng) = stadium_locs.loc[venue, ['Lat', 'Long']]
    
    travel_distance = geodesic_distance(home_lat, home_lng, venue_lat, venue_lng)
    return 1 - abs(travel_distance - reference_distance)
    
def get_score(expected_scores, score_array, n_sim, return_tries = True):
    tf = sim_util.sim(expected_scores['T'][0], expected_scores['T'][1], n_sim)
    cf = np.random.binomial(tf, expected_scores['C'])
    pf = sim_util.sim(expected_scores['P'][0], expected_scores['P'][1], n_sim)
    dgf = sim_util.sim(expected_scores['DG'][0], expected_scores['DG'][1], n_sim)

    score = sim_util.calculate_score((tf, cf, pf, dgf), score_array)

    if return_tries:
        return score, tf
    else:
        return score

def matchup(team_1, team_2, venue = None):
    ts = time.time()
    global team_homes, stadium_locs

    team_1_home = team_homes[1][team_1]
    team_2_home = team_homes[1][team_2]

    if venue is None:
        venue = team_homes[1][team_1]

    (venue_lat, venue_lng) = stadium_locs.loc[venue, ['Lat', 'Long']]
    (team_1_home_lat, team_1_home_lng) = stadium_locs.loc[team_1_home, ['Lat', 'Long']]
    (team_2_home_lat, team_2_home_lng) = stadium_locs.loc[team_2_home, ['Lat', 'Long']]
    team_1_reference_distance = geodesic_distance(team_1_home_lat, team_1_home_lng, venue_lat, venue_lng)
    team_2_reference_distance = geodesic_distance(team_2_home_lat, team_2_home_lng, venue_lat, venue_lng)

    def get_team_1_weight(location):
        return get_travel_weight(location, team_1_home_lat, team_1_home_lng, team_1_reference_distance)

    def get_team_2_weight(location):
        return get_travel_weight(location, team_2_home_lat, team_2_home_lng, team_2_reference_distance)

    team_1_season = pd.DataFrame.from_csv(os.path.join(teamsheetpath, team_1 + '.csv'))
    team_2_season = pd.DataFrame.from_csv(os.path.join(teamsheetpath, team_2 + '.csv'))
    team_1_season['Weight'] = team_1_season['VENUE'].apply(get_team_1_weight)
    team_2_season['Weight'] = team_2_season['VENUE'].apply(get_team_2_weight)
    stats_1, variances_1 = get_residual_performance(team_1_season)
    stats_2, variances_2 = get_residual_performance(team_2_season)
    expected_scores_1 = get_expected_scores(stats_1, stats_2, team_1_season, team_2_season)
    expected_scores_2 = get_expected_scores(stats_2, stats_1, team_2_season, team_1_season)
    var_1 = pd.Series(0.25*(variances_1.loc[['TF', 'PF', 'DF']].values + variances_2.loc[['TA', 'PA', 'DGA']].values), ['T', 'P', 'DG'])
    var_2 = pd.Series(0.25*(variances_2.loc[['TF', 'PF', 'DF']].values + variances_1.loc[['TA', 'PA', 'DGA']].values), ['T', 'P', 'DG'])

    for stat in var_1.index:
        if math.isnan(var_1[stat]):
            var_1[stat] = expected_scores_1[stat]
        if math.isnan(var_2[stat]):
            var_2[stat] = expected_scores_2[stat]

    score_array = [5, 2, 3, 3]
    n_sim = int(5e6)

    expected_scores_1a = {'T': (expected_scores_1['T'], var_1['T']),
                          'C': expected_scores_1['CONPROB'],
                          'P': (expected_scores_1['P'], var_1['P']),
                          'DG': (expected_scores_1['DG'], var_1['DG'])}
    expected_scores_2a = {'T': (expected_scores_2['T'], var_2['T']),
                          'C': expected_scores_2['CONPROB'],
                          'P': (expected_scores_2['P'], var_2['P']),
                          'DG': (expected_scores_2['DG'], var_2['DG'])}

    print(expected_scores_1a)
    print(expected_scores_2a)

    ts = time.time()

    (team_1_scores, team_1_tries) = get_score(expected_scores_1a, score_array, n_sim)
    (team_2_scores, team_2_tries) = get_score(expected_scores_2a, score_array, n_sim)

    te = time.time()

    print(te - ts)

    (team_1_wins, team_2_wins, draws) = sim_util.eval_results(team_1_scores, team_2_scores, po)
    (team_1_tb, team_2_tb) = sim_util.eval_try_bonus(team_1_tries, team_2_tries, 3)
    (team_1_lb, team_2_lb) = sim_util.eval_losing_bonus(team_1_scores, team_2_scores, 7)

    team_1_prob = team_1_wins.mean()
    team_2_prob = team_2_wins.mean()
    draw_prob = draws.mean()
    
    team_1_bpw_prob = (team_1_tb * team_1_wins).mean()
    team_1_bpd_prob = (team_1_tb * draws).mean()
    team_1_bpl_prob = (team_1_tb * team_2_wins).mean()
    team_1_lbp_prob = (team_1_lb).mean()

    team_2_bpw_prob = (team_2_tb * team_2_wins).mean()
    team_2_bpd_prob = (team_2_tb * draws).mean()
    team_2_bpl_prob = (team_2_tb * team_1_wins).mean()
    team_2_lbp_prob = (team_2_lb).mean()

    games = pd.DataFrame.from_items([(team_1, team_1_scores), (team_2, team_2_scores)])
    pre_summaries = games.describe(percentiles = list(np.linspace(0.05, 0.95, 19)))

    summaries = pd.DataFrame(columns = pre_summaries.columns)
    summaries.loc['mean'] = pre_summaries.loc['mean']

    for i in pre_summaries.index:
        try:
            percentile = int(round(float(i[:-1])))
            summaries.loc['{}%'.format(percentile)] = pre_summaries.loc[i]
        except ValueError:
            continue

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