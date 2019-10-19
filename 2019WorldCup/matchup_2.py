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

po = True

stadium_locs = pd.read_csv(os.path.join(os.path.split(__file__)[0], 'StadiumLocs.csv'), index_col = 0)
teamsheetpath = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'Score Tables')

compstat = {'TDF': 'TDA', 'TDA': 'TDF', #Dictionary to use to compare team stats with opponent stats
            'FGF': 'FGA', 'FGA': 'FGF',
            'SFF': 'SFA', 'SFA': 'SFF',
            'PAT1%F': 'PAT1%A', 'PAT1%A': 'PAT1%F',
            'PAT2%F': 'PAT2%A', 'PAT2%A': 'PAT2%F',
            'D2C%F': 'D2C%A', 'D2C%A': 'D2C%F'}

def weighted_variance(data, weights):
    assert len(data) == len(weights), 'Data and weights must be same length'
    weighted_average = np.average(data, weights = weights)
    v1 = weights.sum()
    v2 = np.square(weights).sum()
    return (weights*np.square(data - weighted_average)).sum() / (v1 - (v2/v1))

def geodesic_distance(olat, olng, dlat, dlng):
    '''
    Returns geodesic distance in percentage of half the earth's circumference between two points on the earth's surface

    Parameters
    ----------
    olat (float):
        Origin latitude
    olng (float):
        Origin longitude
    dlat (float):
        Destination latitude
    dlng (float):
        Destination longitude

    Returns
    -------
    distance (float):
        Distance between input points in proportion of half the earth's circumference (returns 1 if input points are antipodes)
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

    Parameters
    ----------
    venue (str):
        Reference venue
    home_lat (float):
        Team's home latitude
    home_lng (float):
        Team's home longitude
    reference_distance (float):
        Distance to compare travel distance to. Equal to 1 if the travel distance equals the reference distance

    Returns
    -------
    weight (float):
        Travel-based weight for averaging team statistics
    '''
    global stadium_locs
    (venue_lat, venue_lng) = stadium_locs.loc[venue, ['Lat', 'Long']]
    
    travel_distance = geodesic_distance(home_lat, home_lng, venue_lat, venue_lng)
    return 1 - abs(travel_distance - reference_distance)

def round_percentage(percentage):
    '''
    Removes `.0` from a string containing a percentage

    Parameters
    ----------
    percentage (str):
        Percentage to round

    Returns
    -------
    percentage (str):
        Rounded percentage
    '''
    return percentage.replace('.0', '')

def get_opponent_stats(opponent, venue):
    '''
    Calculates the statistics of each teams oppopnent

    Parameters
    ----------
    opponent (str):
        Week's opponent
    venue (str):
        Game venue. Must be index in StatiumLocs.csv

    Returns
    -------
    opponent_stats (dict):
        Dictionary of statistics
    '''
    opponent_stats = {}
    global teamsheetpath, stadium_locs
    opp_stats = pd.DataFrame.from_csv(os.path.join(teamsheetpath, opponent + '.csv'))

    #Compute distance weights for opponent
    (venue_lat, venue_lng) = stadium_locs.loc[venue, ['Lat', 'Long']]
    (opponent_home_lat, opponent_home_lng) = stadium_locs.loc[opponent, ['Lat', 'Long']]
    opponent_reference_distance = geodesic_distance(opponent_home_lat, opponent_home_lng, venue_lat, venue_lng)

    def get_opponent_weight(location):
        return get_travel_weight(location, opponent_home_lat, opponent_home_lng, opponent_reference_distance)

    opp_stats['Weight'] = opp_stats['VENUE'].apply(get_opponent_weight)

    #Compute easy to calculate stats
    for stat in opp_stats.columns:
        if stat in ['TDF', 'FGF', 'SFF', 'TDA', 'FGA', 'SFA']:
            opponent_stats[stat] = np.average(opp_stats[stat], weights = opp_stats['Weight'])

    #Calculate percentages. If unable, insert assumed values
    try:
        opponent_stats['PAT1%F'] = float((opp_stats['Weight']*opp_stats['PAT1FS']).sum()) / (opp_stats['Weight']*opp_stats['PAT1FA']).sum()
    except ZeroDivisionError:
        opponent_stats['PAT1%F'] = .942
    try:
        opponent_stats['PAT2%F'] = float((opp_stats['Weight']*opp_stats['PAT2FS']).sum()) / (opp_stats['Weight']*opp_stats['PAT2FA']).sum()
    except ZeroDivisionError:
        opponent_stats['PAT2%F'] = .479
    try:
        opponent_stats['PAT1%A'] = float((opp_stats['Weight']*opp_stats['PAT1AS']).sum()) / (opp_stats['Weight']*opp_stats['PAT1AA']).sum()
    except ZeroDivisionError:
        opponent_stats['PAT1%A'] = .942
    try:
        opponent_stats['PAT2%A'] = float((opp_stats['Weight']*opp_stats['PAT2AS']).sum()) / (opp_stats['Weight']*opp_stats['PAT2AA']).sum()
    except ZeroDivisionError:
        opponent_stats['PAT2%A'] = .479
    try:
        opponent_stats['D2C%F'] = float((opp_stats['Weight']*opp_stats['D2CF']).sum()) / ((opp_stats['Weight']*opp_stats['PAT1AA']).sum() + (opp_stats['Weight']*opp_stats['PAT2AA']).sum() - (opp_stats['Weight']*opp_stats['PAT1AS']).sum() - (opp_stats['Weight']*opp_stats['PAT2AS']).sum())
    except ZeroDivisionError:
        opponent_stats['D2C%F'] = 0.01
    try:
        opponent_stats['D2C%A'] = float(opp_stats['D2CA'].sum()) / ((opp_stats['Weight']*opp_stats['PAT1FA']).sum() + (opp_stats['Weight']*opp_stats['PAT2FA']).sum() - (opp_stats['Weight']*opp_stats['PAT1FS']).sum() - (opp_stats['Weight']*opp_stats['PAT2FS']).sum())
    except ZeroDivisionError:
        opponent_stats['D2C%A'] = 0.01
    return opponent_stats

def get_residual_performance(score_df):
    '''
    Compares teams weekly performances to their opponents averages and gets averages of residuals

    Parameters
    ----------
    team (str):
        Team to get residual stats of

    Returns
    -------
    residual_stats (dict):
        Dictionary of team's residual stats
    '''
    global teamsheetpath, stadium_locs
    #score_df = pd.DataFrame.from_csv(os.path.join(teamsheetpath, team + '.csv'))
    residual_stats = {}
    residual_variances = {}

    #Initialize percentages with null values
    score_df['PAT1%F'] = np.nan
    score_df['PAT2%F'] = np.nan
    score_df['PAT1%A'] = np.nan
    score_df['PAT2%A'] = np.nan
    score_df['D2C%F'] = np.nan
    score_df['D2C%A'] = np.nan

    #For each week, add in percentages. If there is no denominator, assume values. Then add fields containing opponent averages
    for week in score_df.index:
        try:
            score_df['PAT1%F'][week] = float(score_df['PAT1FS'][week]) / score_df['PAT1FA'][week]
        except ZeroDivisionError:
            score_df['PAT1%F'][week] = 0.942
        try:
            score_df['PAT2%F'][week] = float(score_df['PAT2FS'][week]) / score_df['PAT2FA'][week]
        except ZeroDivisionError:
            score_df['PAT2%F'][week] = 0.479
        try:
            score_df['PAT1%A'][week] = float(score_df['PAT1AS'][week]) / score_df['PAT1AA'][week]
        except ZeroDivisionError:
            score_df['PAT1%A'][week] = 0.942
        try:
            score_df['PAT2%A'][week] = float(score_df['PAT2AS'][week]) / score_df['PAT2AA'][week]
        except ZeroDivisionError:
            score_df['PAT2%A'][week] = 0.479
        try:
            score_df['D2C%F'][week] = float(score_df['D2CF'][week]) / (score_df['PAT1AA'][week] + score_df['PAT2AA'][week] - score_df['PAT1AS'][week] - score_df['PAT2AS'][week])
        except ZeroDivisionError:
            score_df['D2C%F'][week] = 0.01
        try:
            score_df['D2C%A'][week] = float(score_df['D2CA'][week]) / (score_df['PAT1FA'][week] + score_df['PAT2FA'][week] - score_df['PAT1FS'][week] - score_df['PAT2FS'][week])
        except ZeroDivisionError:
            score_df['D2C%A'][week] = 0.01

        #Read in opponent's stats
        opponent_stats = get_opponent_stats(score_df['OPP'][week], score_df['VENUE'][week])
        for stat in opponent_stats:
            try:
                score_df['OPP_' + stat][week] = opponent_stats[stat]
            except KeyError: #Create column if it's not there
                score_df['OPP_' + stat] = np.nan
                score_df['OPP_' + stat][week] = opponent_stats[stat]
            
    #Compute difference between team's statistics and their opponents averages         
    for stat in opponent_stats:

        if stat == 'Weight':
            continue

        score_df['R_' + stat] = score_df[stat] - score_df['OPP_' + compstat[stat]]
        if stat in ['TDF', 'FGF', 'SFF', 'TDA', 'FGA', 'SFA']:
            residual_stats[stat] = np.average(score_df['R_' + stat], weights = score_df['Weight'])
            residual_variances[stat] = weighted_variance(score_df['R_' + stat], score_df['Weight'])
        elif stat == 'PAT1%F':
            try:
                residual_stats[stat] = (score_df['R_PAT1%F'].multiply(score_df['PAT1FA'])*score_df['Weight']).sum() / (score_df['PAT1FA']*score_df['Weight']).sum()
            except ZeroDivisionError:
                residual_stats[stat] = 0.0
        elif stat == 'PAT2%F':
            try:
                residual_stats[stat] = (score_df['R_PAT2%F'].multiply(score_df['PAT2FA'])*score_df['Weight']).sum() / (score_df['PAT2FA']*score_df['Weight']).sum()
            except ZeroDivisionError:
                residual_stats[stat] = 0.0
        elif stat == 'PAT1%A':
            try:
                residual_stats[stat] = (score_df['R_PAT1%A'].multiply(score_df['PAT1AA'])*score_df['Weight']).sum() / (score_df['PAT1AA']*score_df['Weight']).sum()
            except ZeroDivisionError:
                residual_stats[stat] = 0.0
        elif stat == 'PAT2%A':
            try:
                residual_stats[stat] = (score_df['R_PAT2%A'].multiply(score_df['PAT2AA'])*score_df['Weight']).sum() / (score_df['PAT2AA']*score_df['Weight']).sum()
            except ZeroDivisionError:
                residual_stats[stat] = 0.0
        elif stat == 'D2C%F':
            try:
                residual_stats[stat] = (score_df['R_D2C%F'].multiply(score_df['PAT1AA'] + score_df['PAT2AA'] - score_df['PAT1AS'] - score_df['PAT2AS'])*score_df['Weight']).sum() / ((score_df['PAT1AA'] + score_df['PAT2AA'] - score_df['PAT1AS'] - score_df['PAT2AS'])*score_df['Weight']).sum()
            except ZeroDivisionError:
                residual_stats[stat] = 0.0
        elif stat == 'D2C%A':
            try:
                residual_stats[stat] = (score_df['R_D2C%A'].multiply(score_df['PAT1FA'] + score_df['PAT2FA'] - score_df['PAT1FS'] - score_df['PAT2FS'])*score_df['Weight']).sum() / ((score_df['PAT1FA'] + score_df['PAT2FA'] - score_df['PAT1FS'] - score_df['PAT2FS'])*score_df['Weight']).sum()
            except ZeroDivisionError:
                residual_stats[stat] = 0.0
        
        try:
            residual_stats['GOFOR2'] = float(score_df['PAT2FA'].sum()) / score_df['TDF'].sum()
        except ZeroDivisionError:
            residual_stats['GOFOR2'] = .1
    
    #print(residual_stats)
    #print(residual_variances)
    return residual_stats, pd.Series(residual_variances)

def get_expected_scores(team_1_stats, team_2_stats, team_1_df, team_2_df):
    '''
    Gets expected values for number of touchdowns, field goals, and safeties, as well as probabilities of going for and making PATs

    Parameters
    ----------
    team_1_stats (dict):
        Dictionary of team 1's residual statistics
    team_2_stats (dict):
        Dictionary of team 2's residual statistics
    team_1_df (dict):
        Team 1's weekly results
    team_2_df (dict):
        Team 2's weekly results

    Returns
    -------
    expected_scores (dict):
        Team 1's expected number of touchdowns, field goals, safeties, and PAT percentages
    '''
    expected_scores = {}
    for stat in team_1_stats:
        expected_scores['TD'] = mean([team_1_stats['TDF'] + np.average(team_2_df['TDA'], weights = team_2_df['Weight']),
                                      team_2_stats['TDA'] + np.average(team_1_df['TDF'], weights = team_1_df['Weight'])])
        expected_scores['FG'] = mean([team_1_stats['FGF'] + np.average(team_2_df['FGA'], weights = team_2_df['Weight']),
                                      team_2_stats['FGA'] + np.average(team_1_df['FGF'], weights = team_1_df['Weight'])])
        expected_scores['S'] = mean([team_1_stats['SFF'] + np.average(team_2_df['SFA'], weights = team_2_df['Weight']),
                                     team_2_stats['SFA'] + np.average(team_1_df['SFF'], weights = team_1_df['Weight'])])
        expected_scores['S'] = max(expected_scores['S'], 0.01)
        expected_scores['GOFOR2'] = team_1_stats['GOFOR2']
        try:
            pat1prob = mean([team_1_stats['PAT1%F'] + (team_2_df['Weight']*team_2_df['PAT1AS']).astype('float').sum() / (team_2_df['Weight']*team_2_df['PAT1AA']).sum(),
                             team_2_stats['PAT1%A'] + (team_1_df['Weight']*team_1_df['PAT1FS']).astype('float').sum() / (team_1_df['Weight']*team_1_df['PAT1FA']).sum()])
            #print(team_1_stats['PAT1%F'], team_2_stats['PAT1%A'])
            #print((team_2_df['Weight']*team_2_df['PAT1AS']).astype('float').sum() / (team_2_df['Weight']*team_2_df['PAT1AA']).sum())
            #print((team_1_df['Weight']*team_1_df['PAT1FS']).astype('float').sum() / (team_1_df['Weight']*team_1_df['PAT1FA']).sum())
            #print('\n')

        except ZeroDivisionError:
            pat1prob = np.nan
        if not math.isnan(pat1prob):
            expected_scores['PAT1PROB'] = min(pat1prob, 0.99)
        else:
            expected_scores['PAT1PROB'] = 0.942
        #expected_scores['PAT1PROB'] = max(expected_scores['PAT1PROB'], 0.99)
        
        try:
            pat2prob = mean([team_1_stats['PAT2%F'] + (team_2_df['Weight']*team_2_df['PAT2AS']).astype('float').sum() / (team_2_df['Weight']*team_2_df['PAT2AA']).sum(),
                             team_2_stats['PAT2%A'] + (team_1_df['Weight']*team_1_df['PAT2FS']).astype('float').sum() / (team_1_df['Weight']*team_1_df['PAT2FA']).sum()])
        except ZeroDivisionError:
            pat2prob = np.nan
        if not math.isnan(pat2prob):
            expected_scores.update({'PAT2PROB': min(max(pat2prob, 0), 1)})
        else:
            expected_scores.update({'PAT2PROB': 0.479})

        try:
            d2cprob = mean([team_2_stats['D2C%F'] + (team_1_df['Weight']*team_1_df['D2CA']).astype('float').sum() / (team_1_df['Weight']*(team_1_df['PAT1AA'] + team_1_df['PAT2AA'] - team_1_df['PAT1AS'] - team_1_df['PAT2AS'])).sum(),
                            team_1_stats['D2C%A'] + (team_2_df['Weight']*team_2_df['D2CF']).astype('float').sum() / (team_2_df['Weight']*(team_2_df['PAT1FA'] + team_2_df['PAT2FA'] - team_2_df['PAT1FS'] - team_2_df['PAT2FS'])).sum()])
        except ZeroDivisionError:
            d2cprob = np.nan
        if not math.isnan(d2cprob):
            expected_scores['D2CPROB'] = max(d2cprob, 0)
        else:
            expected_scores['D2CPROB'] = 0.01 
    
    return expected_scores

#def get_score(expected_scores):
#    '''
#    Obtains the score based on random simulation using the expected scores as expected values

#    Parameters
#    ----------
#    expected scores (dict):
#        Expected number of touchdowns, field goals, safeties, and PAT percentages

#    Returns
#    -------
#    score (int):
#        Score
#    '''
#    #Add contribution of touchdowns, field goals, and safeties
#    score = 0
#    if expected_scores['TD'] > 0:
#        tds = poisson(expected_scores['TD'])
#    else:
#        tds = poisson(0.01) #Filler so it's not zero every time
#    score = score + 6 * tds
#    if expected_scores['FG'] > 0:
#        fgs = poisson(expected_scores['FG'])
#    else:
#        fgs = poisson(0.01)
#    score = score + 3 * fgs
#    if expected_scores['S'] > 0:
#        sfs = poisson(expected_scores['S'])
#    else:
#        sfs = poisson(0.01)
#    score = score + 2 * sfs

#    d2c = 0
#    #Add PATs
#    for td in range(tds):
#        go_for_2_determinant = uniform(0, 1)
#        if go_for_2_determinant <= expected_scores['GOFOR2']: #Going for 2
#            successful_pat_determinant = uniform(0, 1)
#            if successful_pat_determinant <= expected_scores['PAT2PROB']:
#                score = score + 2
#            else:
#                d2c_determinant = uniform(0, 1)
#                if d2c_determinant <= expected_scores['D2CPROB']:
#                    d2c += 1
#                else:
#                    continue

#        else: #Going for 1
#            successful_pat_determinant = uniform(0, 1)
#            if successful_pat_determinant <= expected_scores['PAT1PROB']:
#                score = score + 1
#            else:
#                d2c_determinant = uniform(0, 1)
#                if d2c_determinant <= expected_scores['D2CPROB']:
#                    d2c += 1
#                else:
#                    continue
#    return score, d2c

def game(team_1, team_2,
         expected_scores_1, expected_scores_2):
    '''
    Simulation of a single game between two teams

    Parameters
    ----------
    team_1 (str):
        Initials of team 1
    team_2 (str):
        Initials of team 2
    expected_scores_1 (dict):
        Team 1's expected scores
    expected_scores_2 (dict):
        Team 2's expected scores

    Returns
    -------
    Summary (dict):
        Summary of game's results
    '''
    (score_1, d2c2) = get_score(expected_scores_1)
    (score_2, d2c1) = get_score(expected_scores_2)
    score_1 += 2*d2c1
    score_2 += 2*d2c2

    if score_1 > score_2: #Give team 1 a win if their score is higher
        win_1 = 1.
        win_2 = 0.
    elif score_2 > score_1: #Give team 2 a win if their score is higher
        win_1 = 0.
        win_2 = 1.
    else: #If the scores are the same, give both teams a half win
        win_1 = 0.5
        win_2 = 0.5

    summary = {team_1: [win_1, score_1],
               team_2: [win_2, score_2]}

    return summary

def get_score(expected_scores, score_array, n_sim, return_tries = False):
    tdf = sim_util.sim(expected_scores['TD'][0], expected_scores['TD'][1], n_sim)
    pat2a = np.random.binomial(tdf, expected_scores['GoFor2'])
    pat1a = tdf - pat2a
    pat1f = np.random.binomial(pat1a, expected_scores['PAT1'])
    pat2f = np.random.binomial(pat2a, expected_scores['PAT2'])
    patfail = tdf - pat1f - pat2f #Unsuccessful PATs
    fgf = sim_util.sim(expected_scores['FG'][0], expected_scores['FG'][1], n_sim)
    sf = sim_util.sim(expected_scores['S'][0], expected_scores['S'][1], n_sim)

    score = sim_util.calculate_score((tdf, pat1f, pat2f, fgf, sf), score_array)

    return score, patfail
            
def matchup(team_1, team_2, venue = None):
    '''
    The main script. Simulates a matchup between two teams at least 5 million times

    Parameters
    ----------
    team_1 (str):
        The first team's initials
    team_2 (str):
        The second team's initials
    venue (str, optional):
        Code for venue. If not set it will be team_1

    Returns
    -------
    output (dict): #CreativeName
        Dictionary containing chances of winning and score distribution characteristics
    '''
    #If no venue is specified, set venue to Team 1's location
    if venue == None:
        venue = team_1

    #Get reference distance for calculation of each team's geographic weights in averaging
    (venue_lat, venue_lng) = stadium_locs.loc[venue, ['Lat', 'Long']]
    (team_1_home_lat, team_1_home_lng) = stadium_locs.loc[team_1, ['Lat', 'Long']]
    (team_2_home_lat, team_2_home_lng) = stadium_locs.loc[team_2, ['Lat', 'Long']]
    team_1_reference_distance = geodesic_distance(team_1_home_lat, team_1_home_lng, venue_lat, venue_lng)
    team_2_reference_distance = geodesic_distance(team_2_home_lat, team_2_home_lng, venue_lat, venue_lng)

    def get_team_1_weight(location):
        return get_travel_weight(location, team_1_home_lat, team_1_home_lng, team_1_reference_distance)

    def get_team_2_weight(location):
        return get_travel_weight(location, team_2_home_lat, team_2_home_lng, team_2_reference_distance)

    #Read in teams' performances and calculate expected scores based on them
    ts = time.time()
    team_1_season = pd.DataFrame.from_csv(os.path.join(teamsheetpath, team_1 + '.csv'))
    team_2_season = pd.DataFrame.from_csv(os.path.join(teamsheetpath, team_2 + '.csv'))
    team_1_season['Weight'] = team_1_season['VENUE'].apply(get_team_1_weight)
    team_2_season['Weight'] = team_2_season['VENUE'].apply(get_team_2_weight)
    stats_1, variances_1 = get_residual_performance(team_1_season)
    stats_2, variances_2 = get_residual_performance(team_2_season)
    expected_scores_1 = get_expected_scores(stats_1, stats_2, team_1_season, team_2_season)
    expected_scores_2 = get_expected_scores(stats_2, stats_1, team_2_season, team_1_season)
    var_1 = pd.Series(0.25*(variances_1.loc[['TDF', 'FGF', 'SFF']].values + variances_2.loc[['TDA', 'FGA', 'SFA']].values), ['TD', 'FG', 'S'])
    var_2 = pd.Series(0.25*(variances_2.loc[['TDF', 'FGF', 'SFF']].values + variances_1.loc[['TDA', 'FGA', 'SFA']].values), ['TD', 'FG', 'S'])

    #print(var_1)
    #print(var_2)

    score_array = [6, 1, 2, 3, 2]
    n_sim = int(5e6)

    expected_scores_1a = {'TD': (expected_scores_1['TD'], var_1['TD']),
                          'GoFor2': expected_scores_1['GOFOR2'],
                          'PAT1': expected_scores_1['PAT1PROB'],
                          'PAT2': expected_scores_1['PAT2PROB'],
                          'D2C': expected_scores_1['D2CPROB'],
                          'FG': (expected_scores_1['FG'], var_1['FG']),
                          'S': (expected_scores_1['S'], var_1['S'])}
    expected_scores_2a = {'TD': (expected_scores_2['TD'], var_2['TD']),
                          'GoFor2': expected_scores_2['GOFOR2'],
                          'PAT1': expected_scores_2['PAT1PROB'],
                          'PAT2': expected_scores_2['PAT2PROB'],
                          'D2C': expected_scores_2['D2CPROB'],
                          'FG': (expected_scores_2['FG'], var_2['FG']),
                          'S': (expected_scores_2['S'], var_2['S'])}

    

    #print(expected_scores_1)
    #print(expected_scores_2)

    ts = time.time()

    (team_1_scores, team_1_patfail) = get_score(expected_scores_1a, score_array, n_sim, True)
    (team_2_scores, team_2_patfail) = get_score(expected_scores_2a, score_array, n_sim, True)

    team_1_scores += 2*np.random.binomial(team_2_patfail, expected_scores_1a['D2C'])
    team_2_scores += 2*np.random.binomial(team_1_patfail, expected_scores_2a['D2C'])

    te = time.time()
    print(te - ts)

    (team_1_wins, team_2_wins) = sim_util.eval_results(team_1_scores, team_2_scores, True)
    #team_1_wins += 0.5*draws
    #team_2_wins += 0.5*draws

    #print(np.round(team_1_wins.mean(), 4), round(team_1_scores.mean(), 1))
    #print(np.round(team_2_wins.mean(), 4), round(team_2_scores.mean(), 1))

    team_1_prob = team_1_wins.mean()
    team_2_prob = team_2_wins.mean()

    ##Initialize with no wins or scores for each team
    #team_1_wins = 0
    #team_2_wins = 0
    #team_1_draws = 0
    #team_2_draws = 0
    #team_1_scores = []
    #team_2_scores = []

    ##Iterate at least five million times, and then check to see if probabilities of win converge
    #i = 0
    #error = 1
    #min_iter = int(5e6)
    #while error > 0.000001 or i < min_iter: #Run until convergence after 5 million iterations
    #    summary = game(team_1, team_2,
    #                   expected_scores_1, expected_scores_2)

    #    team_1_prev_wins = team_1_wins
    #    team_1_wins += summary[team_1][0]
    #    team_2_wins += summary[team_2][0]
    #    team_1_scores.append(summary[team_1][1])
    #    team_2_scores.append(summary[team_2][1])
    #    team_1_prob = float(team_1_wins) / len(team_1_scores)
    #    team_2_prob = float(team_2_wins) / len(team_2_scores)

    #    #Compute convergence statistic after minimum iterations
    #    if i >= min_iter:
    #        team_1_prev_prob = float(team_1_prev_wins) / i
    #        error = team_1_prob - team_1_prev_prob
    #    i = i + 1
    #if i == min_iter:
    #    print('Probability converged within %d iterations'%(min_iter))
    #else:
    #    print('Probability converged after ' + str(i) + ' iterations')

    games = pd.DataFrame.from_items([(team_1, team_1_scores), (team_2, team_2_scores)])
    summaries = games.describe(percentiles = np.arange(0.05, 1, 0.05))

    #Remove decimal points from summary indices
    summaries = summaries.reset_index()
    summaries['index'] = summaries['index'].apply(round_percentage)
    summaries = summaries.set_index('index')

    output = {'ProbWin': {team_1: team_1_prob, team_2: team_2_prob}, 'Scores': summaries}

    print(team_1 + '/' + team_2 + ' score distributions computed in {0} seconds'.format(round(time.time() - ts, 1)))

    return output

if __name__ == '__main__':
    forecast = matchup('NE', 'LAR', 'ATL')
    print(forecast)