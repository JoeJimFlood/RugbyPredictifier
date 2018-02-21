import pandas as pd
import numpy as np
import os
import sys

def bin_scores(score, bins):
    if score <= bins[0]:
        return [1, 0, 0, 0]
    elif score <= bins[1]:
        return [0, 1, 0, 0]
    elif score <= bins[2]:
        return [0, 0, 1, 0]
    else:
        return [0, 0, 0, 1]

base_path = os.path.split(__file__)[0]
source_path = os.path.split(base_path)[0]
sys.path.append(base_path)
validation_file = os.path.join(base_path, 'ValidationData.xlsx')

import matchup as m0
import matchup_w_distance as m1

rounds = list(range(8, 18)) + ['QF', 'SF', 'F']
#rounds = [17, 'QF', 'SF', 'F']

outdata0 = pd.DataFrame(columns = ['ROUND', 'TEAM', 'OPP', 'VENUE', 'SCORE',
                                   'EXPECTED', 'Q1', 'Q2', 'Q3',
                                   'INQ1', 'INQ2', 'INQ3', 'INQ4'])
outdata1 = outdata0.copy()

counter = 0

for round in rounds:

    print('ROUND {}'.format(round))

    validation_data = pd.read_excel(validation_file, 'Round{}'.format(round))

    for f in os.listdir(os.path.join(source_path, 'teamcsvs')):
        data = pd.read_csv(os.path.join(source_path, 'teamcsvs', f))
        data['ROUND'] = data['ROUND'].astype(str)
        try:
            data.set_index('ROUND').loc[:str(round)].iloc[:-1].to_csv(os.path.join(base_path, 'teamcsvs', f))
        except KeyError:
            try:
                last_round = round - 1
            except TypeError:
                guide = {'QF': 17, 'SF': 'QF', 'F': 'SF'}
                last_round = guide[round]
                if last_round in data.index:
                    data.set_index('ROUND').loc[:str(last_round)].to_csv(os.path.join(base_path, 'teamcsvs', f))
                else:
                    pass

    for matchupno in validation_data.index:

        (home_team, away_team, venue, hscore, ascore) = validation_data.loc[matchupno]

        results0 = m0.matchup(home_team, away_team)
        results1 = m1.matchup(home_team, away_team, venue)

        hexpected0 = results0['Scores'][home_team]['mean']
        aexpected0 = results0['Scores'][away_team]['mean']
        hexpected1 = results1['Scores'][home_team]['mean']
        aexpected1 = results1['Scores'][away_team]['mean']

        hq0 = list(results0['Scores'][home_team][['25%', '50%', '75%']])
        aq0 = list(results0['Scores'][away_team][['25%', '50%', '75%']])
        hq1 = list(results1['Scores'][home_team][['25%', '50%', '75%']])
        aq1 = list(results1['Scores'][away_team][['25%', '50%', '75%']])

        hb0 = bin_scores(hscore, hq0)
        ab0 = bin_scores(ascore, hq0)
        hb1 = bin_scores(hscore, hq1)
        ab1 = bin_scores(ascore, hq1)

        outdata0.loc[counter] = [round, home_team, away_team, venue, hscore, hexpected0] + hq0 + hb0
        outdata1.loc[counter] = [round, home_team, away_team, venue, hscore, hexpected1] + hq1 + hb1

        counter += 1

        outdata0.loc[counter] = [round, away_team, home_team, venue, ascore, aexpected0] + aq0 + ab0
        outdata1.loc[counter] = [round, away_team, home_team, venue, ascore, aexpected1] + aq1 + ab1

        counter += 1

    pd.Panel({'NoWeights': outdata0, 'Weights': outdata1}).to_excel(os.path.join(base_path, 'ValidationResultsRaw.xlsx'))

print(outdata0)
print(outdata1)