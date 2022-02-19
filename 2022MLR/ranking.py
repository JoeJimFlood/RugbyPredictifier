import pandas as pd
import numpy as np
import os
from scipy.stats.distributions import norm
#from subprocess import Popen

#wd = r'D:\RugbyPredictifier\2018SuperRugby\teamcsvs'

def rank(wd, roundno):

    pf = {}
    pa = {}

    scores = np.array([5, 2, 3, 3])

    for team in os.listdir(wd):
        data = pd.read_csv(os.path.join(wd, team))
        pf[team[:-4]] = np.dot(data[['TF', 'CF', 'PF', 'DGF']], scores).mean()
        pa[team[:-4]] = np.dot(data[['TA', 'CA', 'PA', 'DGA']], scores).mean()

    results = pd.DataFrame(index = pf.keys(), columns = ['Attack', 'Defense', 'Overall'])

    for team in os.listdir(wd):
        data = pd.read_csv(os.path.join(wd, team))
        data['For'] = np.dot(data[['TF', 'CF', 'PF', 'DGF']], scores)
        data['Against'] = np.dot(data[['TA', 'CA', 'PA', 'DGA']], scores)
        data['OppFor'] = data['OPP'].map(pf)
        data['OppAgainst'] = data['OPP'].map(pa)
        data['Attack'] = data['For'] - data['OppAgainst']
        data['Defense'] = data['Against'] - data['OppFor']
        data['Overall'] = data['Attack'] - data['Defense']
    
        results.loc[team[:-4]] = data[results.columns].mean()

    results['Standardised'] = (results['Overall'] - results['Overall'].mean())/results['Overall'].std()
    
    results['Quantile'] = norm.cdf(results['Standardised'].astype(float))

    outfile = os.path.join(os.path.split(wd)[0], 'Rankings', 'RankingsRound{}.csv'.format(roundno))
    results.sort_values('Overall', ascending = False).to_csv(outfile)
    return results
    #Popen(outfile, shell = True)