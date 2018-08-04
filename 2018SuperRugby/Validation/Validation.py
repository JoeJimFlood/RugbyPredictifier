import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr as correl
from scipy.special import erfinv
import os
import sys

def probit(p):
    '''
    Probit function (inverse of standard normal cummulative distribution function)
    '''
    return np.sqrt(2)*erfinv(2*p-1)

#Set paths and read in data
base_path = os.path.split(__file__)[0]
sys.path.append(base_path)
from validation_util import *

infile = os.path.join(base_path, 'ValidationData.csv')
data = pd.read_csv(infile)

#Define regression data as data from after international break (Round 17 onwards) and perform regression analysis
reg_data = data.query('Round >= 17')
reg_data['Constant'] = np.ones_like(reg_data.index)
reg = sm.OLS(reg_data['Actual Score'], reg_data[['Expected Score', 'Constant']])
res = reg.fit()

#Write regression results to file
regression_file = os.path.join(base_path, 'RegressionResults.txt')
f = open(regression_file, 'w')
f.write(str(res.summary()))
f.close()

#Obtain color gradient for plotting
color_gradient = color_interpolate('#ff0000', '#0000ff', 19)

#Create figure
plt.figure(figsize = (7.5, 7.5))

#For each round plot actual vs expected scores with the size and color based on the round number
for r in range(2, 21):
    label = 'Round %s'%(r)
    if r == 20:
        label = 'Finals'
    round_data = data.query('Round == @r')
    plt.scatter(round_data['Expected Score'], round_data['Actual Score'],
                s = r, color = color_gradient[r-2], alpha = 0.8, label = label)

#Plot regression line against the data
(xmin, xmax) = (reg_data['Expected Score'].min(), reg_data['Expected Score'].max())
(ymin, ymax) = (res.params[0]*xmin + res.params[1], res.params[0]*xmax + res.params[1])
plt.plot([xmin, xmax], [ymin, ymax], 'b--', label = 'Round 17+')

#Format plot
plt.xlabel('Expected Score')
plt.ylabel('Actual Score')
plt.xticks(range(-10, 90, 10))
plt.yticks(range(0, 90, 10))
plt.axis('equal')
plt.legend(loc = 'upper right')
title_lines = ['Rugby Predictifier Validation: 2018 Super Rugby Season',
               'Actual Scores vs. Expected Scores (Average of 5,000,000 Simulations)',
               'Round 17+: Actual = {0}\u00d7Expected - {1}, r\u00b2 = {2}'.format(round(res.params[0], 2),
                                                                                    abs(round(res.params[1], 2)),
                                                                                    round(res.rsquared_adj, 3))]
plt.title('\n'.join(title_lines))
ax = plt.gca()
ax.set_axisbelow(True)
plt.grid(True)

#Write plot to file
scatterplot_file = os.path.join(base_path, 'ScoreScatterplot.png')
plt.savefig(scatterplot_file)
plt.clf()
plt.close()

#Compute percentage of actual scores in forecast quartiles
#Scores on the boundary go half to lower and half to upper
N_reg = len(reg_data.index)
q1_l = (reg_data['Actual Score'] <= reg_data['25%']).sum() / N_reg
q2_l = ((reg_data['Actual Score'] <= reg_data['50%']) * (reg_data['Actual Score'] > reg_data['25%'])).sum() / N_reg
q3_l = ((reg_data['Actual Score'] <= reg_data['75%']) * (reg_data['Actual Score'] > reg_data['50%'])).sum() / N_reg
q4_l = (reg_data['Actual Score'] > reg_data['75%']).sum() / N_reg

q1_u = (reg_data['Actual Score'] < reg_data['25%']).sum() / N_reg
q2_u = ((reg_data['Actual Score'] < reg_data['50%']) * (reg_data['Actual Score'] >= reg_data['25%'])).sum() / N_reg
q3_u = ((reg_data['Actual Score'] < reg_data['75%']) * (reg_data['Actual Score'] >= reg_data['50%'])).sum() / N_reg
q4_u = (reg_data['Actual Score'] >= reg_data['75%']).sum() / N_reg

q1 = 0.5*(q1_l+q1_u)
q2 = 0.5*(q2_l+q2_u)
q3 = 0.5*(q3_l+q3_u)
q4= 0.5*(q4_l+q4_u)

p = np.array([q1, q2, q3, q4])
n = np.array(4*[N_reg])
se = np.sqrt(p*(1-p)/n)

#Create bar plot
plt.figure(figsize = (7.5, 2.5))
plt.plot([0, 1], [0.25, 0.25], 'k--')
plt.bar([0, 0.25, 0.5, 0.75], p, 4*[0.25],
        yerr = probit(0.975)*se, error_kw = {'capsize': 7},
        align = 'edge', facecolor = '#00d3ca', edgecolor = 'k')
plt.xticks([0, 0.25, 0.5, 0.75, 1], ['', '1st Quartile', '2nd Quartile', '3rd Quartile', ''])
plt.yticks(np.arange(0, 0.6, 0.1), ['0%', '10%', '20%', '30%', '40%', '50%'])
plt.xlim(0, 1)
plt.title('Round 17+ Score Distribution Validation')
plt.ylabel('% of Scores within\nForecast Quartiles')

#Write plot to file
barplot_file = os.path.join(base_path, 'QuartileBarPlot.png')
plt.savefig(barplot_file)
plt.clf()
plt.close()

print('Done')