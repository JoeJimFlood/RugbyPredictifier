import pandas as pd
import matchup
import xlsxwriter
import xlrd
import sys
import time
import collections
import os

round_timer = time.time()

round_number = 'F_Matrix'

matchups = collections.OrderedDict()
matchups['Matchups'] = [('RSA', 'ARG'),
                        ('NZL', 'AUS')]
                      

location = os.getcwd().replace('\\', '/')
output_file = location + '/predictions/Round_' + str(round_number) + '.xlsx'

week_book = xlsxwriter.Workbook(output_file)
header_format = week_book.add_format({'align': 'center', 'bold': True, 'bottom': True})
index_format = week_book.add_format({'align': 'right', 'bold': True})
score_format = week_book.add_format({'num_format': '#0', 'align': 'right'})
percent_format = week_book.add_format({'num_format': '#0%', 'align': 'right'})

for game_time in matchups:
    
    sheet = week_book.add_worksheet(game_time)
    sheet.write_string(1, 0, 'Chance of Winning', index_format)
    sheet.write_string(2, 0, 'Expected Score', index_format)
    for i in range(1, 20):
        sheet.write_string(2+i, 0, str(5*i) + 'th Percentile Score', index_format)
    sheet.freeze_panes(0, 1)
    games = matchups[game_time]
    for i in range(len(games)):
        home = games[i][0]
        away = games[i][1]
        homecol = 3 * i + 1
        awaycol = 3 * i + 2
        sheet.write_string(0, homecol, home, header_format)
        sheet.write_string(0, awaycol, away, header_format)
        
        results = matchup.matchup(home, away)
        probwin = results['ProbWin']
        sheet.write_number(1, homecol, probwin[home], percent_format)
        sheet.write_number(1, awaycol, probwin[away], percent_format)
        home_dist = results['Scores'][home]
        away_dist = results['Scores'][away]
        home_bp = results['Bonus Points'][home]
        away_bp = results['Bonus Points'][away]
        sheet.write_number(2, homecol, home_dist['mean'], score_format)
        sheet.write_number(2, awaycol, away_dist['mean'], score_format)
        for i in range(1, 20):
            #print(type(home_dist))
            #print(home_dist[str(5*i)+'%'])
            sheet.write_number(2+i, homecol, home_dist[str(5*i)+'%'], score_format)
            sheet.write_number(2+i, awaycol, away_dist[str(5*i)+'%'], score_format)
        if i != len(games) - 1:
            sheet.write_string(0, 3 * i + 3, ' ')

week_book.close()

print('Round ' + str(round_number) + ' predictions calculated in ' + str(round((time.time() - round_timer) / 60, 2)) + ' minutes')