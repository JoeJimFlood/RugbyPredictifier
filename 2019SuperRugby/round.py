import os
os.chdir(os.path.dirname(__file__))

import pandas as pd
import matchup
import ranking
import xlsxwriter
import xlrd
import sys
import time
import collections
import matplotlib.pyplot as plt
from math import ceil, sqrt, log2

def rgb2hex(r, g, b):
    r_hex = hex(r)[-2:].replace('x', '0')
    g_hex = hex(g)[-2:].replace('x', '0')
    b_hex = hex(b)[-2:].replace('x', '0')
    return '#' + r_hex + g_hex + b_hex

plot_shape = {1: (1, 1),
              2: (1, 2),
              3: (2, 2),
              4: (2, 2),
              5: (2, 3),
              6: (2, 3),
              7: (3, 3)}

round_timer = time.time()

round_number = 7

matchups = collections.OrderedDict()

matchups['Friday'] = [('HURRICANES', 'CRUSADERS'),
                      ('WARATAHS', 'SUNWOLVES', 'NTL')]
matchups['Saturday'] = [('BLUES', 'STORMERS'),
                        ('REDS', 'REBELS'),
                        ('SHARKS', 'BULLS'),
                        ('JAGUARES', 'CHIEFS')]

location = os.getcwd().replace('\\', '/')
stadium_file = location + '/StadiumLocs.csv'
teamloc_file = location + '/TeamHomes.csv'
output_file = location + '/Weekly Forecasts/Round_' + str(round_number) + '.xlsx'
output_fig = location + '/Weekly Forecasts/Round_' + str(round_number) + '.png'

rankings = ranking.rank(os.path.join(location, 'Score Tables'), round_number)

n_games = 0
for day in matchups:
    n_games += len(matchups[day])

colours = {}
team_formats = {}
colour_df = pd.DataFrame.from_csv(location + '/colours.csv')
teams = list(colour_df.index)
for team in teams:
    primary = rgb2hex(int(colour_df.loc[team, 'R1']), int(colour_df.loc[team, 'G1']), int(colour_df.loc[team, 'B1']))
    secondary = rgb2hex(int(colour_df.loc[team, 'R2']), int(colour_df.loc[team, 'G2']), int(colour_df.loc[team, 'B2']))
    colours[team] = (primary, secondary)


plt.figure(figsize = (15, 15), dpi = 96)
plt.title('Round ' + str(round_number))
counter = 0

stadiums = pd.read_csv(stadium_file, index_col = 0)
teamlocs = pd.read_csv(teamloc_file, header = None, index_col = 0)[1]

for read_data in range(1):

    week_book = xlsxwriter.Workbook(output_file)
    header_format = week_book.add_format({'align': 'center', 'bold': True, 'bottom': True})
    index_format = week_book.add_format({'align': 'right', 'bold': True})
    score_format = week_book.add_format({'num_format': '#0', 'align': 'right'})
    percent_format = week_book.add_format({'num_format': '#0%', 'align': 'right'})
    merged_format = week_book.add_format({'num_format': '#0.00', 'align': 'center'})
    merged_format2 = week_book.add_format({'num_format': '0.000', 'align': 'center'})
    for team in teams:
        team_formats[team] = week_book.add_format({'align': 'center', 'bold': True, 'border': True,
                                                   'bg_color': colours[team][0], 'font_color': colours[team][1]})

    for game_time in matchups:
        if read_data:
            data_book = xlrd.open_workbook(output_file)
            data_sheet = data_book.sheet_by_name(game_time)
        sheet = week_book.add_worksheet(game_time)
        sheet.write_string(1, 0, 'City', index_format)
        sheet.write_string(2, 0, 'Quality', index_format)
        sheet.write_string(3, 0, 'Entropy', index_format)
        sheet.write_string(4, 0, 'Hype', index_format)
        sheet.write_string(5, 0, 'Chance of Winning', index_format)
        sheet.write_string(6, 0, 'Expected Score', index_format)
        for i in range(1, 20):
            sheet.write_string(6+i, 0, str(5*i) + 'th Percentile Score', index_format)
        sheet.write_string(26, 0, 'Chance of Bonus Point Win', index_format)
        #sheet.write_string(23, 0, 'Chance of 4-Try Bonus Point with Draw', index_format)
        #sheet.write_string(24, 0, 'Chance of 4-Try Bonus Point with Loss', index_format)
        sheet.write_string(27, 0, 'Chance of Losing Bonus Point', index_format)
        sheet.freeze_panes(0, 1)
        games = matchups[game_time]
        for i in range(len(games)):
            home = games[i][0]
            away = games[i][1]
            
            try:
                venue = games[i][2]
            except IndexError:
                venue = teamlocs.loc[home]
            stadium = stadiums.loc[venue, 'Venue']
            city = stadiums.loc[venue, 'City']
            country = stadiums.loc[venue, 'Country']
            print('Simulating {0} vs {1} at {2}'.format(home, away, stadium))
            homecol = 3 * i + 1
            awaycol = 3 * i + 2
            sheet.write_string(0, homecol, home, team_formats[home])
            sheet.write_string(0, awaycol, away, team_formats[away])
            sheet.write_string(0, awaycol + 1, ' ')
            if read_data: #Get rid of this as I never use this option anymore
                sheet.write_number(5, homecol, data_sheet.cell(1, homecol).value, percent_format)
                sheet.write_number(5, awaycol, data_sheet.cell(1, awaycol).value, percent_format)
                for rownum in range(6, 26):
                    sheet.write_number(rownum, homecol, data_sheet.cell(rownum, homecol).value, score_format)
                    sheet.write_number(rownum, awaycol, data_sheet.cell(rownum, awaycol).value, score_format)
                for rownum in range(26, 30):
                    sheet.write_number(rownum, homecol, data_sheet.cell(rownum, homecol).value, percent_format)
                    sheet.write_number(rownum, awaycol, data_sheet.cell(rownum, awaycol).value, percent_format)
            else:
                results = matchup.matchup(home, away)
                probwin = results['ProbWin']
                hwin = probwin[home]
                awin = probwin[away]
                draw = 1 - hwin - awin

                #Calculate hype
                home_ranking = rankings.loc[home, 'Quantile']
                away_ranking = rankings.loc[away, 'Quantile']
                ranking_factor = (home_ranking + away_ranking)/2
                #uncertainty_factor = 1 - (hwin - awin)**2
                hp = hwin/(1-draw)
                ap = awin/(1-draw)
                entropy = -hp*log2(hp) - ap*log2(ap)
                hype = 100*ranking_factor*entropy

                sheet.write_number(5, homecol, probwin[home], percent_format)
                sheet.write_number(5, awaycol, probwin[away], percent_format)
                home_dist = results['Scores'][home]
                away_dist = results['Scores'][away]
                home_bp = results['Bonus Points'][home]
                away_bp = results['Bonus Points'][away]
                sheet.write_number(6, homecol, home_dist['mean'], score_format)
                sheet.write_number(6, awaycol, away_dist['mean'], score_format)
                for i in range(1, 20):
                    #print(type(home_dist))
                    #print(home_dist[str(5*i)+'%'])
                    sheet.merge_range(1, homecol, 1, awaycol, city, merged_format)
                    sheet.merge_range(2, homecol, 2, awaycol, ranking_factor, merged_format2)
                    sheet.merge_range(3, homecol, 3, awaycol, entropy, merged_format2)
                    sheet.merge_range(4, homecol, 4, awaycol, hype, merged_format)
                    sheet.write_number(6+i, homecol, home_dist[str(5*i)+'%'], score_format)
                    sheet.write_number(6+i, awaycol, away_dist[str(5*i)+'%'], score_format)
                    sheet.write_number(26, homecol, home_bp['4-Try Bonus Point with Win'], percent_format)
                    #sheet.write_number(23, homecol, home_bp['Try-Scoring Bonus Point with Draw'], percent_format)
                    #sheet.write_number(24, homecol, home_bp['Try-Scoring Bonus Point with Loss'], percent_format)
                    sheet.write_number(27, homecol, home_bp['Losing Bonus Point'], percent_format)
                    sheet.write_number(26, awaycol, away_bp['4-Try Bonus Point with Win'], percent_format)
                    #sheet.write_number(23, awaycol, away_bp['Try-Scoring Bonus Point with Draw'], percent_format)
                    #sheet.write_number(24, awaycol, away_bp['Try-Scoring Bonus Point with Loss'], percent_format)
                    sheet.write_number(27, awaycol, away_bp['Losing Bonus Point'], percent_format)
            if i != len(games) - 1:
                sheet.write_string(0, 3 * i + 3, ' ')

            counter += 1
            
            if n_games == 5 and counter == 5:
                plot_pos = 6
            elif n_games == 7 and counter == 7:
                plot_pos = 8
            elif n_games == 8 and counter == 8:
                plot_pos = 9
            else:
                plot_pos = counter

            plt.subplot(plot_shape[n_games][0], plot_shape[n_games][1], plot_pos)
            labels = [home[:3], away[:3], 'DRAW']
            values = [hp, ap, 1 - hwin - awin]
            colors = [colours[home][0], colours[away][0], '#808080']
            ex = 0.05
            explode = [ex, ex, ex]
            plt.pie(values,
                    colors = colors,
                    labels = labels,
                    explode = explode,
                    autopct='%.0f%%',
                    startangle = 90,
                    labeldistance = 1,
                    textprops = {'backgroundcolor': '#ffffff', 'ha': 'center', 'va': 'center', 'fontsize': 12})
            plt.title(home + ' vs ' + away + '\n' + stadium + '\n' + city + ', ' + country + '\nHype: ' + str(int(round(hype, 0))), size = 12)
            plt.axis('equal')

    week_book.close()

plt.savefig(output_fig)

print('Round ' + str(round_number) + ' predictions calculated in ' + str(round((time.time() - round_timer) / 60, 2)) + ' minutes')