import pandas as pd
import matchup
import xlsxwriter
import xlrd
import sys
import time
import collections
import os
import matplotlib.pyplot as plt

def rgb2hex(r, g, b):
    r_hex = hex(r)[-2:].replace('x', '0')
    g_hex = hex(g)[-2:].replace('x', '0')
    b_hex = hex(b)[-2:].replace('x', '0')
    return '#' + r_hex + g_hex + b_hex

round_timer = time.time()

round_number = 5

matchups = collections.OrderedDict()
matchups['Sunday'] = [('SAN DIEGO', 'OHIO'),
                      ('SACRAMENTO', 'DENVER')]

n_games = 0
for day in matchups.keys():
    n_games += len(matchups[day])

location = os.getcwd().replace('\\', '/')
output_file = location + '/Weekly Forecasts/Week_' + str(round_number) + '.xlsx'
output_fig = location + '/Weekly Forecasts/Week_' + str(round_number) + '.png'

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

abbr = {}
abbr_df = pd.DataFrame.from_csv(location + '/abbr.csv')
for team in teams:
    abbr[team] = abbr_df.loc[team, 'ABBR']


plt.figure(figsize = (15, 15), dpi = 96)
plt.title('Round ' + str(round_number))
counter = 0

for read_data in range(1):

    week_book = xlsxwriter.Workbook(output_file)
    header_format = week_book.add_format({'align': 'center', 'bold': True, 'bottom': True})
    index_format = week_book.add_format({'align': 'right', 'bold': True})
    score_format = week_book.add_format({'num_format': '#0', 'align': 'right'})
    percent_format = week_book.add_format({'num_format': '#0%', 'align': 'right'})
    for team in teams:
        team_formats[team] = week_book.add_format({'align': 'center', 'bold': True, 'border': True,
                                                   'bg_color': colours[team][0], 'font_color': colours[team][1]})

    

    for game_time in matchups:
        if read_data:
            data_book = xlrd.open_workbook(output_file)
            data_sheet = data_book.sheet_by_name(game_time)
        sheet = week_book.add_worksheet(game_time)
        sheet.write_string(1, 0, 'Chance of Winning', index_format)
        sheet.write_string(2, 0, 'Expected Score', index_format)
        for i in range(1, 20):
            sheet.write_string(2+i, 0, str(5*i) + 'th Percentile Score', index_format)
        sheet.write_string(22, 0, 'Chance of Bonus Point Win', index_format)
        #sheet.write_string(23, 0, 'Chance of 4-Try Bonus Point with Draw', index_format)
        #sheet.write_string(24, 0, 'Chance of 4-Try Bonus Point with Loss', index_format)
        sheet.write_string(23, 0, 'Chance of Losing Bonus Point', index_format)
        sheet.freeze_panes(0, 1)
        games = matchups[game_time]
        for i in range(len(games)):
            home = games[i][0]
            away = games[i][1]
            homecol = 3 * i + 1
            awaycol = 3 * i + 2
            sheet.write_string(0, homecol, home, team_formats[home])
            sheet.write_string(0, awaycol, away, team_formats[away])
            sheet.write_string(0, awaycol + 1, ' ')
            if read_data:
                sheet.write_number(1, homecol, data_sheet.cell(1, homecol).value, percent_format)
                sheet.write_number(1, awaycol, data_sheet.cell(1, awaycol).value, percent_format)
                for rownum in range(2, 22):
                    sheet.write_number(rownum, homecol, data_sheet.cell(rownum, homecol).value, score_format)
                    sheet.write_number(rownum, awaycol, data_sheet.cell(rownum, awaycol).value, score_format)
                for rownum in range(22, 26):
                    sheet.write_number(rownum, homecol, data_sheet.cell(rownum, homecol).value, percent_format)
                    sheet.write_number(rownum, awaycol, data_sheet.cell(rownum, awaycol).value, percent_format)
            else:
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
                    sheet.write_number(22, homecol, home_bp['4-Try Bonus Point with Win'], percent_format)
                    #sheet.write_number(23, homecol, home_bp['Try-Scoring Bonus Point with Draw'], percent_format)
                    #sheet.write_number(24, homecol, home_bp['Try-Scoring Bonus Point with Loss'], percent_format)
                    sheet.write_number(23, homecol, home_bp['Losing Bonus Point'], percent_format)
                    sheet.write_number(22, awaycol, away_bp['4-Try Bonus Point with Win'], percent_format)
                    #sheet.write_number(23, awaycol, away_bp['Try-Scoring Bonus Point with Draw'], percent_format)
                    #sheet.write_number(24, awaycol, away_bp['Try-Scoring Bonus Point with Loss'], percent_format)
                    sheet.write_number(23, awaycol, away_bp['Losing Bonus Point'], percent_format)
            if i != len(games) - 1:
                sheet.write_string(0, 3 * i + 3, ' ')
            

            counter += 1
            hwin = probwin[home]
            awin = probwin[away]

            if n_games == 2:
                plt.subplot(1, 2, counter)
            else:
                plt.subplot(1, 1, counter)
            labels = [abbr[home], abbr[away]]
            values = [hwin, awin]
            colors = [colours[home][0], colours[away][0]]
            ex = 0.05
            explode = [ex, ex]
            plt.pie(values,
                    colors = colors,
                    labels = labels,
                    explode = explode,
                    autopct='%.0f%%',
                    startangle = 90,
                    labeldistance = 1,
                    textprops = {'backgroundcolor': '#ffffff', 'ha': 'center', 'va': 'bottom'})
            plt.title(home + ' vs ' + away, size = 24)
            plt.axis('equal')
            #if n_games == 2:
            #    v = plt.axis()
            #    plt.axis([v[0], v[1], v[2], 0.5*v[3]])

    week_book.close()

plt.savefig(output_fig)

print('Round ' + str(round_number) + ' predictions calculated in ' + str(round((time.time() - round_timer) / 60, 2)) + ' minutes')