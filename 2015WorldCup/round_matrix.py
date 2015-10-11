import pandas as pd
import matchup
import xlsxwriter
import xlautofit
import xlrd
import sys
import time
import collections
import os

round_timer = time.time()

round_number = 'QF_Matrix'

matchups = collections.OrderedDict()
matchups['RSA'] = [('RSA', 'WAL'),
                   ('RSA', 'NZL'),
                   ('RSA', 'FRA'),
                   ('RSA', 'IRE'),
                   ('RSA', 'ARG'),
                   ('RSA', 'AUS'),
                   ('RSA', 'SCO')]
matchups['WAL'] = [('WAL', 'NZL'),
                   ('WAL', 'FRA'),
                   ('WAL', 'IRE'),
                   ('WAL', 'ARG'),
                   ('WAL', 'AUS'),
                   ('WAL', 'SCO')]
matchups['NZL'] = [('NZL', 'FRA'),
                   ('NZL', 'IRE'),
                   ('NZL', 'ARG'),
                   ('NZL', 'AUS'),
                   ('NZL', 'SCO')]
matchups['FRA'] = [('FRA', 'IRE'),
                   ('FRA', 'ARG'),
                   ('FRA', 'AUS'),
                   ('FRA', 'SCO')]
matchups['IRE'] = [('IRE', 'ARG'),
                   ('IRE', 'AUS'),
                   ('IRE', 'SCO')]
matchups['ARG'] = [('ARG', 'AUS'),
                   ('ARG', 'SCO')]
matchups['AUS'] = [('AUS', 'SCO')]
                      

location = os.getcwd().replace('\\', '/')
output_file = location + '/predictions/Round_' + str(round_number) + '.xlsx'

for read_data in range(2):

    week_book = xlsxwriter.Workbook(output_file)
    header_format = week_book.add_format({'align': 'center', 'bold': True, 'bottom': True})
    index_format = week_book.add_format({'align': 'right', 'bold': True})
    score_format = week_book.add_format({'num_format': '#0', 'align': 'right'})
    percent_format = week_book.add_format({'num_format': '#0%', 'align': 'right'})

    
    if read_data:
        colwidths = xlautofit.even_widths_single_index(output_file)

    for game_time in matchups:
        if read_data:
            data_book = xlrd.open_workbook(output_file)
            data_sheet = data_book.sheet_by_name(game_time)
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
            if read_data:
                sheet.write_number(1, homecol, data_sheet.cell(1, homecol).value, percent_format)
                sheet.write_number(1, awaycol, data_sheet.cell(1, awaycol).value, percent_format)
                for rownum in range(2, 22):
                    sheet.write_number(rownum, homecol, data_sheet.cell(rownum, homecol).value, score_format)
                    sheet.write_number(rownum, awaycol, data_sheet.cell(rownum, awaycol).value, score_format)
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
            if i != len(games) - 1:
                sheet.write_string(0, 3 * i + 3, ' ')
            if read_data:
                for colnum in range(sheet.dim_colmax):
                    sheet.set_column(colnum, colnum, colwidths[sheet.name][colnum])

    week_book.close()

print('Round ' + str(round_number) + ' predictions calculated in ' + str(round((time.time() - round_timer) / 60, 2)) + ' minutes')