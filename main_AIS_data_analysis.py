# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 12:21:17 2019

@author: Sarah Paradis
"""

import AIS
import os

##
# 1. Open all data
##

df = AIS.read_all_data('/Users/sarahparadis/Dropbox/PhD/Mapas/ABRIC_work/AIS-Blanes/DATA2018')

##
# 2. Extract vessels by minute
##

# additional columns to be kept in the DataFrame
additional_columns = ['Mmsi', 'CallSign', 'Latitud', 'Longitud', 'Latitud_decimal',
                      'Longitud_decimal', 'Sog', 'Cog', 'Heading', 'Rot',
                      'Eslora', 'Manga']

df = AIS.fishing_effort_min(df,
                            datetime_column='Fecha_Posicion',
                            name_column='Nombre',
                            additional_columns=additional_columns)

##
# 3. Add additional information from Fleet Register database
##

df = AIS.additional_data(df,
                         name_column='Nombre',
                         file_fleet='/Users/sarahparadis/Dropbox/PhD/Mapas/ABIDES/AIS/all_vessels_cat.csv',
                         Fleet_column='Vessel Name')

df_no_data = AIS.no_data_fleet(df,
                               column_name='Gear Main',
                               vessel_name='Nombre',
                               output_name='missing_data.xlsx',
                               file_dir='/Users/sarahparadis/Dropbox/PhD/Mapas/ABRIC_work/AIS-Blanes/DATA2018')

df_vessels = AIS.all_vessels(df,
                             name_column='Nombre',
                             output_name='all_vessels.xlsx',
                             file_dir='/Users/sarahparadis/Dropbox/PhD/Mapas/ABRIC_work/AIS-Blanes/DATA2018')
# Make sure using these Excel files that all the vessels have been assigned the additional information from Fleet
# Register database.

##
# 4. Extract trawlers
##

df_OTB = AIS.filter_trawlers(df, column_gear=['Gear Main', 'Gear Sec'])

##
# 5. Filter outliers from dataset
##

print(df_OTB['Sog'].describe())
df_OTB = df_OTB[df_OTB['Sog'] < 20]
df_OTB = df_OTB[df_OTB['Sog'] != 0]
print(df_OTB['Sog'].describe())

AIS.save_all_data(df_OTB, '/Users/sarahparadis/Dropbox/PhD/Mapas/ABRIC_work/AIS-Blanes/DATA2018/Output') # You can save the data halfway if you want. Uncomment if so.
df_OTB = AIS.read_all_data('/Users/sarahparadis/Dropbox/PhD/Mapas/ABRIC_work/AIS-Blanes/DATA2018/Output')  # Re-open all the data if you stopped at this point. Uncomment if so.

##
# 6. Draw histogram of speed over ground to identify trawling speed intervals
##

df_speed = AIS.define_gauss_distribution(df_OTB['Sog'])  # Defines the bimodal gaussian distribution of the dataset
print(df_speed)

# Define the trawling speed range, using average trawling speed +- 2 standard deviation
min_speed = round((df_speed.iloc[0, 0] - 2 * df_speed.iloc[1, 0]), 1)
max_speed = round((df_speed.iloc[0, 0] + 2 * df_speed.iloc[1, 0]), 1)
print('In this dataset, trawling is considered to take place between %.1f and %.1f knots.' % (min_speed, max_speed))

# Define the navigating speed range, using average navigating speed +- standard deviation
min_nav = round((df_speed.iloc[3, 0] - 2 * df_speed.iloc[4, 0]), 1)
max_nav = round((df_speed.iloc[3, 0] + 2 * df_speed.iloc[4, 0]), 1)
print('In this dataset, navigating is considered to take place between %.1f and %.1f knots.' % (min_nav, max_nav))

# Draw the histogram and save it

AIS.draw_histogram(df_OTB['Sog'], df_speed, 'Histogram_trawling_2018_BLANES')

##
# 7. Classify trawling activities based on their speed
##

df_OTB['Sog_criteria'] = AIS.classify_sog(sog_col=df_OTB['Sog'],
                                          min_trawl=min_speed,
                                          max_trawl=max_speed,
                                          min_nav=min_nav)

# Save all data into a new Excel file
# AIS.save_all_data(df_OTB, 'AIS_2017/Output Excel', 'xls')

# Save all data into csv files
AIS.save_all_data(df_OTB, '/Users/sarahparadis/Dropbox/PhD/Mapas/ABRIC_work/AIS-Blanes/DATA2018/Output_csv', 'csv')

##
# 8. Identify trawling activities
##

# Set counter to 0
cnt = 0

# Open each file and execute the function to identify trawling activities and save it in a different directory
# as a file ending with "_trawling"

file_dir = '/Users/sarahparadis/Dropbox/PhD/Mapas/ABRIC_work/AIS-Blanes/DATA2018/Output_csv'
dir_output = '/Users/sarahparadis/Dropbox/PhD/Mapas/ABRIC_work/AIS-Blanes/DATA2018/Output_trawling'

for file in os.listdir(file_dir):  # Open all Excel files in the directory
    if file.endswith('xls') or file.endswith('xlsx') or file.endswith('csv'):
        df = AIS.read_df(file, file_dir)
        df_OTB, n_trawls = AIS.identify_trawling(df,
                                             datetime_column='Fecha_Posicion_min',
                                             name_column='Nombre',
                                             min_haul=100,
                                             min_duration_false_positive=10,
                                             min_duration_false_negative=5,
                                             AIS_turn_off=50,
                                             start_trawl_id=cnt + 1)
        cnt += n_trawls
        print('\n')
        print(cnt)
        file = file[:-4] + '_trawling' + file[-4:]  # adds "_trawling" between file name and its extension
        AIS.save_df(df_OTB, file, dir_output)

