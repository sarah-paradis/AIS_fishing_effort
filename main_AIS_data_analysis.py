# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 12:21:17 2019

@author: user
"""
import os
import pandas as pd

import AIS_additional_data as AIS
import fishing_effort_mins

file_dir = 'Prova'
assert os.path.exists(file_dir)

additional_columns = ['Mmsi', 'Latitud', 'Longitud', 'Latitud_decimal', 
                              'Longitud_decimal', 'Sog', 'Cog', 'Heading', 'Rot',
                              'Eslora', 'Manga']

# Data is saved into a new directory named "Output"
fishing_effort_mins.fishing_effort_min(file_dir, 
                                       datetime_column = 'Fecha_Posicion', 
                                       name_column = 'Nombre',
                                       additional_columns = additional_columns)

dir_output = os.path.join(file_dir, 'Output/Output_trawl')
os.makedirs(dir_output, exist_ok=True)

df = pd.DataFrame() # Create empty DataFrame

cnt = 0

#♦ Identify trawling activities

for file in os.listdir(os.path.join(file_dir, 'Output')): # Open all Excel files in the directory
    if file[-3:] == 'xls':
        with AIS.print_with_time('Opening file ',file):
            df = pd.read_excel(os.path.join(file_dir, 'Output', file))
        
        # Add additional data to the dataframe, from Fleet Register on the Net
        df = AIS.additional_data(df,
                                 name_column = 'Nombre', 
                                 Fleet_column = 'Vessel Name')
        
        # Extract only trawlers (OTB)
        df = df[(df.loc[:,['Gear Main', 'Gear Sec']] == 'OTB').any(axis=1)]
        # Reset index since filtering out samples must have altered the index of DataFrame
        df = df.reset_index().drop('index', axis=1)
        
        # Add criteria column for trawling speed (Sog)
        df['Sog_criteria'] = AIS.classify_sog(sog_col=df['Sog'],
                                              sog_low=1.0,
                                              sog_high=3.8)
        
        #♦ Identify trawling activities
        df, n_trawls = AIS.identify_trawling(df,
                                             datetime_column = 'Fecha_Posicion_min',
                                             min_haul = 20,
                                             min_duration_false_negative = 5,
                                             AIS_turn_off = 60,
                                             start_trawl_id = cnt + 1)
        cnt += n_trawls
        new_file = file[:-4]+'_trawl.xls'
        with AIS.print_with_time('Saving file ', new_file):
            df.to_excel(os.path.join(dir_output,new_file), index=None)