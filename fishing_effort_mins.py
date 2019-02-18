#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:14:13 2019

@author: sarahparadis
"""

import pandas as pd
import os

import timeit
from contextlib import contextmanager

@contextmanager
def print_with_time(s):
    print(s, end="", flush=True)
    start = timeit.default_timer()
    yield
    print("\t[%.2fs]" %(timeit.default_timer() - start), flush=True)

def fishing_effort_min(file_dir):
    """
    Opens all AIS files and extracts the first entry of each minute for 
    each vessel during the sampling period.
    file_dir = directory with all the AIS files saved as Excel (.xls)    
    """
    dir_output = os.path.join(file_dir, 'Output')
    os.makedirs(dir_output, exist_ok=True)
    
    df = pd.DataFrame() # Create empty DataFrame 
    
    for file in os.listdir(file_dir): # Open all Excel files in the directory 
        if file[-3:] == 'xls':
            with print_with_time('Opening file '+file):
                data = pd.read_excel(os.path.join(file_dir, file))
                df = df.append(data)     
        
    # Create column 'Fecha_Posicion_min' from string format to date format rounded to minutes
    with print_with_time('Converting to "date-time" format'):
        df['Fecha_Posicion_min'] = pd.to_datetime(df['Fecha_Posicion'], dayfirst=True)
    with print_with_time('Rounding time to minutes'):
        df['Fecha_Posicion_min'] = df["Fecha_Posicion_min"].values.astype('<M8[m]')
    
    # Extract the first entry of each minute for each vessel and each day
    with print_with_time('Extracting the first entry of each minute for each vessel and day'):
        df_min = df.groupby(['Nombre','Fecha_Posicion_min']).first()

    df_min = df_min.reset_index() # Reset index 
    
    # Organize DataFrame eliminating the unnecessary columns and putting them in order
    df_min = df_min[['Nombre', 'Fecha_Posicion', 'Fecha_Posicion_min',
                              'Mmsi', 'Latitud', 'Longitud', 'Latitud_decimal', 
                              'Longitud_decimal', 'Sog', 'Cog', 'Heading', 'Rot',
                              'Eslora', 'Manga']]
    # Create a new column with only the month and year (to make a .csv file with this data)
    with print_with_time('Grouping data by month'):
        df_min['Mes'] = df_min['Fecha_Posicion_min'].dt.to_period('M')

    # Extract the months in the files to later create individual dataframes 
    # and the names of the output .csv files for each month
        months = list(df_min.Mes.unique()) # list of the months from which the DataFrames will be extracted
        months_name = [month.strftime('%m_%Y') for month in months] # list of the names of the months for the .csv files
    
    for month, month_name in zip(months, months_name):
        df_month = df_min[df_min['Mes'] == month] # Create a new DataFrame with the data of that specific month
        file_name = month_name+'.csv'
        print('Exporting '+file_name+' to .csv file')
        df_month.to_csv(os.path.join(dir_output,file_name), index=None)
            
  
if __name__ == "__main__": 
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("file_dir",
                        type=str,
                        help="Directory where all AIS data is saved as Excel files (.xls)")
    args = parser.parse_args()
    
    fishing_effort_min(args.file_dir)
