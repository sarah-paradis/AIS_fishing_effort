#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:14:13 2019

@author: sarahparadis
"""

import pandas as pd
import os

def fishing_effort_min(file_dir):
    """
    Opens all AIS files and extracts the first entry of each minute for 
    each vessel during the sampling period.
    file_dir = directory with all the AIS files saved as Excel (.xls)    
    """
#    file_output = os.mkdir(os.path.join(file_dir, 'Output'))
    
    df = pd.DataFrame() # Create empty DataFrame 
    
    for file in os.listdir(file_dir): # Open all Excel files in the directory 
        if file[-3:] == 'xls':
            print('Opening file '+file)
            data = pd.read_excel(os.path.join(file_dir, file))
            df = df.append(data)     
        
    # Convert column of 'Fecha_Posicion' from string format to date format
    df['Fecha_Posicion'] = pd.to_datetime(df['Fecha_Posicion'], dayfirst=True)
    
    # Copy date column in a new column, because first column will be rounded up to minutes
    df['Fecha_Hora'] = df['Fecha_Posicion']
    
    # Create a new column with only the date (no time)
    df['Fecha'] = df['Fecha_Hora'].dt.to_period('D')
    
    df = df.set_index('Fecha_Posicion')
    
    # Extract the first entry of each minute for each vessel and each day
    print('Extracting the first entry of each minute for each vessel and day')
    df_min = df.groupby(['Nombre','Fecha']).resample('T').first().dropna(subset=['Nombre'])
    
    # Drop columns because they are repeated
    df_min = df_min.drop(columns=['Nombre','Fecha']) 
    df_min = df_min.reset_index() # Reset index 
    
    # Organize DataFrame eliminating the unnecessary columns and putting them in order
    df_min = df_min[['Nombre', 'Fecha', 'Fecha_Posicion', 'Fecha_Hora',
                                  'Mmsi', 'Latitud', 'Longitud', 'Latitud_decimal', 
                                  'Longitud_decimal', 'Sog', 'Cog', 'Heading', 'Rot',
                                  'Eslora', 'Manga']]
    # Create a new column with only the month and year (to make a .csv file with this data)
    df_min['Mes'] = df_min['Fecha_Hora'].dt.to_period('M')

    # Extract the months in the files to later create individual dataframes 
    # and the names of the output .csv files for each month
    months = [] # list of the months from which the DataFrames will be extracted
    months_name = [] # list of the names of the months for the .csv files
    for i in range(len(df_min.Mes.unique())):
        month = df_min.Mes.unique()[i]
        month_name = month.strftime('%m_%Y') # Needs to be converted to string to name the .csv files
        months.append(month)
        months_name.append(month_name)
 
    # Create empty DataFrame for all the months in the files 
    m = {}
    for month in months:
        m[month] = pd.DataFrame()

    a = 0
    for month, df_month in m.items():
        df_month = df_min[df_min['Mes'] == month] # Create a new DataFrame with the data of that specific month
        file_name = months_name[a]+'.csv'
        print('Exporting '+file_name+' to .csv file')
#        df_month.to_csv(os.path.join(file_output, file_name)) # Save the data in a new .csv file with the name of the month
        df_month.to_csv(file_name, index=None)
        a += 1
        


file_dir = '/Users/sarahparadis/Dropbox/Sarah Paradis - Tesis/Palamos/ABIDES/Trampas/AIS/DATA/'

fishing_effort_min(file_dir)
    
    
if __name__ == "__main__": 
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("file_dir",
                        type=str,
                        help="Directory where all AIS data is saved as Excel files (.xls)")
    args = parser.parse_args()
    
    fishing_effort_min(args.file_dir)
