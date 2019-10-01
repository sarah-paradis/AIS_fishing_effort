#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 16:14:13 2019
@author: sarahparadis
"""

import pandas as pd
pd.set_option('mode.chained_assignment', 'raise')

import os
import timeit
from contextlib import contextmanager

@contextmanager
def print_with_time(*s):
    print(*s, end="", flush=True)
    start = timeit.default_timer()
    yield
    print("\t[%.2fs]" %(timeit.default_timer() - start), flush=True)

def fishing_effort_min(file_dir, datetime_column, name_column, 
                       additional_columns):
    """
    Opens all AIS files and extracts the first entry of each minute for 
    each vessel during the sampling period.
    
    file_dir = directory with all the AIS files saved as Excel
    
    datetime_column = name of column in dataframe that has the date and 
    time of vessel positioning (type: string)
    
    name_column = name of column in dataframe that has the name/code of 
    each vessel

    additional_columns = list with the name of the columns in the dataframe 
    that would be exported    
    """
    dir_output = os.path.join(file_dir, 'Output')
    os.makedirs(dir_output, exist_ok=True)
    
    df = pd.DataFrame() # Create empty DataFrame 
    
    for file in os.listdir(file_dir): # Open all Excel files in the directory 
        if file[-3:] == 'xls' or file[-4:] == 'xlsx':
            with print_with_time('Opening file '+file):
                data = pd.read_excel(os.path.join(file_dir, file))
                df = df.append(data)     
        
    # Create new date-time column rounded to minutes
    with print_with_time('Converting to "date-time" format'):
        new_column = datetime_column + '_min'
        df[new_column] = pd.to_datetime(df[datetime_column], dayfirst=True)
    with print_with_time('Rounding time to minutes'):
        df[new_column] = df[new_column].values.astype('<M8[m]')
    
    # Extract the first entry of each minute for each vessel and each day
    with print_with_time('Extracting the first entry of each minute for each vessel and day'):
        df_min = df.groupby([name_column, new_column]).first()

    df_min = df_min.reset_index() # Reset index 
    
    # Organize DataFrame eliminating the unnecessary columns and putting them in order
    df_min = df_min[[name_column] + [datetime_column] + [new_column] + additional_columns]

    # Create a new column with only the month and year (to make a .csv file with this data)
    with print_with_time('Grouping data by month and day'):
        df_min['Month'] = df_min[new_column].dt.to_period('M')
        df_min['Date'] = df_min[new_column].dt.to_period('D')

    # Extract the months in the files to later create individual dataframes 
    # and the names of the output .csv files for each month
        months = sorted(list(df_min.Month.unique())) # list of the months from which the DataFrames will be extracted
        months_name = [month.strftime('%Y_%m') for month in months] # list of the names of the months for the .csv files
        days = []
        for month in months:
            day_month = sorted(list(df_min['Date'][df_min['Month'] == month].unique()))
            days.append(day_month)
    
    # Exporting files
    max_excel_rows = 65000 # actually it is 65536 (2^16)
    for month, month_name, idx in zip(months, months_name, range(len(months))):
        df_month = df_min[df_min['Month'] == month] # Create a new DataFrame with the data of that specific month
        file_name = month_name+'.xls'
        if len(df_month) < max_excel_rows:
            with print_with_time('Exporting '+file_name+' to .xls file'):
                df_month.to_excel(os.path.join(dir_output,file_name), index=None)
        else:
            with print_with_time('Exporting '+ file_name + ' in different .xls files'):
                while len(days[idx]) > 0: # save files
                    for i in range(len(days[idx])): # days
                        if len(df_month[df_month['Date'].isin(days[idx][:i+1])]) > max_excel_rows:
                            i = i-1
                            break
                    file_name = month_name+'_'+days[idx][i].strftime('%d')+'.xls'
                    print('\n Exporting '+file_name)
                    file_name = os.path.join(dir_output,file_name)
                    df_save = df_month[df_month['Date'].isin(days[idx][:i+1])]
                    assert len(df_save) > 0, "Too many rows for day %s. Cannot save in .xls."%days[idx][i+1].strftime('%d')
                    df_save.to_excel(file_name, index=None)
                    days[idx] = days[idx][i+1:]
    
  
if __name__ == "__main__": 
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("file_dir",
                        type=str,
                        help="Directory where all AIS data is saved as Excel files (.xls)")
    
    args = parser.parse_args()
#    args = argparse.Namespace(file_dir='DATA') 'Data' added in the Spyder Run->Configuration per file->command line options
    
    fishing_effort_min(args.file_dir)