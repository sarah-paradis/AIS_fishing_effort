# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:40:29 2019
@author: Sarah Paradis
"""

#%% Function definitions

import pandas as pd
pd.set_option('mode.chained_assignment', 'raise')

import os
import numpy as np
import timeit
from contextlib import contextmanager



@contextmanager
def print_with_time(*s):
    print(*s, end="", flush=True)
    start = timeit.default_timer()
    yield
    print("\t[%.2fs]" %(timeit.default_timer() - start), flush=True)

def additional_data(df, name_column, Fleet_column):
    """
    Add all the additional data (Gear type, Gt, Power (kw), Construction year, 
    Port). Additional data are extracted from Fleet Register on the Net: 
        http://ec.europa.eu/fisheries/fleet/index.cfm
    
    df = Dataframe with data from all fishing vessels
    
    name_column = name of column in original dataframe that has the name/code of 
    each vessel.
    
    Fleet_column = name of column in Fleet Register on the Net that has the 
    name/code of each vessel.
    """
    # Fishing fleet on the net    
    df_vessel_info = pd.read_csv('all_vessels_cat.csv', engine='python', delimiter=';')
    
    # ADDING THE ADDITIONAL DATA IN THE ORIGINAL DATAFRAME
    with print_with_time('Adding additional data to the DataFrame'):
        # Create dictionary of the data we want to add from Fishing Fleet on the Net
        gear_main = df_vessel_info.groupby([Fleet_column])['Gear Main Code'].first().to_dict()
        gear_sec = df_vessel_info.groupby([Fleet_column])['Gear Sec Code'].first().to_dict()
        ton_gt = df_vessel_info.groupby([Fleet_column])['Ton Gt'].first().to_dict()
        ton_oth = df_vessel_info.groupby([Fleet_column])['Ton Oth'].first().to_dict()
        power_main = df_vessel_info.groupby([Fleet_column])['Power Main'].first().to_dict()
        power_aux = df_vessel_info.groupby([Fleet_column])['Power Aux'].first().to_dict()
        const_year = df_vessel_info.groupby([Fleet_column])['Construction Year'].first().to_dict()
        port = df_vessel_info.groupby([Fleet_column])['Port Name'].first().to_dict()
        
        # Create new column based on data from Fishing Fleet on the Net
        df['Gear Main'] = df[name_column].map(gear_main)
        df['Gear Sec'] = df[name_column].map(gear_sec)
        df['Ton Gt'] = df[name_column].map(ton_gt)
        df['Ton other'] = df[name_column].map(ton_oth)
        df['Power Main'] = df[name_column].map(power_main)
        df['Power Aux'] = df[name_column].map(power_aux)
        df['Construction Year'] = df[name_column].map(const_year)
        df['Port Name'] = df[name_column].map(port)
        
    return df

def classify_sog(sog_col, sog_low, sog_high):
    """ Classifies Speed Over Ground (SOG) into low speed (0), medium speed (1), and high speed (2)
    where medium speed is the trawler speed when operating (doing a haul)
    sog_col = column where SOG is saved as in the DataFrame
    sog_low = minimum trawling speed
    sog_high = maximum trawling speed
    """
    with print_with_time('Classifying by Speed Over Ground'):
        class_sog = np.zeros(len(sog_col), dtype=int)
        class_sog[(sog_col >= sog_low) & (sog_col <= sog_high)] = 1
        class_sog[sog_col > sog_high] = 2
    return class_sog

def get_chunk_indices(a, in_same_chunk_fn = lambda x, y: x == y):
    len_a = len(a)
    if type(a) is pd.DataFrame:
        a = a.iloc
    diffs = [not in_same_chunk_fn(a[i], a[i+1]) for i in range(len_a-1)]
    indices = np.arange(len_a)
    start_indices = indices[1:][diffs].tolist() # index of the first different value
    start_indices.insert(0,0) # insert first index
    end_indices = indices[:-1][diffs].tolist() # index of the last 'equal' value of the chunk
    end_indices.append(len_a-1) # insert last index
    return list(zip(start_indices, end_indices))

def get_same_period_fn(minutes_diff, criteria):
    def same_period(x, y):
        """ Evaluates entries that are registered in less than 'minutes_diff' in order to decide if they
        belong in the same chunk
        
        minutes_diff = time interval to check a chunk
        
        criteria  = list of column names to check equalness
        """
        time_diff = pd.Timedelta(minutes=minutes_diff)
        res = y['Fecha_Posicion_min'] - x['Fecha_Posicion_min'] < time_diff

        for crit in criteria:
            res = res and (x[crit] == y[crit])
        return res
    return same_period

def identify_trawling(df, datetime_column, min_haul, min_duration_false_negative, AIS_turn_off, start_trawl_id=1):
    """ 
    Identifies trawling and haul ids.
    First:
    Find false-positives: when vessel is navigating at trawling speed but
    not doing a haul.
    This is corrected by establishing a minimum trawling duration (min_haul)
    Then:
    Find false-negatives: when vessel decreases/increases speed below/above
    trawling threshold but is actually still trawling.
    These are identified by establishing a maximum time that the vessel can 
    be doing a haul at a slower speed (min_duration_false_negative)
    Finally:
    Creates new columns (Trawl, Haul_id) establishing whether vessel is trawling and the haul ID
    
    Parameters:
    df = DataFrame
    datetime_column = name of column in dataframe that has the date and time of vessel positioning (type: string)
    min_duration_false_negative = duration that these events take in minutes (maximum duration)
    min_haul = minimum duration of a haul (time in minutes)
    AIS_turn_off = maximum time that the AIS is turned off before considering it belongs to a different haul
    
    Returns DataFrame with the additional columns of 'Sog criteria', 'Trawling'
    and 'Haul id'
    """

    # Create extra columns that Trawling and Haul_id that will be needed in the future
    df['Trawling'] = np.zeros(len(df), dtype=bool)
    df['Haul id'] = np.full(len(df), np.nan, dtype=np.float32)

    # Convert column into datetime to be able to recognize duration of each section
    df[datetime_column] = pd.to_datetime(df[datetime_column])

    with print_with_time('Identifying false-positives'):
        # Converts min_haul into datetime format
        min_haul = pd.Timedelta(minutes = min_haul)
        # Creates a list of tuples (index start, index end) of all the 'chunks' based on same SOG criteria (0,1,2)
        # considering that the vessel's AIS has been turned off during less than 'AIS_turn_off'.
        classify_trawling_list = get_chunk_indices(df, get_same_period_fn(AIS_turn_off, 
                                                                          ['Sog_criteria', 
                                                                           'Nombre',
                                                                           'Date']))
        for i, j in classify_trawling_list:
            if df['Sog_criteria'][i] == 1:
                if df[datetime_column][j]-df[datetime_column][i] > min_haul:
                    df.loc[i:j, 'Trawling'] = True

    # Convert min_duration_false_negative into minutes (time format)
    min_duration = pd.Timedelta(minutes = min_duration_false_negative)

    with print_with_time('Identifying false-negatives'):
        # Check if 0 (low speed) or 2 (high speeds) are between 1 (trawling speed) and its duration.
        # If the duration of these reductions in speeds (between trawling) are less
        # than the specified time criteria, it is converted into 1 (trawling speed)
        for idx in range(1, len(classify_trawling_list)-2):
            prev_trawl = df['Trawling'][classify_trawling_list[idx-1][1]] # Checks Trawling of previous chunk
            prev_class = df['Sog_criteria'][classify_trawling_list[idx-1][1]] # Checks Sog criteria of previous chunk
            next_class = df['Sog_criteria'][classify_trawling_list[idx+1][1]] # Checks Sog criteria of following chunk
            if prev_class == 1 and next_class == 1 and prev_trawl == True: # If current classification is between two trawling classifications
                start,end = classify_trawling_list[idx]
                if df[datetime_column][end] - df[datetime_column][start] <= min_duration:
                    df.loc[start:end, 'Trawling'] = True

    with print_with_time('Identifying Haul_id'):
        cnt = 0
        new_trawling_list = get_chunk_indices(df, get_same_period_fn(AIS_turn_off,
                                                                     ['Trawling', 
                                                                      'Nombre',
                                                                      'Date']))
        for i,j in new_trawling_list:
            if df['Trawling'][i] == True:
                df.loc[i:j, 'Haul id'] = cnt + start_trawl_id
                cnt += 1
    return df, cnt

#%% Calling functions
if __name__ == "__main__":
    file_dir = 'Data/Output/Prova'
    assert os.path.exists(file_dir)

    dir_output = os.path.join(file_dir, 'Output_trawl')
    os.makedirs(dir_output, exist_ok=True)

    df = pd.DataFrame() # Create empty DataFrame

    cnt = 0
    for file in os.listdir(file_dir): # Open all Excel files in the directory
        if file[-3:] == 'xls':
            with print_with_time('Opening file ',file):
                df = pd.read_excel(os.path.join(file_dir, file))
            #df = additional_data(df)
            df['Sog_criteria'] = classify_sog(sog_col=df['Sog'],
                                              sog_low=1.0,
                                              sog_high=3.8)
            df, n_trawls = identify_trawling(df,
                                             min_haul=20,
                                             min_duration_false_negative=5,
                                             AIS_turn_off=60,
                                             start_trawl_id=cnt+1)
            cnt += n_trawls
            new_file = file[:-4]+'_trawl.xls'
            with print_with_time('Saving file ', new_file):
                df.to_excel(os.path.join(dir_output,new_file), index=None)