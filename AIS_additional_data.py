# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 16:40:29 2019

@author: Sarah Paradis
"""

#%% Function definitions

import pandas as pd
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

def additional_data(df):
    """
    Add all the additional data (Gear type, Gt, Power (kw), Construction year, 
    Port, and whether the vessels belong to the BOE list of shrimp trawlers, or are
    available from VMS). Most additional data are extracted from Fishing Fleet on the Net 
    """
    # Fishing fleet on the net    
    df_vessel_info = pd.read_csv('all_vessels_cat.csv', engine='python', delimiter=';')
    
    # BOE shrimp
    shrimp_BOE = ['APOLO', 'BAHIA DE PALAMOS', 'BONOMAR F.', 'CANIGO', 'ESTRELLA DEL SUR III', 
                  'GERMANOR', 'GERMANS GRAS', 'J PIJOAN SEGUNDO', "L ARJAU", "L'ESPAVIL",
                  "L'HAVANERA", 'LA PUNTAIRE', 'MANDORRI', 'MIGUEL CARDENAL', 'MONTSE', 
                  'NOU GISBERT', 'NOVA GASELA', 'NUEVO SIBONEY', 'PEL BLANC XATONA', 
                  'PEPITA MARTI', 'SOLRAIG', 'TIA CINTA']
    
    # Giulia shrimp
    shrimp_Giulia = ['APOLO', 'ATLAS', 'BAHIA DE PALAMOS', 'BAIX EMPORDA', 'BONOMAR', 
                     'BONOMAR F.', 'CAIRO', 'CAIRO II', 'CANIGO', 'DARNACULLETA PRIMERO',
                     'EMPORDA', 'ESTRELLA DEL SUR III', 'GACELA', 'GERMANOR', 'GERMANS GRAS',
                     'GISBERT', 'J PIJOAN SEGUNDO', 'JOMARA', 'JUAN Y ANGELA', 'JUAN Y VIRGILIO', 
                     "L ARJAU", "L'ESPAVIL", "L'HAVANERA", "L'HORITZO U", 'LA PUNTAIRE', 
                     'LEVANTINA', 'MANDORRI', 'MARC', 'MARIA PARIS', 'MIGUEL CARDENAL', 
                     'MONTSE', 'NOU GISBERT', 'NOVA GASELA', 'NUEVO SIBONEY', 'PEPITA MARTI', 
                     'PERLA CUARTO', 'ROSA ILLA', 'SOLRAIG', 'TIA CINTA', 'TRESA']
    
    # VMS trawlers
    VMS_Palamos = ['APOLO', 'AVANZA', 'BAHIA DE PALAMOS', 'BONOMAR F.', 'CANIGO', 
                   'ESTRELLA DEL SUR III', 'GERMANOR', 'GERMANS GRAS', 'J PIJOAN SEGUNDO', 
                   'JUAN Y VIRGILIO', 'L ARJAU', 'LESPAVIL', "L'HAVANERA", 'LHORITZO U',
                   'MANDORRI', 'MIGUEL BERTRAN', 'MIGUEL CARDENAL', 'MONTSE', 
                   'NUEVO SIBONEY', 'ORATGE PRIMERO', 'PEL BLANC XATONA', 'PEPITA MARTI', 
                   'PERLA CUARTO', 'RAMONA', 'SOLRAIG', 'TIA CINTA']
    VMS_Arenys = ['COCA','COSTA DEL MARESME','FARO DE CALELLA','JOAN I FRANCISCO',
                  'LAIETA PRIMERO','LLAVANERES','MONT PALAU','PERLA TERCERO','ROCA GROSSA',
                  'DARNACULLETA PRIMERO']
    VMS_Blanes = ['BAHIA DE BLANES','BLANDA II','BLANDA TERCERO','BRISA DEL MAR','ES NIELL',
                  'ESTEL.LADA','FAMILIA LEON','MARINER','MARROI SEGON','NA TERESA',
                  'NOVA ROSA MARI','PERET SEGON','PUNTA SANTA ANA','VERGE DEL VILA']
    VMS_Barcelona = ['BONAMAR DOS','EL FAIRELL','EL GALAN','FRANCESC I LLUIS','JOANET',
                     'L ESCANDALL','L OSTIA','MAIRETA CUARTA','MAIRETA III','MAR VELLA',
                     'NUS','ORMARANT','SANT PAU','LA FERROSA']
    
    # ADDING THE ADDITIONAL DATA IN THE ORIGINAL DATAFRAME
    with print_with_time('Adding additional data to the DataFrame'):
        # Create dictionary of the data we want to add from Fishing Fleet on the Net
        gear_main = df_vessel_info.groupby(['Vessel Name'])['Gear Main Code'].first().to_dict()
        gear_sec = df_vessel_info.groupby(['Vessel Name'])['Gear Sec Code'].first().to_dict()
        ton_gt = df_vessel_info.groupby(['Vessel Name'])['Ton Gt'].first().to_dict()
        ton_oth = df_vessel_info.groupby(['Vessel Name'])['Ton Oth'].first().to_dict()
        power_main = df_vessel_info.groupby(['Vessel Name'])['Power Main'].first().to_dict()
        power_aux = df_vessel_info.groupby(['Vessel Name'])['Power Aux'].first().to_dict()
        const_year = df_vessel_info.groupby(['Vessel Name'])['Construction Year'].first().to_dict()
        port = df_vessel_info.groupby(['Vessel Name'])['Port Name'].first().to_dict()
        
        # Create new column based on data from Fishing Fleet on the Net
        df['Gear Main'] = df['Nombre'].map(gear_main)
        df['Gear Sec'] = df['Nombre'].map(gear_sec)
        df['Ton Gt'] = df['Nombre'].map(ton_gt)
        df['Ton other'] = df['Nombre'].map(ton_oth)
        df['Power Main'] = df['Nombre'].map(power_main)
        df['Power Aux'] = df['Nombre'].map(power_aux)
        df['Construction Year'] = df['Nombre'].map(const_year)
        df['Port Name'] = df['Nombre'].map(port)
        
        
        # Create a new Column that specifies whether the ship vessel is found in each dataset (BOE, Giulia, VMS)
        df['BOE_shrimp'] = df['Nombre'].isin(shrimp_BOE)
        df['Giulia_shrimp'] = df['Nombre'].isin(shrimp_Giulia)
        df['VMS'] = df['Nombre'].isin(VMS_Palamos or VMS_Arenys or VMS_Blanes or VMS_Barcelona)
        
        # Create extra columns that Trawling and Haul_id that will be needed in the future
        df['Trawling'] = np.zeros(len(df), dtype=bool)
        df['Haul id'] = np.full(len(df), np.nan, dtype=np.float32)    
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

def get_chunk_indices(a):
    a = np.asarray(a)
    diffs = a[:-1] != a[1:] # True when two consecutive values are different
    indices = np.arange(len(a))
    start_indices = indices[1:][diffs].tolist() # index of the first different value
    start_indices.insert(0,0) # insert first index
    end_indices = indices[:-1][diffs].tolist() # index of the last 'equal' value of the chunk
    end_indices.append(len(a)-1) # insert last index
    return start_indices, end_indices
def get_chunks(a):
    a = np.asarray(a)
    start_indices, end_indices = get_chunk_indices(a)
    values = a[start_indices]
    return list(zip(start_indices, end_indices, values))

def identify_trawling(df, min_duration_false_negative, min_haul, start_trawl_id=1):
    """ 
    Identifies trawling and haul ids.
    
    First:
    Find false-negatives: when vessel decreases/increases speed below/above
    trawling threshold but is actually still trawling.
    These are identified by establishing a maximum time that the vessel can 
    be doing a haul at a slower speed (min_duration_false_negative)
    
    Then:
    Find false-positives: when vessel is navigating at trawling speed but
    not doing a haul.
    This is corrected by establishing a minimum trawling duration (min_haul)
    
    Finally:
    Creates new columns (Trawl, Haul_id) establishing whether vessel is trawling and the haul ID
    
    Parameters:
    df = DataFrame
    min_duration_false_negative = duration that these events take in minutes (maximum duration)
    min_haul = minimum duration of a haul (time in minutes)
    
    Returns DataFrame with the additional columns of 'Sog criteria', 'Trawling'
    and 'Haul id'
    """
    # Convert column into datetime to be able to recognize duration of each section
    df['Fecha_Posicion_min'] = pd.to_datetime(df['Fecha_Posicion_min'])
    # Convert min_duration_false_negative into minutes (time format)
    min_duration = pd.Timedelta(minutes = min_duration_false_negative)
    # Get chunks of start and end indexes of each criteria. get_chunks returns 
    # a tuple of 3 values: start_index, end_index, classification (in this case: 0,1,2)
    with print_with_time('Identifying false-negatives'):
        classify_trawling_list = get_chunks(df['Sog criteria'])
        # Check if 0 (low speed) or 2 (high speeds) are between 1 (trawling speed) and its duration.
        # If the duration of these reductions in speeds (between trawling) are less
        # than the specified time criteria, it is converted into 1 (trawling speed)
        for i in range(1, len(classify_trawling_list)-2):
            prev_class = classify_trawling_list[i-1][2]
            next_class = classify_trawling_list[i+1][2]
            if prev_class == 1 and next_class == 1: # If current classification is between two trawling classifications
                start,end = classify_trawling_list[i][0:2]
                if df['Fecha_Posicion_min'][end] - df['Fecha_Posicion_min'][start] <= min_duration:
                    classify_trawling_list[i] = (start, end, 1)
        # Decompresses the classification into a new column, correcting the false-negatives
        sog_crit_2 = np.zeros(len(df),dtype=int)
        for element in classify_trawling_list:
            sog_crit_2[element[0]:element[1]+1] = element[2]
        # Get chunks of start and end indexes of each criteria
        classify_trawling_list_2 = get_chunks(sog_crit_2)

    with print_with_time('Identifying false-positives, Trawling, and Haul_id'):
        # Converts min_haul into datetime format
        min_haul = pd.Timedelta(minutes = min_haul)
        # Checks whether the trawl duration is greater than the minimum duration of 
        # haul and saves the information in a new column
        cnt = 0
        for i in classify_trawling_list_2:
            if i[2] == 1:
                if df['Fecha_Posicion_min'][i[1]]-df['Fecha_Posicion_min'][i[0]] > min_haul:
                    df['Trawling'][i[0]:i[1]+1] = True
                    df['Haul id'][i[0]:i[1]+1] = cnt + start_trawl_id
                    cnt += 1
    return df, cnt

#%% Calling functions
file_dir = 'DATA/Output/'

dir_output = os.path.join(file_dir, 'Output_trawl')
os.makedirs(dir_output, exist_ok=True)

df = pd.DataFrame() # Create empty DataFrame 

cnt = 0
for file in os.listdir(file_dir): # Open all Excel files in the directory 
    if file[-3:] == 'xls':
        with print_with_time('Opening file ',file):
            df = pd.read_excel(os.path.join(file_dir, file))
        df = additional_data(df)
        df['Sog criteria'] = classify_sog(df['Sog'], 1.0, 3.8)
        df, n_trawls = identify_trawling(df, 5, 10, cnt+1)
        cnt += n_trawls
        new_file = file[:-4]+'_trawl.xls'
        with print_with_time('Saving file ', new_file):
            df.to_excel(os.path.join(dir_output,new_file), index=None)
