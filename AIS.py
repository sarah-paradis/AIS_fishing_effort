# Script where all the functions required to process AIS data is stored

import os
import numpy as np
import pandas as pd
import timeit
from contextlib import contextmanager
from pylab import plot, hist, legend
from scipy.optimize import curve_fit
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm


@contextmanager
def print_with_time(*s):
    """
    Function to calculate the time it takes for a process to complete.
    To use, type "with print_with_time():"
    You should add in parenthesis the action you're doing as a string.
    The function prints out the action and the time it took to complete it in seconds
    """
    print(*s, end="", flush=True)
    start = timeit.default_timer()
    yield
    print("\t[%.2fs]" % (timeit.default_timer() - start), flush=True)


def read_all_data(file_dir):
    """
    Function that reads all the files (Excels) in a directory as one DataFrame.

    file_dir = file directory (string)
    """
    df = pd.DataFrame()  # Create empty DataFrame
    for file in os.listdir(file_dir):  # Open all files in the directory
        if file.endswith('xls') or file.endswith('xlsx') or file.endswith('csv'):
            data = read_df(file, file_dir)
            df = df.append(data, sort=False)
    df = df.reset_index()
    return df

def save_df(df, file, dir_output=None):
    """
    Saves DataFrame (.xls or .csv)
    """
    # Makes sure the output directory (file_dir) exists, if not, it creates it.
    if dir_output is not None:
        os.makedirs(dir_output, exist_ok=True)
        file = os.path.join(dir_output, file)

    file_type = file[-4:]
    if file_type in ['.csv', '.xls']:
        with print_with_time('Saving file as ' + file):
            if file_type == '.xls':
                df.to_excel(file, index=None)
            if file_type == '.csv':
                df.to_csv(file, index=None)
    else:
        raise TypeError('Data not .xls or .csv type')

def read_df(file, file_dir=None):
    """
    Reads a file (either csv or excel) as a DataFrame

    """
    if file_dir is not None:
        file = os.path.join(file_dir, file)
    file_type = file[-4:]
    if file_type in ['.csv', '.xls', 'xlsx']:
        with print_with_time('Opening file ' + file):
            if file_type == '.csv':
                df = pd.read_csv(file)
            elif file_type == '.xls':
                df = pd.read_excel(file)
            elif file_type == 'xlsx':
                df = pd.read_excel(file)
    else:
        raise TypeError('File not recognized as a .csv or .xls')
    return df

def save_all_data(df, dir_output='Output', output_type='csv'):
    """
    Exports a DataFrame into Excel (.xls) files separated by month, such as "2019_02.xls".
    If the number of rows (entries) exceeds the maximum Excel file size, the file is separated into several Excel files
    and the name of the file indicates the maximum day included in that file, such as "2019_02_16.xls"

    df = DataFrame to be exported

    dir_output = directory where the data will be saved. If None, saves in the same working directory. If the
    directory doesn't exist, it creates it.

    output_type = type of output file, either Excel (.xls or .xlsx) or csv. Default is csv.
    """

    # Makes sure the output directory (file_dir) exists, if not, it creates it.
    os.makedirs(dir_output, exist_ok=True)

    # Extract the months in the files to later create individual dataframes
    # and the names of the output .csv files for each month
    if type(df['Month'][0]) is not str:
        months = sorted(list(df.Month.unique()))  # list of the months from which the DataFrames will be extracted
        months_name = [month.strftime('%Y_%m') for month in months]  # list of the names of the months for the .csv files
    else:
        months = sorted(list(df.Month.unique()))
        months_name = months
    days = []
    for month in months:
        day_month = sorted(list(df['Date'][df['Month'] == month].unique()))
        days.append(day_month)

    # Exporting files
    if not output_type.startswith("."):
        output_type = "."+output_type

    if output_type == '.csv':
        max_rows = np.inf  # no maximum rows in a csv file
    elif output_type == '.xlsx':
        max_rows = 1000000  # actually it is 1048576
    elif output_type == '.xls':
        max_rows = 65000  # actually it is 65536 (2^16)
    else:
        raise TypeError(output_type + " is not a valid output type")

    for month, month_name, idx in zip(months, months_name, range(len(months))):
        df_month = df[df['Month'] == month]  # Create a new DataFrame with the data of that specific month
        file_name = month_name + output_type
        if len(df_month) < max_rows:
            with print_with_time('Exporting ' + file_name + ' to file'):
                if output_type == '.xls' or output_type == '.xlsx':
                    df_month.to_excel(os.path.join(dir_output, file_name), index=None)
                elif output_type == '.csv':
                    df_month.to_csv(os.path.join(dir_output, file_name), index=None)
        else:
            assert output_type != '.csv'
            with print_with_time('Exporting %s in different %s files' % (file_name, output_type)):
                while len(days[idx]) > 0:  # save files
                    for i in range(len(days[idx])):  # days
                        if len(df_month[df_month['Date'].isin(days[idx][:i + 1])]) > max_rows:
                            i = i - 1
                            break
                    if type(df['Month'][0]) is not str:
                        file_name = month_name + '_' + days[idx][i].strftime('%d') + output_type
                    else:
                        file_name = month_name + '_' + days[idx][i][-2:] + output_type
                    print('\n Exporting ' + file_name)
                    file_name = os.path.join(dir_output, file_name)
                    df_save = df_month[df_month['Date'].isin(days[idx][:i + 1])]
                    assert len(df_save) > 0, "Too many rows for day %s. Cannot save in %s" % (days[idx][
                        i + 1].strftime('%d'), output_type)
                    df_save.to_excel(file_name, index=None)
                    days[idx] = days[idx][i + 1:]


def fishing_effort_min(df, datetime_column, name_column,
                       additional_columns):
    """
    Opens all AIS files and extracts the first entry of each minute for
    each vessel during the sampling period.

    df = DataFrame that needs to be processed

    datetime_column = name of column in dataframe that has the date and
    time of vessel positioning (type: string)

    name_column = name of column in dataframe that has the name/code of
    each vessel

    additional_columns = list with the name of the columns in the dataframe
    that would be exported

    Returns a dataframe with data every minute
    """

    # Create new date-time column rounded to minutes
    with print_with_time('Converting to "date-time" format'):
        new_column = datetime_column + '_min'
        df[new_column] = pd.to_datetime(df[datetime_column], dayfirst=True)
    with print_with_time('Rounding time to minutes'):
        df[new_column] = df[new_column].values.astype('<M8[m]')

    # Extract the first entry of each minute for each vessel and each day
    with print_with_time('Extracting the first entry of each minute for each vessel and day'):
        df_min = df.groupby([name_column, new_column]).first()

    df_min = df_min.reset_index()  # Reset index

    # Organize DataFrame eliminating the unnecessary columns and putting them in order
    df_min = df_min[[name_column] + [datetime_column] + [new_column] + additional_columns]

    # Create a new column with only the month and year
    with print_with_time('Grouping data by month and day'):
        df_min['Month'] = df_min[new_column].dt.to_period('M')
        df_min['Date'] = df_min[new_column].dt.to_period('D')
        df_min[datetime_column] = pd.to_datetime(df_min[datetime_column], dayfirst=True)
        df_min['Time'] = df_min[datetime_column].dt.time

        # Extract the months in the files to later create individual DataFrames
        # and the names of the output .csv files for each month
        months = sorted(list(df_min.Month.unique()))  # list of the months from which the DataFrames will be extracted
        months_name = [month.strftime('%Y_%m') for month in months]  # list of the names of the months for the .csv files
        days = []
        for month in months:
            day_month = sorted(list(df_min['Date'][df_min['Month'] == month].unique()))
            days.append(day_month)
    return df_min


def all_vessels(df, name_column, output_name, file_dir=None):
    """
    Creates an Excel file of all the vessels that were found in the DataSet

    df = name of complete DataFrame. If the data is separated into different files, open them all using
    the read_all_data function found in read_all_data.py

    name_column = column name where the name of the vessels are stored

    output_name = name of Excel file with all the vessels in the DataSet

    file_dir = directory where the Excel will be saved as
    """
    if not output_name.endswith(".xlsx"):
        output_name += ".xlsx"
    if file_dir is not None:
        # makes sure the output directory (file_dir) exists, if not, it creates it.
        os.makedirs(file_dir, exist_ok=True)
        # joins path name of the directory with the output file name if the output directory file is given
        file_name = os.path.join(file_dir, output_name)
    else:
        file_name = output_name
    with print_with_time('Extracting vessels in the area'):
        # Extract the names of the vessels
        # Extract the first entry of all vessels, to get their identifier
        df_vessels = df.groupby([name_column]).first()
        df_vessels = df_vessels.reset_index()  # Reset index
        df_vessels.to_excel(file_name, index=None)
    return df_vessels


def additional_data(df, name_column, file_fleet, Fleet_column):
    """
    Add all the additional data (Gear type, Gt, Power (kw), Construction year,
    Port). Additional data are extracted from Fleet Register on the Net:
        http://ec.europa.eu/fisheries/fleet/index.cfm

    df = Dataframe with data from all fishing vessels

    name_column = name of column in original DataFrame that has the name/code of
    each vessel.

    file_fleet = file from Fleet Register that has all the data to be extracted

    Fleet_column = name of column in Fleet Register on the Net that has the
    name/code of each vessel.

    Returns a DataFrame
    """

    # Fishing fleet on the net
    global df_vessel_info
    if file_fleet.endswith('csv'):
        df_vessel_info = pd.read_csv(file_fleet, engine='python', delimiter=';')
    elif file_fleet[-3:] == 'xls' or file_fleet[-4:] == 'xlsx':
        df_vessel_info = pd.read_excel(file_fleet)

    # ADDING THE ADDITIONAL DATA IN THE ORIGINAL DATAFRAME
    with print_with_time('Adding additional data to the DataFrame'):
        # Create dictionary of the data we want to add from Fishing Fleet on the Net
        gear_main = df_vessel_info.groupby([Fleet_column])['Gear Main Code'].last().to_dict()
        gear_sec = df_vessel_info.groupby([Fleet_column])['Gear Sec Code'].last().to_dict()
        ton_gt = df_vessel_info.groupby([Fleet_column])['Ton Gt'].last().to_dict()
        ton_oth = df_vessel_info.groupby([Fleet_column])['Ton Oth'].last().to_dict()
        power_main = df_vessel_info.groupby([Fleet_column])['Power Main'].last().to_dict()
        power_aux = df_vessel_info.groupby([Fleet_column])['Power Aux'].last().to_dict()
        const_year = df_vessel_info.groupby([Fleet_column])['Construction Year'].last().to_dict()
        port = df_vessel_info.groupby([Fleet_column])['Port Name'].last().to_dict()

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


def no_data_fleet(df, column_name, vessel_name, output_name, file_dir=None):
    """
    Extracts an Excel with the name of all the vessels that were not paired with the data from Fleet Register database

    df = DataFrame that has already been paired with Fleet Register database

    column_name = name of column where parameters from Fleet Register were added

    vessel_name = name of column with the Vessel name/code

    output_name = name of Excel file to be exported

    file_dir = name of directory where the Excel file will be saved in. If None, the Excel file will be saved in the
    same working directory

    Returns a DataFrame
    """

    if not output_name.endswith(".xlsx"):
        output_name += ".xlsx"
    if file_dir is not None:
        # makes sure the output directory (file_dir) exists, if not, it creates it.
        os.makedirs(file_dir, exist_ok=True)
        # joins path name of the directory with the output file name if the output directory file is given
        file_name = os.path.join(file_dir, output_name)
    else:
        file_name = output_name

    with print_with_time('Extracting vessels with no data in Fleet Register'):
        df_no_data = df[df.loc[:, column_name].isna()]
        # Extracts only the first entry of the vessels that were not included
        df_no_data = df_no_data.groupby(vessel_name).first()
        df_no_data = df_no_data.reset_index()
        df_no_data.to_excel(file_name, index=None)
    return df_no_data


def filter_trawlers(df, column_gear, gear_name = 'OTB'):
    """
    Filters DataFrame for a specific gear type. Default extracts vessels with "OTB" (otter trawl boards).

    df = DataFrame with the whole dataset

    column_gear = name of column that has the fishing gear type.
    If there are multiple columns, introduce names as a list

    gear_name = name of the gear type to be extracted. Default is 'OTB'.
    """
    with print_with_time('Extracting gear type'):
        # Extract by gear type
        assert type(column_gear) in (list, str), "Data with fishing gear is not correctly given"
        if type(column_gear) is list:
            # Checks if any of the columns has the gear_name as fishing gear
            df_trawl = df[(df.loc[:, column_gear] == gear_name).any(axis=1)]
        else:
            # Checks if vessel fishing gear has the gear_name
            df_trawl = df[df.loc[:, column_gear] == gear_name]
        # Reset index since filtering out samples must have altered the index of DataFrame
        df_trawl = df_trawl.reset_index().drop('index', axis=1)
    return df_trawl


def define_gauss_distribution(data):
    """
    Models two Gaussian distributions of the dataset (speed of bottom trawlers) to identify the speed at which
    bottom trawlers operates

    data = Series where the speed of trawlers is saved in, usually labelled "Sog" for "Speed over ground"
    """
    with print_with_time('Defining Gaussian bimodal distribution of trawling speeds'):
        # Create a list of 101 Sog values that go from the minimum value to the maximum value at equal intervals
        x = np.linspace(min(data), max(data), num=101)
        # Calculates the frequency of the previous values
        y = np.array(data.value_counts(bins=100, sort=False))
        x = (x[1:] + x[:-1]) / 2  # for len(x)==len(y)

        def gauss(x, mu, sigma, A):
            return A * np.exp(-(x - mu) ** 2 / 2 / sigma ** 2)

        def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2):
            return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2)

        # expected are the initial expected values: x, mu1, sigma1, A1, mu2, sigma2, A2
        expected = [3, 1, max(y[:50]), 11, 1, max(y[50:])]

        # Calculates the parameters of the two bimodal distribution
        params, covar = curve_fit(bimodal, x, y, expected)
        sigma = np.sqrt(np.diag(covar))

        # Creates a DataFrame of the parameters
        df_params = pd.DataFrame(data={'params': params, 'sigma': sigma})
    return df_params

def draw_histogram(data, df_speed=None, fig_name=None, format_fig='eps'):
    """
    Creates a histogram of a dataset. Ideally used to plot speed of fishing vessels to identify trawling speed

    data = series where the data is saved (usually saved under a column named "Sog")

    df_speed = DataFrame where the information on the speed a trawler operates at or navigates is given.
    If it is not specified, it will not annotate the histogram.

    fig_name = name of the graph to be saved as. If it is not specified, the histogram will not be saved.

    format_fig = format of the figure to be saved. Default is set to "eps". Also accepts "jpeg", "tiff", etc.

    """
    with print_with_time('Plotting histogram of dataset'):
        ### Create a histogram of the whole dataset
        # Set style
        sns.set_style('whitegrid')
        sns.set_style('ticks')
        sns.set_context('notebook', font_scale=1.25)
        # Plot the figure
        ax = sns.distplot(data, kde=False, color='grey')
        ax.set(xlabel='Speed over ground')
        ax.set(xlim=(0, 16))
        ax.set(ylabel='Frequency')
        y_max = max(list(h.get_height() for h in sns.distplot(data).patches))
        ax.set(ylim=(0, y_max*1.5))
        if df_speed is not None:
            # Establish parameters of trawling
            avg_speed_trawling = df_speed.iloc[0,0]
            std_dev_trawling = df_speed.iloc[1,0]
            max_freq_trawling = df_speed.iloc[2,0]
            min_speed = round((df_speed.iloc[0, 0] - 2 * df_speed.iloc[1, 0]), 1)
            max_speed = round((df_speed.iloc[0, 0] + 2 * df_speed.iloc[1, 0]), 1)

            # Annotate the graph with trawling information
            ax.text((min_speed + std_dev_trawling * 0.5), y_max*1.25, 'Trawling', color='blue', size=13.5)
            ax.text((min_speed + std_dev_trawling * 0.5), y_max*1.1, '%.1f - %.1f kn' %(min_speed, max_speed), color='blue', size=10)

            # Estabilsh parameters of navigating
            avg_speed_nav = df_speed.iloc[3,0]
            std_dev_nav = df_speed.iloc[4,0]
            max_freq_nav = df_speed.iloc[5,0]

            # Annotate the graph with navigating information
            ax.text((avg_speed_nav - std_dev_nav), y_max*0.5, 'Navigating', color='black')

            # Plot Gaussian distribution between 0 and 8 with .0001 steps.
            x_axis = np.arange(0, 8, 0.0001)
            ax.plot(x_axis, norm.pdf(x_axis, avg_speed_trawling, std_dev_trawling)*200000, linestyle='-', color='blue')

            # Plot limits where we consider trawling speeds
            plt.axvline(min_speed, color='blue', linestyle='--')
            plt.axvline(max_speed, color='blue', linestyle='--')
        fig = ax.get_figure()
        fig.tight_layout()
    if fig_name is not None:
        with print_with_time('Saving histogram as '+fig_name+'.'+format_fig):
            file_name = fig_name+'.'+format_fig
            fig.savefig(file_name, format=format_fig, dpi=500)


def classify_sog(sog_col, min_trawl, max_trawl, min_nav):
    """ Classifies Speed Over Ground (SOG) into low speed (0), medium speed (1), and high speed (2)
    where medium speed is the trawler speed when operating (doing a haul)
    sog_col = column where SOG is saved as in the DataFrame
    min_trawl = minimum trawling speed
    max_trawl = maximum trawling speed
    min_nav = minimum navigating speed
    """
    with print_with_time('Classifying by Speed Over Ground'):
        class_sog = np.zeros(len(sog_col), dtype=int)
        class_sog[(sog_col >= min_trawl) & (sog_col <= max_trawl)] = 1
        class_sog[(sog_col > max_trawl) & (sog_col < min_nav)] = 2
        class_sog[sog_col >= min_nav] = 3
    return class_sog


def get_chunk_indices(a, in_same_chunk_fn=lambda x, y: x == y):
    len_a = len(a)
    if type(a) is pd.DataFrame:
        a = a.iloc
    diffs = [not in_same_chunk_fn(a[i], a[i + 1]) for i in range(len_a - 1)]
    indices = np.arange(len_a)
    start_indices = indices[1:][diffs].tolist()  # index of the first different value
    start_indices.insert(0, 0)  # insert first index
    end_indices = indices[:-1][diffs].tolist()  # index of the last 'equal' value of the chunk
    end_indices.append(len_a - 1)  # insert last index
    return list(zip(start_indices, end_indices))


def get_same_period_fn(minutes_diff, criteria, datetime_column):
    def same_period(x, y):
        """ Evaluates entries that are registered in less than 'minutes_diff' in order to decide if they
        belong in the same chunk

        minutes_diff = time interval to check a chunk

        criteria  = list of column names to check equalness

        datetime_column = column with datetime information
        """
        time_diff = pd.Timedelta(minutes=minutes_diff)
        res = y[datetime_column] - x[datetime_column] < time_diff

        for crit in criteria:
            res = res and (x[crit] == y[crit])
        return res

    return same_period


def identify_trawling(df, datetime_column, name_column, min_duration_false_positive,
                      min_duration_false_negative, min_haul, AIS_turn_off, start_trawl_id=1):
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
    name_column = name of column in dataframe with the name/code ID of the vessel
    min_duration_false_positive = duration of continued entries classified as trawling need to take place
    min_duration_false_negative = duration that these events take in minutes (maximum duration)
    min_haul = minimum duration of a haul (time in minutes)
    AIS_turn_off = maximum time that the AIS is turned off before considering it belongs to a different haul

    Returns DataFrame with the additional columns of 'Sog criteria', 'Trawling'
    and 'Haul id'
    """

    # Create extra columns that Trawling and Haul_id that will be needed in the future
    df['Sog_criteria_temp'] = df['Sog_criteria']
    df['Trawling'] = np.zeros(len(df), dtype=bool)
    df['Haul id'] = np.full(len(df), np.nan, dtype=np.float32)

    # Convert column into datetime to be able to recognize duration of each section
    df[datetime_column] = pd.to_datetime(df[datetime_column])

    with print_with_time('Getting set of trawling criteria'):
        # Creates a list of tuples (index start, index end) of all the 'chunks' based on same SOG criteria (0,1,2)
        # considering that the vessel's AIS has been turned off during less than 'AIS_turn_off'.
        classify_trawling_list = get_chunk_indices(df, get_same_period_fn(AIS_turn_off,
                                                                          ['Sog_criteria',
                                                                           name_column,
                                                                           'Date'],
                                                                          datetime_column))
    with print_with_time('Identifying false-positives'):
        # Converts min_haul into datetime format
        min_duration_false_positive = pd.Timedelta(minutes=min_duration_false_positive)
        for i, j in classify_trawling_list:
            if df['Sog_criteria'][i] == 1:
                if df[datetime_column][j] - df[datetime_column][i] > min_duration_false_positive:
                    df.loc[i:j, 'Sog_criteria_temp'] = 1
                else:
                    df.loc[i:j, 'Sog_criteria_temp'] = 0

    with print_with_time('Identifying false-negatives'):
        # Convert min_duration_false_negative into minutes (time format)
        min_duration = pd.Timedelta(minutes=min_duration_false_negative)
        # Check if 0 (low speed) or 2 (high speeds) are between 1 (trawling speed) and its duration.
        # If the duration of these reductions in speeds (between trawling) are less
        # than the specified time criteria, it is converted into 1 (trawling speed)
        for idx in range(1, len(classify_trawling_list) - 2):
            current_class = df['Sog_criteria'][classify_trawling_list[idx][1]] # Checks Sog criteria of current chunk
            if current_class == 0 or current_class == 2:
                # prev_trawl = df['Trawling'][classify_trawling_list[idx - 1][1]]  # Checks Trawling of previous chunk
                # next_trawl = df['Trawling'][classify_trawling_list[idx +1][1]] # Checks Trawling of next chunk
                prev_class = df['Sog_criteria_temp'][classify_trawling_list[idx - 1][1]]  # Checks Sog criteria of previous chunk
                next_class = df['Sog_criteria_temp'][classify_trawling_list[idx + 1][1]]  # Checks Sog criteria of following chunk
                if prev_class == 1 and next_class == 1:
                # if prev_class == 1 and next_class == 1 and (prev_trawl == True or next_trawl == True):
                    start, end = classify_trawling_list[idx]
                    if df[datetime_column][end] - df[datetime_column][start] <= min_duration:
                        df.loc[start:end, 'Sog_criteria_temp'] = 1
    with print_with_time('Getting new set of trawling criteria'):
        # Creates a list of tuples (index start, index end) of all the 'chunks' based on same SOG criteria (0,1,2)
        # considering that the vessel's AIS has been turned off during less than 'AIS_turn_off'.
        classify_trawling_list = get_chunk_indices(df, get_same_period_fn(AIS_turn_off,
                                                                          ['Sog_criteria_temp',
                                                                           name_column,
                                                                           'Date'],
                                                                          datetime_column))
    with print_with_time('Identifying trawling activity after the previous corrections'):
        # Converts min_haul into datetime format
        min_haul = pd.Timedelta(minutes=min_haul)
        for i, j in classify_trawling_list:
            if df['Sog_criteria_temp'][i] == 1:
                if df[datetime_column][j] - df[datetime_column][i] > min_haul:
                    df.loc[i:j, 'Trawling'] = True

    with print_with_time('Identifying Haul_id'):
        cnt = 0
        new_trawling_list = get_chunk_indices(df,
                                              get_same_period_fn(AIS_turn_off,
                                                                 ['Trawling',
                                                                  name_column,
                                                                  'Date'],
                                                                 datetime_column))
        for i, j in new_trawling_list:
            if df['Trawling'][i] == True:
                df.loc[i:j, 'Haul id'] = cnt + start_trawl_id
                cnt += 1
    return df, cnt

