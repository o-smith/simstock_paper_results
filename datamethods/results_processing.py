"""
File containing functions to read in various files associated
with EnergyPlus outputs in order analyse simulation results.
"""

import os, re, csv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TimeSeries:
    """
    Class in which to store time series data. 

    Attributes
    ----------
    t: list
        List of time points in date-time format.

    x: list
        List of values in the timeseries.
    """
    start_date = datetime(2023, 1, 1, 0, 0, 0)

    def __init__(self) -> None:

        # Create a list of datetime objects at hourly intervals for a whole year
        self.t = [self.start_date + timedelta(hours=i) for i in range(365*24)]

        # And an empty list to store values
        self.x = []

    def total(self) -> float:
        """
        Return the sum of the timeseries.
        """
        return sum(self.x)
    
    def max(self) -> float:
        """
        Return the maximum value of a timeseries.
        """
        return max(self.x)
    
    def min(self) -> float:
        """
        Return the minimum value of a timeseries.
        """
        return min(self.x)
    
    def mean(self) -> float:
        """
        Return the mean value of a timeseries.
        """
        return np.mean(self.x)
    
    def weekly_aggregates(self):
        """
        Aggregate time series data to cumulative weekly 
        values with corresponding week datetimes.

        Returns
        -------
            List of cumulative weekly values.
            List of corresponding week datetimes.
        """
        # Create a DataFrame from the lists
        df = pd.DataFrame({'value': self.x, 'datetime': self.t})

        # Set 'datetime' as the index
        df.set_index('datetime', inplace=True)

        # Resample the data to weekly frequency and sum the values
        weekly_data = df.resample('W').sum()

        # Calculate cumulative sum
        cumulative_weekly_data = weekly_data.cumsum()

        return cumulative_weekly_data
    

class TemperatureTimeSeries(TimeSeries):
    """
    Class to store temperature time series data.
    Inherits from TimeSeries class. 

    Attributes
    ----------
    t: list
        List of time points in date-time format.

    x: list
        List of values in the timeseries.
    """
    
    def __init__(self) -> None:

        # Create a list of datetime objects at hourly intervals for a whole year
        self.t = [self.start_date + timedelta(hours=i) for i in range(365*24)]

        # And an empty list to store values
        self.x = []

        # Minimum and maximum comfortable temperatures
        self.min_temp = 16.0
        self.max_temp = 26.0

    def hours_above_max(self) -> int:
        return sum(1 for value in self.x if value > self.max_temp)
    
    def hours_below_min(self) -> int:
        return sum(1 for value in self.x if value < self.min_temp)
    
        

def add_timeseries(ts1: TimeSeries, ts2: TimeSeries) -> TimeSeries:
    """
    A function to add together two time series.
    """
    # Handle base cases
    if len(ts1.x) == 0:
        summed = ts2.x
    elif len(ts2.x) == 0:
        summed = ts1.x

    # Add together time series
    else: 
        summed = [x1 + x2 for x1, x2 in zip(ts1.x, ts2.x)]

    # Bundle into a new TimeSeries object and return
    out_ts = TimeSeries()
    out_ts.x = summed
    return out_ts



def parse_block(file_path: str, block_name: str) -> dict:
    """
    This is a function that can extract blocks from an IDF
    file and return them as a dictionary.

    Paramters
    ---------
    file_path: str
        The file path to the IDF.

    block_name: str
        The name of the block to extract from the IDF.

    Returns
    -------
    results: dict
        A dictionary containing the contents of the requested
        block from the IDF.
    """
    result = {}
    in_block = False

    # Read IDF content from the file
    with open(file_path, 'r') as file:
        idf_content = file.read()

    # Split the IDF content into lines
    lines = idf_content.split('\n')

    # Iterate through lines to find the specified block
    for line in lines:

        # Check if we are in the block
        if line.strip().startswith(block_name + ','):
            in_block = True

        # This is a check for the end of the block
        elif in_block and not line.strip():
            return result
        
        # Look at block contents
        elif in_block and line.strip():
            # Split each line into key and value using the delimiter
            parts = line.split('!-')
            key = parts[1].strip()
            value = parts[0].strip(',; ')
            
            # Convert value to appropriate type (int or float if possible)
            if value.isdigit():
                value = int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                value = float(value)

            # Add this line's contents to the dictionary
            result[key] = value

    return result


def extract_total_energy_from_csv(file_path: str) -> tuple[float, float]:
    """
    Function to extract the the total_energy and the total_energy_per_area, 
    both measured in kWh, from a eplustbl.csv file. Note that these values 
    are probably going to be aggregates over an entire built island.

    Parameters
    ----------
    file_path: str
        The file path the to relevent eplustbl.csv file.

    Returns
    -------
    total_energy: float
        The total energy usage over the year (kWh).

    total_energy_per_floor area: float
        The total energy usage over the year (kWh), normalised
        by floor area.
    """
    Total_Energy = None
    Energy_Per_Area = None

    # Open the csv file and read one row at a time. If the row
    # has a more than 2 fields, check if it is the row for total
    # site energy and if so extract the values.
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row:
                if len(row)>2:
                    if row[1] == "Total Site Energy":
                        Total_Energy = float(row[2])
                        Energy_Per_Area = float(row[3])

    return Total_Energy, Energy_Per_Area


def get_building_files(dir: str) -> dict:
    """
    Function to return a dictionary mapping built island numbers
    to output file numbers.

    The function works by looking at all of the IDF files in the 
    directory dir, and then looks inside them to find the
    built island number. 

    Parameters
    ----------
    dir: str
        File path to the directory containing IDF files.

    Returns
    -------
    bi_to_file: dict
        A dictionary whose keys are built island numbers
        and values are file numbers.
    """

    # Compile a dictionary whose keys are the building name and 
    # vars are the file number
    bi_to_file_dict = {}

    # Regular expression pattern to extract the integer from the file name
    pattern = re.compile(r'built_island_(\d+)\.idf', re.IGNORECASE)

    # Iterate over all files in the directory
    for root, _, files in os.walk(dir):
        for file in files:
            # Check if the file has a ".idf" extension and 
            # matches the naming pattern
            match = pattern.match(file)
            if match:
                # Extract the integer from the file name
                island_number = int(match.group(1))

                # Construct the full path to the IDF file
                file_path = os.path.join(root, file)

                # Get the building name from this idf file
                bi_name = parse_block(file_path, "Building")["Name"]

                bi_to_file_dict[bi_name] = island_number

    return bi_to_file_dict


def get_scus_elec_timeseries(scu: int, out_dir: str) -> dict:
    """
    A function that takes a scu and returns a dictionary whose
    key val pairs are floor number and timeseries.

    Parameters
    ----------
    scu: int
        The SCU in question.

    out_dir: str
        The output director in which this built island's results
        can be found. This will be of the form:
            .../built_island_x_ep_outputs

    Returns
    -------
    timeseries: dict
        A dictionary whose key value pairs are floor numbers and
        timeseries objects.
    """
    # Open the mtr file and read it in one line at a time
    fpath = os.path.join(out_dir, "eplusout.mtr")
    with open(fpath, 'r') as file:
        mtr_content = file.read()

    # Split the content into lines
    lines = mtr_content.split('\n')

    # Initialise a dictionary of data codes
    data_codes = {}

    # Iterate over the lines sequentially
    for l in lines:
        line = l.split(",")

        # Check if we have come to the end of the file's data dictionary
        if line[0].strip() == "End of Data Dictionary":
            break
        
        # Pattern to looks for
        pattern = r'Electricity:Zone:(?P<scu>\d+)_FLOOR_(?P<floor_num>\d+) \[J\] !Hourly'

        # If the line has this pattern, then see if the scu number matches
        # the scu we are looking for
        match = re.search(pattern, line[2].strip())
        if match:
            if match.group("scu").strip() == str(scu):

                # If we have found the relevent line, then extract
                # the floor num and the data code
                floor_num = match.group("floor_num")
                data_code = line[0]
                
                # Store them in the dictioary of data codes
                data_codes[floor_num] = data_code

    # We should now have a dictionary that tells us the codes
    # for where the energy usage data is for each floor
    # So lets start the output dictionary whose keys will be
    # The floor numbers and values the timeseries
    out_dict = {}
    for floor_no in data_codes.keys():
        past_data_dict = False
        time_series = TimeSeries()
        for l in lines:
            line = l.split(",")

            # Skip the data dictionary at the head of the file
            if line[0].strip() == "End of Data Dictionary":
                past_data_dict = True

            # If we are already past the head of the file
            # we can start reading the lines normally
            if past_data_dict:

                # Check if the line has the corect code
                if str(line[0].strip()) == str(data_codes[floor_no]):

                    # Get the energy usage value and convert to kWh
                    time_series.x.append(float(line[1].strip())/3.6e+6)

        # Now save this time-series in the output dictionary
        out_dict[int(floor_no)] = time_series

    return out_dict


def get_scus_hvac_timeseries(scu: int, out_dir: str) -> dict:
    """
    A function that takes a scu and returns a dictionary whose
    key val pairs are floor number and timeseries.

    Parameters
    ----------
    scu: int
        The SCU in question.

    out_dir: str
        The output director in which this built island's results
        can be found. This will be of the form:
            .../built_island_x_ep_outputs

    Returns
    -------
    timeseries: dict
        A dictionary whose key value pairs are floor numbers and
        timeseries objects.
    """
    # Open the mtr file and read it in one line at a time
    fpath = os.path.join(out_dir, "eplusout.eso")
    with open(fpath, 'r') as file:
        mtr_content = file.read()

    # Split the content into lines
    lines = mtr_content.split('\n')

    # Initialise a dictionary of data codes
    data_codes = {}

    # Iterate over the lines sequentially
    for l in lines:
        line = l.split(",")

        # Check if we have come to the end of the file's data dictionary
        if line[0].strip() == "End of Data Dictionary":
            break
        
        if len(line) >= 4:
            if line[3].strip() == "Zone Ideal Loads Zone Total Heating Energy [J] !Hourly":

                # Pattern to look for
                pattern = r"(?P<scu>\d+)_FLOOR_(?P<floor_num>\d+)_HVAC"

                # If the line has this pattern, then see if the scu number matches
                # the scu we are looking for
                match = re.search(pattern, line[2].strip())
                if match:
                    if match.group("scu").strip() == str(scu):

                        # If we have found the relevent line, then extract
                        # the floor num and the data code
                        floor_num = match.group("floor_num") + "_heat"
                        data_code = line[0]

                        # Store them in the dictioary of data codes
                        data_codes[floor_num] = data_code

        if len(line) >= 4:
            if line[3].strip() == "Zone Ideal Loads Zone Total Cooling Energy [J] !Hourly":

                # Pattern to look for
                pattern = r"(?P<scu>\d+)_FLOOR_(?P<floor_num>\d+)_HVAC"

                # If the line has this pattern, then see if the scu number matches
                # the scu we are looking for
                match = re.search(pattern, line[2].strip())
                if match:
                    if match.group("scu").strip() == str(scu):

                        # If we have found the relevent line, then extract
                        # the floor num and the data code
                        floor_num = match.group("floor_num") + "_cool"
                        data_code = line[0]

                        # Store them in the dictioary of data codes
                        data_codes[floor_num] = data_code

    # We should now have a dictionary that tells us the codes
    # for where the energy usage data is for each floor
    # So lets start the output dictionary whose keys will be
    # The floor numbers and values the timeseries
    out_dict = {}
    for floor_no in data_codes.keys():

        time_series_heat = TimeSeries()
        time_series_cool = TimeSeries()

        s = floor_no.split("_")
        if s[-1] == "heat":
            passed_data_dict = False
            
            for l in lines:
                line = l.split(",")

                # Skip the data dictionary at the head of the file
                if line[0].strip() == "End of Data Dictionary":
                    passed_data_dict = True

                # If we are already past the head of the file
                # we can start reading the lines normally
                if passed_data_dict:

                    # Check if the line has the corect code
                    if str(line[0].strip()) == str(data_codes[floor_no]):

                        # Get the energy usage value and convert to kWh
                        time_series_heat.x.append(float(line[1].strip())/3.6e+6)

        if s[-1] == "cool":
            passed_data_dict = False
            
            for l in lines:
                line = l.split(",")

                # Skip the data dictionary at the head of the file
                if line[0].strip() == "End of Data Dictionary":
                    passed_data_dict = True

                # If we are already past the head of the file
                # we can start reading the lines normally
                if passed_data_dict:

                    # Check if the line has the corect code
                    if str(line[0].strip()) == str(data_codes[floor_no]):

                        # Get the energy usage value and convert to kWh
                        time_series_cool.x.append(float(line[1].strip())/3.6e+6)

        # Now save this time-series in the output dictionary
        out_dict[int(s[0])] = add_timeseries(time_series_cool, time_series_heat)

    return out_dict


def get_temperature_timeseries(scu: int, out_dir: str) -> dict:
    """
    A function that takes a scu and returns a dictionary whose
    key val pairs are floor number and timeseries.

    Parameters
    ----------
    scu: int
        The SCU in question.

    out_dir: str
        The output director in which this built island's results
        can be found. This will be of the form:
            .../built_island_x_ep_outputs

    Returns
    -------
    timeseries: dict
        A dictionary whose key value pairs are floor numbers and
        timeseries objects.
    """
    # Open the mtr file and read it in one line at a time
    fpath = os.path.join(out_dir, "eplusout.eso")
    with open(fpath, 'r') as file:
        mtr_content = file.read()

    # Split the content into lines
    lines = mtr_content.split('\n')

    # Initialise a dictionary of data codes
    data_codes = {}

    # Iterate over the lines sequentially
    for l in lines:
        line = l.split(",")

        # Check if we have come to the end of the file's data dictionary
        if line[0].strip() == "End of Data Dictionary":
            break
        
        if len(line) >= 4:
            if line[3].strip() == "Zone Mean Air Temperature [C] !Hourly":

                # Pattern to look for
                pattern = r"(?P<scu>\d+)_FLOOR_(?P<floor_num>\d+)"

                # If the line has this pattern, then see if the scu number matches
                # the scu we are looking for
                match = re.search(pattern, line[2].strip())
                if match:
                    if match.group("scu").strip() == str(scu):

                        # If we have found the relevent line, then extract
                        # the floor num and the data code
                        floor_num = match.group("floor_num")
                        data_code = line[0]

                        # Store them in the dictioary of data codes
                        data_codes[floor_num] = data_code

    # We should now have a dictionary that tells us the codes
    # for where the energy usage data is for each floor
    # So lets start the output dictionary whose keys will be
    # The floor numbers and values the timeseries
    out_dict = {}
    for floor_no in data_codes.keys():

        time_series = TemperatureTimeSeries()
        passed_data_dict = False
        
        for l in lines:
            line = l.split(",")

            # Skip the data dictionary at the head of the file
            if line[0].strip() == "End of Data Dictionary":
                passed_data_dict = True

            # If we are already past the head of the file
            # we can start reading the lines normally
            if passed_data_dict:

                # Check if the line has the corect code
                if str(line[0].strip()) == str(data_codes[floor_no]):

                    # Get the energy usage value and convert to kWh
                    time_series.x.append(float(line[1].strip()))

        # Now save this time-series in the output dictionary
        out_dict[int(floor_no)] = time_series

    return out_dict