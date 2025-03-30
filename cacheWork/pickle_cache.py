# 2017-02-25 00:00:00,"{'Timestamp': Timestamp('2017-02-25 00:00:00'), 'Report': 'DBQ UA /OV DBQ180030/TM 0000/FL320/TP B737/TB LGT CHOP', 'Priority': <Priority.RTN: 'UA'>, 'Location': Location(lat=41.90183761765585, lon=-90.70909881591665), 'Altitude': Altitude(err=None, min=27000, max=37000), 'Aircraft': <Aircraft.MED: 'M'>, 'Turbulence': Turbulence(duration=None, intensity=<Intensity.LGT: 'LGT'>, type=<Type.CHOP: 'CHOP'>, altitude=Altitude(err=None, min=27000, max=37000))}"

import pandas as pd
import csv
import ast
from typing import List
from datetime import datetime
import re

import pirep as pr
from pirep.defs.report import PilotReport as pilotrep
from pirep.defs.location import Location
from pirep.defs.altitude import Altitude
from pirep.defs.aircraft import Aircraft
from pirep.defs.turbulence import Turbulence

# timestamp => datetime
# Report => str (raw report that was made)
# priority => pr.Priority ENUM
# location => pr.Location
#             | > lat / lon => float / float
# altitude => pr.Altitude
#             | > err => Alt.err
#             | > Min / ma => int / int
# Aircraft => pr.Aircraft (ENUM)
# turbulence => pr.turbulence
#               | > Duration  => turbulence.duration (ENUM)
#               | > Intensity => turbulence.intensity (ENUM)
#               | > Type      => turbulence.type (ENUM)
#               | > Altitude  => pr.Altitude (See above)

# def parse_single_pirep(pirep : str):
#     df = pd.read_csv("/skyblue/PIREPcacheFull.csv")
    
#     s = df['Data'].iloc[0]
#     del df

#     report_dict = ast.literal_eval(s)
    
#     print(type(report_dict))
#     print(report_dict)
def parse_single_pirep(report_str: str):
    # Manually extract the fields using regular expressions (example)
    timestamp_pattern = r"Timestamp\((.*?)\)"
    priority_pattern = r"<Priority\.(.*?)>"
    location_pattern = r"Location\(lat=(.*?), lon=(.*?)\)"
    altitude_pattern = r"Altitude\(err=(.*?), min=(.*?), max=(.*?)\)"
    aircraft_pattern = r"<Aircraft\.(.*?)>"
    turbulence_pattern = r"Turbulence\(duration=(.*?), intensity=(.*?), type=(.*?), altitude=(.*?)\)"
    report_pattern = r"'Report': '(.*?)'"

    # Find all fields using regular expressions
    timestamp_match = re.search(timestamp_pattern, report_str)
    priority_match = re.search(priority_pattern, report_str)
    location_match = re.search(location_pattern, report_str)
    altitude_match = re.search(altitude_pattern, report_str)
    aircraft_match = re.search(aircraft_pattern, report_str)
    turbulence_match = re.search(turbulence_pattern, report_str)
    report_match = re.search(report_pattern, report_str)

    # Extract the matched fields
    timestamp = timestamp_match.groups() if timestamp_match else None
    priority_str = priority_match.groups() if priority_match else None
    location_str = location_match.groups() if location_match else None
    altitude_str = altitude_match.groups() if altitude_match else None
    aircraft_str = aircraft_match.groups() if aircraft_match else None
    turbulence_str = turbulence_match.groups() if turbulence_match else None
    print(f"Timestamp = {timestamp}")
    timestamp_data = datetime.strptime(timestamp[0].strip("\'"), "%Y-%m-%d %H:%M:%S")
    print(f"TS Data = {timestamp_data}")
    print(f"Pri = {priority_str}")
    prio = pilotrep.Priority(priority_str[0].split()[1].strip("\'"))
    print(f"Priority is {prio}")
    print(f"loc lat= {location_str}")
    lat = float(location_str[0])
    lon = float(location_str[1])
    loc = Location(lat=lat, lon=lon)
    print(f"Location loc = {loc}")
    print(f"alt = {altitude_str}")  
    alt_err = Altitude.Error(altitude_str[0]) if altitude_str[0] != 'None' else None
    alt = Altitude(err=alt_err, min=int(altitude_str[1]), max=int(altitude_str[2]))
    print(f"Altidute is {alt}")
    print(f"aircarft = {aircraft_str}")
    arcrft = Aircraft(aircraft_str[0].split()[1].strip("\'"))
    print(f"Aircraft is {arcrft}")
    print(f"turb = {turbulence_str}")
    duration = Turbulence.Duration(turbulence_str[0].split()[1].split("\'")[1]) if turbulence_str[0] != 'None' else None
    intensity = Turbulence.Intensity(turbulence_str[1].split()[1].split("\'")[1])
    typ = Turbulence.Type(turbulence_str[2].split()[1].split("\'")[1]) if turbulence_str[2] != 'None' else None
    turb_alt = turbulence_str[3].split("(")
    turb_alt = turb_alt[1].split(", ")
    turb_alt_err = turb_alt[0].split("=")[1]
    turb_alt_err = Altitude.Error(turb_alt_err) if turb_alt_err != 'None' else None
    turb_alt_min = int(turb_alt[1].split('=')[1].strip("\'"))
    turb_alt_max = int(turb_alt[2].split('=')[1].strip("\'"))
    turb_alt = Altitude(err=turb_alt_err, min=turb_alt_min, max=turb_alt_max)

    turb = Turbulence(duration=duration, intensity=intensity, type=typ, altitude=turb_alt)
    return {
        "Timestamp"  : timestamp_data,
        "Report"     : report_match.group(1),
        "Priority"   : prio,
        "Location"   : loc,
        "Altitude"   : alt,
        "Aircraft"   : arcrft,
        "Turbulence" : turb,
    }




if __name__=='__main__':
    df = pd.read_csv("/skyblue/PIREPcacheFull.csv")
    s = df['Data'].iloc[0]
    df['Data'].apply(parse_single_pirep)

    print(df['Data'].iloc[0])