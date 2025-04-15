import re
from math import asin, atan2, cos, degrees, radians, sin

import pandas as pd
from pydantic import BaseModel
from scipy import constants as u

from ..sources import SRC_AIRPORTS, SRC_NAVAIDS

# Format [1234N56789E] (12.34N, 56.789E) or similar
LOC_LATLON = re.compile(
    r"\s*(?P<lat>[0-9]{2,4})(?P<latsign>[NS])"
    r"\s*(?P<lon>[0-9]{2,5})(?P<lonsign>[EW])"
)

# Format [ABC-DEF]
LOC_TWOLOC = re.compile(
    r"(?P<loc1>[A-Z0-9]{3,4})\s?-\s?"
    r"(?P<loc2>[A-Z0-9]{3,4})"
)

# Format [ABC123456] (456nm in heading 123 from ABC)
LOC_LOCDIR = re.compile(
    r".*?(?P<loc>[A-Z0-9]{3,4})\s?"
    r"(?P<dir>[0-9]{3})"
    r"(?P<dist>[0-9]{3})"
)

# Format [123NNE OF ABC] (123nm in heading NNE from ABC)
LOC_OFFSET = re.compile(
    r"(?P<dist>[0-9]{1,3})\s?"
    r"(?P<dir>NORTH|EAST|SOUTH|WEST|N|NNE|NE|ENE|E|ESE|SE|SSE|S|SSW|SW|WSW|W|WNW|NW|NNW)\s+(OF )?"
    r"(?P<loc>[A-Z0-9]{3,4})"
)

# Direction to heading conversion
DIRECTIONS = {
    "NORTH": 0,
    "EAST": 90,
    "SOUTH": 180,
    "WEST": 270,
    "N": 0,
    "NNE": 22.5,
    "NE": 45,
    "ENE": 67.5,
    "E": 90,
    "ESE": 112.5,
    "SE": 135,
    "SSE": 157.5,
    "S": 180,
    "SSW": 202.5,
    "SW": 225,
    "WSW": 247.5,
    "W": 270,
    "WNW": 292.5,
    "NW": 315,
    "NNW": 337.5,
}

AIRPORT_CODES = pd.read_csv(SRC_AIRPORTS)
NAVAID_CODES = pd.read_csv(SRC_NAVAIDS)


class CodeError(Exception):
    def __init__(self, code):
        super().__init__(f"Invalid Location Code: {code}")


class Location(BaseModel):
    """A class that encodes locations provided in Pilot Reports."""

    lat: float
    lon: float

    @staticmethod
    def parse(src: str, base: str):
        """Return the location of a Pilot Report.

        ### Parameters
        src (str) : Location string from a Pilot Report
        base (str) : Reporting station code provided at beginning of a Pilot Report

        ### Returns
        Location object consisting of latitude and longitude coordinates.
        """

        # Check for Lat/Lon location format
        if re.match(LOC_LATLON, src):
            m = re.match(LOC_LATLON, src).groupdict()

            lat = m["lat"]
            lon = m["lon"]

            if len(lat) == 2 and len(lon) <= 3:
                # Degrees
                lat = int(lat)
                lon = int(lon)
            else:
                # Degrees + minutes
                lat_min = int(int(lat[-2:]) / u.minute)
                lat = float(f"{lat[:-2]}.{lat_min:.0f}")
                lon_min = int(int(lon[-2:]) / u.minute)
                lon = float(f"{lon[:-2]}.{lon_min:.0f}")

            if m["latsign"] == "W":
                lat *= -1
            if m["lonsign"] == "S":
                lon *= -1

            return Location(lat=lat, lon=lon)

        # Check for offset reference format
        elif re.match(LOC_LOCDIR, src) or re.match(LOC_OFFSET, src):
            m = re.match(LOC_LOCDIR, src)
            m = re.match(LOC_OFFSET, src) if m is None else m

            loc = Location.convert_code(m["loc"])
            dir = DIRECTIONS[m["dir"]] if m["dir"] in DIRECTIONS else int(m["dir"])
            dist = int(m["dist"])

            return loc.offset(dir, dist)

        # Check for waypoint refefrence format
        elif len(src) == 3:
            return Location.convert_code(src)

        return Location(lat=0, lon=0)

    @staticmethod
    def convert_code(code: str):
        """Return the coordinates of a waypoint code.

        ### Parameters
        code (str) : Waypoint code (KBOS/BOS)

        ### Returns
        Location with latitude and longitude corresponding to the waypoint code
        """

        match len(code):

            # Check if IATA code
            case 3:
                results = AIRPORT_CODES[
                    ((AIRPORT_CODES["iata_code"] == code)
                    | (AIRPORT_CODES["local_code"] == code))
                    & (AIRPORT_CODES["continent"] == "NA")
                ]

            # Check if ICAO code
            case 4:
                results = AIRPORT_CODES[
                    ((AIRPORT_CODES["icao_code"] == code)
                    | (AIRPORT_CODES["gps_code"] == code))
                    & (AIRPORT_CODES["continent"] == "NA")
                ]

            case _:
                raise CodeError(code)

        # Fallback to Navaid codes if no airports found
        if len(results) == 0:
            results = NAVAID_CODES[NAVAID_CODES["ident"] == code]

        if len(results) == 0:
            raise CodeError(code)

        waypoint = results.iloc[0]

        return Location(lat=waypoint["latitude_deg"], lon=waypoint["longitude_deg"])

    def offset(self, dir: int, dist: int):
        """Calculate a coordinate offset from a reference

        ### Parameters
        dir (int): A heading between 0 and 360
        dist (int): Offset distance in nautical miles

        ### Returns
        Location with latitude and longitude corresponding to the new location
        """

        r_earth = 6_371_000  # meters

        # Convert units
        lat1 = radians(self.lat)
        lon1 = radians(self.lon)

        dir = radians(dir)
        dist_ang = dist * u.nautical_mile / r_earth

        # Compute new location with a distance and heading from a known location (waypoint)
        # Source: https://www.movable-type.co.uk/scripts/latlong.html
        lat2 = asin(sin(lat1) * cos(dist_ang) + cos(lat1) * sin(dist_ang) * cos(dir))
        lon2 = (
            lon1
            + atan2(
                sin(dir) * sin(dist_ang) * cos(lat1),
                cos(dist_ang) - sin(lat1) * sin(lat2),
            )
            + 540
        ) % 360 - 180

        return Location(lat=degrees(lat2), lon=degrees(lon2))
