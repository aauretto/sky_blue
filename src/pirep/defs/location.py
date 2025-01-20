from pydantic import BaseModel
from scipy import constants as u
from math import radians, degrees, sin, cos, asin, atan2
import re

LOC_LATLON = re.compile(
    r"\s*(?P<lat>[0-9]{2,4})(?P<latsign>[NS])"
    r"\s*(?P<lon>[0-9]{2,5})(?P<lonsign>[EW])"
)

LOC_TWOLOC = re.compile(r"(?P<loc1>[A-Z0-9]{3,4})\s?-\s?(?P<loc2>[A-Z0-9]{3,4})")

LOC_LOCDIR = re.compile(
    r".*?(?P<loc>[A-Z0-9]{3,4})\s?(?P<dir>[0-9]{3})(?P<dist>[0-9]{3})"
)

LOC_OFFSET = re.compile(
    r"(?P<dist>[0-9]{1,3})\s?"
    r"(?P<dir>NORTH|EAST|SOUTH|WEST|N|NNE|NE|ENE|E|ESE|"
    r"SE|SSE|S|SSW|SW|WSW|W|WNW|NW|NNW)\s+(OF )?(?P<loc>[A-Z0-9]{3,4})"
)

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


class Location(BaseModel):
    lat: float
    lon: float

    @classmethod
    def parse(cls, src: str, base: str):
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

        elif re.match(LOC_LOCDIR, src) or re.match(LOC_OFFSET, src):
            m = re.match(LOC_LOCDIR, src)
            m = re.match(LOC_OFFSET, src) if m is None else m

            loc = Location.convert_code(m["loc"])
            dir = DIRECTIONS.get(m["dir"], int(m["dir"]))
            dist = int(m["dist"])

            return loc.offset(dir, dist)

        elif len(src) == 3:
            return Location.convert_code(src)

        return Location(lat=0, lon=0)

    @classmethod
    def convert_code(cls, code: str):
        import pandas as pd

        codes = pd.read_csv("pirep/utils/waypoint_codes.csv")

        match len(code):
            case 3:
                waypoint = codes[codes["iata"] == code].iloc[0]

            case 4:
                waypoint = codes[codes["icao"] == code].iloc[0]

            case _:
                raise ValueError("Invalid code")

        return Location(lat=waypoint["lat"], lon=waypoint["lon"])

    def offset(self, dir: int, dist: int):
        # Convert units
        lat1 = radians(self.lat)
        lon1 = radians(self.lon)

        dir = radians(dir)
        dist_ang = dist * u.nautical_mile / 6_371_000

        # Compute new location
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
