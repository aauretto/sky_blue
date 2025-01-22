import re

# Location RegExps

OV_LATLON = re.compile(
    (
        r"\s?(?P<lat>[0-9]{2,4})(?P<latsign>[NS])"
        r"\s?(?P<lon>[0-9]{2,5})(?P<lonsign>[EW])"
    )
)
OV_LOCDIR = re.compile(
    r".*?(?P<loc>[A-Z0-9]{3,4})\s?(?P<dir>[0-9]{3})(?P<dist>[0-9]{3})"
)
OV_TWOLOC = re.compile(r"(?P<loc1>[A-Z0-9]{3,4})\s?-\s?(?P<loc2>[A-Z0-9]{3,4})")
OV_OFFSET = re.compile(
    (
        r"(?P<dist>[0-9]{1,3})\s?"
        r"(?P<dir>NORTH|EAST|SOUTH|WEST|N|NNE|NE|ENE|E|ESE|"
        r"SE|SSE|S|SSW|SW|WSW|W|WNW|NW|NNW)\s+(OF )?(?P<loc>[A-Z0-9]{3,4})"
    )
)

# Direction-to-Heading Mappings

DRCT2DIR = {
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
    "NW": 305,
    "NNW": 327.5,
    "NORTH": 0,
    "EAST": 90,
    "SOUTH": 180,
    "WEST": 270,
}

# Priority Enumeration

from enum import Enum


class Priority(str, Enum):
    """Types of reports."""

    def __str__(self):
        """When we want the str repr."""
        return str(self.value)

    UA = "UA"
    UUA = "UUA"


# PIREP Class

from pydantic import BaseModel
import datetime


class PilotReport(BaseModel):
    """A Pilot Report."""

    text: str = None
    priority: Priority = None

    latitude: float = None
    longitude: float = None

    aircraft_type: str = None

    time: datetime.datetime = None
    is_duplicate: bool = False


def _rectify_identifier(station, textprod):
    """Rectify the station identifer to IEM Nomenclature."""
    station = station.strip()
    if len(station) == 4 and station.startswith("K"):
        return station[1:]
    if len(station) == 3 and not textprod.source.startswith("K"):
        return textprod.source[0] + station
    return station


def _parse_lonlat(text):
    """Convert string into lon, lat values"""
    # 2500N07000W -or- 25N070W -or- 25N70W
    # FMH-12 says this is in degrees and minutes!
    d = re.match(OV_LATLON, text).groupdict()

    lat = d["lat"]
    lon = d["lon"]

    if len(lat) == 2 and len(lon) <= 3:
        # We have integer values :/
        lat = int(lat)
        lon = int(lon)
    else:
        # We have Degrees and minutes
        _d = int(float(lat[-2:]) / 60.0 * 10000.0)
        lat = float(f"{lat[:-2]}.{_d:.0f}")
        _d = int(float(lon[-2:]) / 60.0 * 10000.0)
        lon = float(f"{lon[:-2]}.{_d:.0f}")
    if d["latsign"] == "S":
        lat *= -1
    if d["lonsign"] == "W":
        lon *= -1

    return lon, lat


class Pirep:
    def __init__(self, text) -> None:
        """constructor"""

        self.text = text
        self.reports = []
        self.parse_reports()

    def parse_reports(self) -> None:
        """Parse the raw text into the self.reports list"""

        txt = self.text if self.text[:2] != "\001\n" else self.text[2:]
        lines = txt.split("\n")

        # There may be an AWIPSID in line 3 or silly aviation control char
        pos = 3 if len(lines[2]) < 10 or lines[2].startswith("\x1e") else 2
        meat = "".join(lines[pos:])
        for report in meat.split("="):
            if report.strip() == "":
                continue
            res = self.parse_report(" ".join(report.strip().split()))
            if res is not None and res.valid is not None:
                self.reports.append(res)

    def parse_report(self, report):
        """Parse a single report into a PIREP object"""
        pirep = PilotReport()
        pirep.text = report

        for i, token in enumerate(report.split("/")):
            token = token.strip()

            # Priority
            if i == 0:
                if len(token) > 10:
                    print("Aborting as not-PIREP? |%s|", report)
                    return None
                if token.find(" UUA") > 0:
                    pirep.priority = Priority.UUA
                else:
                    pirep.priority = Priority.UA

                continue

            # Aircraft Type
            TOKEN_AIRCRAFT_TYPE = "TP "
            if token.startswith(TOKEN_AIRCRAFT_TYPE):
                pirep.aircraft_type = token[len(TOKEN_AIRCRAFT_TYPE) :]

            # Location
            TOKEN_LOCATION = "OV "
            if token.startswith(TOKEN_LOCATION):
                dist = 0
                bearing = 0
                remainder = token[3:]

                if len(remainder) == 3:
                    loc = _rectify_identifier(remainder, self)
                elif remainder.startswith("FINAL RWY"):
                    loc = _rectify_identifier(report[:8].split()[0], self)
                elif len(remainder) == 4:
                    loc = _rectify_identifier(remainder, self)
                elif re.match(OV_OFFSET, remainder):
                    d = re.match(OV_OFFSET, remainder).groupdict()
                    loc = _rectify_identifier(d["loc"], self)
                    dist = int(d["dist"])
                    bearing = DRCT2DIR[d["dir"]]
                elif remainder.find("-") > 0 and re.match(OV_TWOLOC, remainder):
                    d = re.match(OV_TWOLOC, remainder).groupdict()
                    numbers = re.findall("[0-9]{6}", remainder)
                    if numbers:
                        bearing = int(numbers[0][:3])
                        dist = int(numbers[0][3:])
                        loc = _rectify_identifier(d["loc2"], self)
                    else:
                        # Split the distance between the two points
                        lats = []
                        lons = []
                        for loc in [d["loc1"], d["loc2"]]:
                            loc = _rectify_identifier(loc, self)
                            if loc not in self.nwsli_provider:
                                self.warnings.append(
                                    f"Unknown location: {loc} '{report}'"
                                )
                            else:
                                lats.append(self.nwsli_provider[loc]["lat"])
                                lons.append(self.nwsli_provider[loc]["lon"])
                        if len(lats) == 2:
                            _pr.latitude = sum(lats) / 2.0
                            _pr.longitude = sum(lons) / 2.0
                        continue
                elif re.match(OV_LOCDIR, remainder):
                    # KFAR330008
                    d = re.match(OV_LOCDIR, remainder).groupdict()
                    loc = _rectify_identifier(d["loc"], self)
                    bearing = int(d["dir"])
                    dist = int(d["dist"])
                elif re.match(OV_LATLON, remainder):
                    _pr.longitude, _pr.latitude = _parse_lonlat(remainder)
                    continue
