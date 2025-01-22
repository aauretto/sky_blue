import pandas as pd

aircraft_types = pd.read_csv("aircraft_types.csv")
aircraft_types = aircraft_types[["Type Designator", "WTC"]].drop_duplicates()

import re

FLAGS = {
    "base": r"^(?P<base>[A-Z0-9]{3,4})?\s+(?P<priority>UUA?)",
    "Station": r"^([A-Z0-9]{3,4})",
    "Urgency": r"(UUA?)",
    "Location": r"/OV\s*(.*?)\s*/TM",
    "Altitude": r"/FL\s*(?P<alt>\d{3}|UNKN|DURC|DURD)",
    "Aircraft Type": r"/TP\s*(?P<type>[A-Z0-9]*?)",
    "Turbulence": r"/TB\s*(?P<turbulence>[A-Z0-9\s]*?)",
}

# Location
OV_LATLON = re.compile(
    r"\s?(?P<lat>[0-9]{2,4})(?P<latsign>[NS])"
    r"\s?(?P<lon>[0-9]{2,5})(?P<lonsign>[EW])"
)
OV_LOCDIR = re.compile(
    r".*?(?P<loc>[A-Z0-9]{3,4})\s?(?P<dir>[0-9]{3})(?P<dist>[0-9]{3})"
)
OV_TWOLOC = re.compile(r"(?P<loc1>[A-Z0-9]{3,4})\s?-\s?(?P<loc2>[A-Z0-9]{3,4})")
OV_OFFSET = re.compile(
    r"(?P<dist>[0-9]{1,3})\s?"
    r"(?P<dir>NORTH|EAST|SOUTH|WEST|N|NNE|NE|ENE|E|ESE|"
    r"SE|SSE|S|SSW|SW|WSW|W|WNW|NW|NNW)\s+(OF )?(?P<loc>[A-Z0-9]{3,4})"
)

# Turbulence

## Duration
TB_DUR = re.compile(r"(?P<dur>INTMT|OCNL|CONS)")

## Intensity
TB_INT = re.compile(r"(?P<int>NEG|LGT|MOD|SEV|EXTRM)")

## Type
TB_TYPE = re.compile(r"(?P<type>CAT|CHOP)")

## Altitude
TB_ALT_SINGLE = re.compile(r"(?P<alt>\d{3})")
TB_ALT_RANGED = re.compile(r"(?P<altmin>\d{3})-(?P<altmax>\d{3})")
TB_ALT_MINABV = re.compile(r"ABV\s*(?P<alt>\d{3})")
TB_ALT_MAXBLO = re.compile(r"BLO\s*(?P<alt>\d{3})")

from enum import StrEnum
import datetime as dt


class Priority(StrEnum):
    ROUTINE = "UA"
    URGENT = "UUA"


class AircraftType(StrEnum):
    LGT = "L"
    MED = "M"
    HVY = "H"


class TurbulenceDuration(StrEnum):
    INTERMITTENT = "INTMT"
    OCCASIONAL = "OCNL"
    CONTINUOUS = "CONS"


class TurbulenceIntensity(StrEnum):
    NIL = "NEG"
    LGT = "LGT"
    MOD = "MOD"
    SEV = "SEV"
    EXT = "EXTRM"


class TurbulenceType(StrEnum):
    CAT = "CAT"
    CHOP = "CHOP"


class PilotReport:
    def __init__(
        self, report: str, timestamp: dt.datetime = dt.datetime.now(dt.timezone.utc)
    ) -> None:
        self.timestamp: dt.datetime = timestamp
        self.priority: Priority = Priority.ROUTINE
        self.altitude: int = None
        self.aircraft_type: AircraftType = None
        self.turbulence = {
            "duration": None,
            "intensity": None,
            "type": None,
            "altitude": (None, None),
        }

        # Parse base station and priority flags
        if re.match(FLAGS["base"], report):
            m = re.match(FLAGS["base"], report).groupdict()
            # TODO: Base station
            self.priority = (
                Priority.URGENT if m["priority"] == "UAA" else Priority.ROUTINE
            )

        # Parse location
        if re.match(FLAGS["Location"], report):
            m = re.match(FLAGS["Location"], report)
            pass

        # Parse altitude
        if re.match(FLAGS["Altitude"], report):
            m = re.match(FLAGS["Altitude"], report).groupdict()
            match m["alt"]:
                case "UNKN" | "DURC" | "DURD" | None:
                    self.altitude = -1

                case _:
                    self.altitude = int(m["alt"]) * 100

        # Parse aircraft type
        if re.match(FLAGS["Aircraft Type"], report):
            m = re.match(FLAGS["Aircraft Type"], report).groupdict()
            size_class = aircraft_types[aircraft_types["Type Designator"] == m["type"]][
                "WTC"
            ].iloc[0]

            match size_class:
                case "L/M":
                    self.aircraft_type = AircraftType.LGT

                case "J":
                    self.aircraft_type = AircraftType.HVY

                case _:
                    self.aircraft_type = AircraftType(size_class)

        # Parse turbulence
        if re.match(FLAGS["Turbulence"], report):
            turbulence = re.match(FLAGS["Turbulence"], report).groupdict()["turbulence"]

            # Parse Duration
            with re.match(TB_DUR, turbulence).groupdict()["dur"] as duration:
                self.turbulence["duration"] = TurbulenceDuration(duration)

            # Parse Intensity
            with re.match(TB_INT, turbulence).groupdict()["int"] as intensity:
                self.turbulence["intensity"] = TurbulenceIntensity(intensity)

            # Parse Type
            with re.match(TB_TYPE, turbulence).groupdict()["type"] as type:
                self.turbulence["type"] = TurbulenceType(type)

            # Parse Altitude

            ## MIN
            with re.match(TB_ALT_MINABV, turbulence).groupdict() as m:
                self.turbulence["altitude"] = (int(m["alt"]) * 100, None)

            ## MAX
            with re.match(TB_ALT_MAXBLO, turbulence).groupdict() as m:
                self.turbulence["altitude"] = (None, int(m["alt"]) * 100)

            ## SINGLE
            with re.match(TB_ALT_SINGLE, turbulence).groupdict() as m:
                self.turbulence["altitude"] = (int(m["alt"]) * 100, int(m["alt"]) * 100)

            ## RANGED
            with re.match(TB_ALT_RANGED, turbulence).groupdict() as m:
                self.turbulence["altitude"] = (
                    int(m["altmin"]) * 100,
                    int(m["altmax"]) * 100,
                )


rep = PilotReport(
    "KCMH UA /OV APE 230010/TM 1516/FL085/TP BE20/SK BKN065/WX FV03SM HZ FU/TA 20/TB LGT"
)

print(rep.priority, rep.aircraft_type, rep.altitude, rep.turbulence)
