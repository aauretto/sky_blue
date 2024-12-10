# Enumerations

from enum import StrEnum, Enum


class Priority(StrEnum):
    RTN = "UA"
    URG = "UUA"


class Aircraft(Enum):
    LGT = 0
    MED = 1
    HVY = 2


from pydantic import NonNegativeInt


class AltitudeError(StrEnum):
    UNKN = "UNKN"
    DURC = "DURC"
    DURD = "DURD"


type Altitude = NonNegativeInt | AltitudeError

# Mappings


import pandas as pd

CODES = pd.read_csv("codes.csv")
CODES = CODES[["type", "size"]].drop_duplicates().set_index("type")
CODES = CODES.to_dict("index")


DIRECTION = {
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

# RegExps

import re

FULL = re.compile(
    r"^(?P<station>[A-Z]{3,4})"
    r"\s+(?P<priority>UA|UUA)"
    r"\s*/OV\s?(?P<location>[A-Z0-9\s]*)"
    r"\s*/TM\s?(?P<time>[0-9]{4})"
    r"\s*/FL\s?(?P<altitude>[0-9]{3}|DURC|DURD|UNKN)"
    r"\s*/TP\s?(?P<aircraft>[A-Z0-9]{3,4})"
    r"(?P<rest>[A-Z0-9\s/-]*)$"
)

REST = re.compile(
    r"/((?P<flag>SK|WX|TA|WV|TB|IC|RM)"
    r"\s?(?P<value>[A-Z0-9\s-]*"
    r"(?:/?(?!SK|WX|TA|WV|TB|IC|RM)[A-Z0-9\s\+-]*)*))*"
)

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

ALT_MINABV = re.compile(r"ABV\s*(?P<alt1>[0-9]{3})")

ALT_MAXBLO = re.compile(r"BLO\s*(?P<alt2>[0-9]{3})")

ALT_SINGLE = re.compile(r"(?P<alt>[0-9]{3}|DURC|DURD|UNKN)")

ALT_RANGED = re.compile(r"(?P<alt1>[0-9]{3})\s?-\s?(?P<alt2>[0-9]{3})")

TURB = re.compile(
    r"("
    r"\s*(?P<duration>INTMT|OCNL|CONS)?"
    r"\s*(?P<intensity>NEG|LGT|MOD|SEV|EXTRM)"
    r"\s*(?P<type>CAT|CHOP)?"
    r"\s*(?P<altitude>[A-Z0-9\s-]*)?"
    r"/?)+"
)
