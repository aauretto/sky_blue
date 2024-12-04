from datetime import datetime
from typing import Any
from pydantic import BaseModel
from enum import Enum, StrEnum
import pandas as pd

CODES = pd.read_csv("codes.csv")
CODES = CODES[["type", "size"]].drop_duplicates().set_index("type")
CODES = CODES.to_dict("index")


# Altitude
class Altitude(BaseModel):
    class AltitudeType(StrEnum):
        NONE = ""
        UNKN = "UNKN"
        DURC = "DURC"
        DURD = "DURD"

    type: AltitudeType = AltitudeType.UNKN
    alt_min: int = None
    alt_max: int = None

    @property
    def alt(self) -> int:
        return self.alt_min if self.alt_min == self.alt_max else None

    @setattr
    def alt(self, altitude: int) -> None:
        self.alt_min = self.alt_max = altitude

    @property
    def alt_range(self) -> tuple[int, int]:
        return (self.alt_min, self.alt_max)

    @classmethod
    def decode(src: str) -> Any:
        pass


# Priority
class Priority(StrEnum):
    RTN = "UA"
    URG = "UUA"


# Aircraft
class Aircraft(Enum):
    LGT = 0
    MED = 1
    HVY = 2

    @classmethod
    def decode(src: str) -> Any:
        match CODES[src]["size"]:
            case "J":
                return Aircraft.HVY
            case "L/M":
                return Aircraft.MED
            case size:
                return Aircraft(size)


# Location
class Location(BaseModel):
    lat: float = None
    lon: float = None

    @classmethod
    def decode(src: str, base: str = "") -> Any:
        return Location(lat=0, lon=0)


# Turbulence
class Turbulence(BaseModel):
    class Duration(StrEnum):
        INT = "INTMT"
        OCL = "OCNL"
        CON = "CONS"

    class Intensity(StrEnum):
        NEG = "NEG"
        LGT = "LGT"
        MOD = "MOD"
        SEV = "SEV"
        EXT = "EXTRM"

    class Type(StrEnum):
        CHOP = "CHOP"
        CLEAR = "CAT"
        # CLOUD = ""

    duration: Duration = None
    intensity: Intensity = Intensity.NEG
    type: Type = None
    altitude: tuple[Altitude, Altitude]


# Report
class Report(BaseModel):
    # Required flags
    timestamp: datetime = None
    priority: Priority = Priority.RTN
    location: Location = None
    altitude: Altitude = Altitude(Altitude.AltitudeType.UNKN)
    aircraft: Aircraft = None

    # Additional flags
    turbulence: list[Turbulence] = []
