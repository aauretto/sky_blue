import re
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel

from pirep.defs.altitude import Altitude

TURBULENCE = re.compile(
    r"("
    r"\s*(?P<duration>INTMT|OCNL|CONS)?"
    r"\s*(?:NEG|LGT|MOD|SEV|EXTRM)?\s*-?"
    r"\s*(?P<intensity>SMOOTH|NEG|LGT|MOD|SEV|EXTRM)"
    r"\s*(?P<type>CAT|CHOP)?"
    r"\s*(?P<altitude>[A-Z0-9\s-]*)?"
    r"/?)+"
)


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
        CLOUD = ""

    duration: Optional[Duration]
    intensity: Intensity = Intensity.NEG
    type: Optional[Type]
    altitude: Altitude = Altitude()

    @staticmethod
    def parse(src: str, fallback: Altitude):
        results = []

        for m in re.finditer(TURBULENCE, src):
            m = m.groupdict()

            altitude = Altitude.parse(m["altitude"])

            results.append(
                Turbulence(
                    duration=(
                        None
                        if m["duration"] is None
                        else Turbulence.Duration(m["duration"])
                    ),
                    intensity=Turbulence.Intensity(m["intensity"]) if m["intensity"] != "SMOOTH" else Turbulence.Intensity.NEG,
                    type=None if m["type"] is None else Turbulence.Type(m["type"]),
                    altitude=altitude if altitude.err != Altitude.Error.UNKN else fallback,
                )
            )

        return results
