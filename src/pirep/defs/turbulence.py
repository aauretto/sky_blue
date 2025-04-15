import re
from enum import StrEnum
from typing import Optional

from pydantic import BaseModel

from pirep.defs.altitude import Altitude

# Turbulence flag format
TURBULENCE = re.compile(
    r"("
    r"\s*(?P<duration>INTMT|OCNL|CONS)?"    # Duration
    r"\s*(?:NEG|LGT|MOD|SEV|EXTRM)?\s*-?"   # Intensity (low end)
    r"\s*(?P<intensity>SMOOTH|NEG|LGT|MOD|SEV|EXTRM)" # Intensity (high end)
    r"\s*(?P<type>CAT|CHOP)?"   # Turbulence type
    r"\s*(?P<altitude>[A-Z0-9\s-]*)?" # Altitude
    r"/?)+"
)


class Turbulence(BaseModel):
    """A class that encodes details of turbulence reported in Pilot Reports."""

    class Duration(StrEnum):
        """A class that enumerates turbulence duration classes."""
        INT = "INTMT"
        OCL = "OCNL"
        CON = "CONS"

    class Intensity(StrEnum):
        """A class that enumerated turbulence intensity classes."""
        NEG = "NEG"
        LGT = "LGT"
        MOD = "MOD"
        SEV = "SEV"
        EXT = "EXTRM"

    class Type(StrEnum):
        """A class that enumerated turbulence types."""
        CHOP = "CHOP"
        CLEAR = "CAT"
        CLOUD = ""

    duration: Optional[Duration]
    intensity: Intensity = Intensity.NEG
    type: Optional[Type]
    altitude: Altitude = Altitude()

    @staticmethod
    def parse(src: str, fallback: Altitude):
        """Parse the turbulence string provided in a Pilot Report.

        ### Parameters
        src (str) : Turbulence details provided with the /TB flag
        fallback (Altitude) : The default altitude provided in the Pilot Report

        ### Returns
        Turbulence object(s) with the relevant fields populated

        ### Notes
        Returns a list if multiple turbulence flags are passed. The results may need to be split using an explode() method.
        """

        results = []

        # Loop over every turbulence flag encountered
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
