from pydantic import BaseModel, NonNegativeInt
from typing import Optional
from enum import StrEnum
import re

ALT_MINABV = re.compile(r"ABV\s*(?P<alt1>[0-9]{3})")  # Go up 2500ft
ALT_MAXBLO = re.compile(r"BLO\s*(?P<alt2>[0-9]{3})")  # Go down 2500ft
ALT_SINGLE = re.compile(r"(?P<alt>[0-9]{3}|DURC|DURD|UNKN)")
ALT_RANGED = re.compile(r"(?P<alt1>[0-9]{3})\s?-\s?(?P<alt2>[0-9]{3})")


class AltitudeError(Exception):
    def __init__(self):
        super().__init__("Unknown Altitude")


class Altitude(BaseModel):
    class Error(StrEnum):
        UNKN = "UNKN"
        DURC = "DURC"
        DURD = "DURD"

    err: Optional[Error] = Error.UNKN
    min: Optional[NonNegativeInt] = None
    max: Optional[NonNegativeInt] = None

    @staticmethod
    def parse(src: str):
        err = Altitude.Error.UNKN
        alt_min = None
        alt_max = None

        for pat in [ALT_MINABV, ALT_MAXBLO, ALT_RANGED]:
            if pat.match(src) is None:
                continue

            m = pat.match(src).groupdict()
            err = None

            # Convert flight levels to altitudes
            if "alt1" in m.keys():
                alt_min = int(m["alt1"]) * 100
            if "alt2" in m.keys():
                alt_max = int(m["alt2"]) * 100

        # TODO do something with abv/blo and durc/durd
        if alt_min is None and alt_max is None and re.match(ALT_SINGLE, src):
            m = ALT_SINGLE.match(src).groupdict()

            match src:
                case "UNKN" | "DURC" | "DURD":
                    # TODO: Fix DURC/DURD cases
                    err = Altitude.Error(src)
                    raise AltitudeError()

                case alt:
                    err = None
                    alt_min = max(0, int(alt) * 100 - 5000)
                    alt_max = min(int(alt) * 100 + 5000, 45_000)

        # Add 5000 ft to the altitude window
        if alt_min is None and err is None:
            alt_min = max(0, alt_max - 5000)
        if alt_max is None and err is None:
            alt_max = min(alt_min + 5000, 45_000)

        return Altitude(err=err, min=alt_min, max=alt_max)
