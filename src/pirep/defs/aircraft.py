from enum import StrEnum

import pandas as pd

from ..sources import SRC_AIRCRAFT

CODES = (
    pd.read_csv(SRC_AIRCRAFT)[["type", "size"]]
    .drop_duplicates()
    .set_index("type")
    .to_dict("index")
)


class Aircraft(StrEnum):
    """A class that enumerates aircraft size classes."""

    UNKN = "UNKN"
    LGT = "L"
    MED = "M"
    HVY = "H"

    @staticmethod
    def parse(src: str):
        """Return the aircraft size class.

        ### Parameters
        src (str) : Aircraft model code

        ### Returns
        UNKN | LGT | MED | HVY : Aircraft size class enumeration.

        ### Notes
        Jumbo jets are downgraded to HVY, and Light-Medium are upgraded to MED
        """

        if src not in CODES:
            return Aircraft.UNKN

        match CODES[src]["size"]:
            case "J":
                return Aircraft.HVY
            case "L/M":
                return Aircraft.MED
            case size:
                return Aircraft(size)
