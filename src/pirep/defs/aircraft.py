from enum import StrEnum
from ..sources import SRC_AIRCRAFT
import pandas as pd

CODES = (
    pd.read_csv(SRC_AIRCRAFT)[["type", "size"]]
    .drop_duplicates()
    .set_index("type")
    .to_dict("index")
)


class Aircraft(StrEnum):
    UNKN = "UNKN"
    LGT = "L"
    MED = "M"
    HVY = "H"

    @staticmethod
    def parse(src: str):
        if src not in CODES:
            return Aircraft.UNKN

        match CODES[src]["size"]:
            case "J":
                return Aircraft.HVY
            case "L/M":
                return Aircraft.MED
            case size:
                return Aircraft(size)
