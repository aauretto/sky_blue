from enum import StrEnum
import pandas as pd

CODES = (
    pd.read_csv("pirep/utils/aircraft_codes.csv")[["type", "size"]]
    .drop_duplicates()
    .set_index("type")
    .to_dict("index")
)


class Aircraft(StrEnum):
    UNKN = "UNKN"
    LGT = "L"
    MED = "M"
    HVY = "H"

    @classmethod
    def parse(cls, src: str):
        if src not in CODES:
            return Aircraft.UNKN

        match CODES[src]["size"]:
            case "J":
                return Aircraft.HVY
            case "L/M":
                return Aircraft.MED
            case size:
                return Aircraft(size)
