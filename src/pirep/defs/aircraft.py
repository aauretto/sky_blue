from enum import StrEnum
import pandas as pd

# TODO: Comment where/why the csv is here and what it is
CODES = (
    pd.read_csv("src/pirep/utils/aircraft_codes.csv")[
        ["type", "size"]
    ]  # TODO explain when and if they would need to pull a new version of this. Link original source
    .drop_duplicates()
    .set_index("type")
    .to_dict("index")
)


class Aircraft(StrEnum):
    """
    TODO
    """

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
