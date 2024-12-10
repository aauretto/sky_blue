import datetime as dt
import pandas as pd
import numpy as np
import numpy.typing as npt


def url(date_s: dt.datetime, date_e: dt.datetime) -> str:
    from urllib import parse
    from pirep.consts import SRC

    # URL parameters
    params = {
        # Convert dates to Zulu time
        "sts": date_s.isoformat().replace("+00:00", "Z"),
        "ets": date_e.isoformat().replace("+00:00", "Z"),
        "artcc": "_ALL",
        "fmt": "csv",
    }

    # Build URL
    url = SRC + parse.urlencode(params)
    return url


def fetch(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)

    # Fix dataframe columns
    df = df[["VALID", "REPORT", "LON", "PRODUCT_ID"]]
    df = df.rename(
        columns={
            "VALID": "Timestamp",
            "REPORT": "Report",
            "LON": "Lat",
            "PRODUCT_ID": "Lon",
        }
    )  # Account for seemingly wrong columns in CSV
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format=f"%Y%m%d%H%M")

    return df


def parse(row: pd.DataFrame) -> pd.DataFrame:
    from pirep.defs.report import PilotReport
    from pirep.defs.location import Location

    try:
        report = PilotReport.parse(row["Report"], timestamp=row["Timestamp"])
        row["Priority"] = report.priority
        row["Location"] = Location(lat=row["Lat"], lon=row["Lon"])
        row["Altitude"] = report.altitude
        row["Aircraft"] = report.aircraft
        row["Turbulence"] = report.turbulence
        return row
    except:
        return row


def parse_all(table: pd.DataFrame) -> pd.DataFrame:
    reports: pd.DataFrame = table.apply(parse, axis=1)
    return reports.drop(columns=["Lat", "Lon"]).explode(column="Turbulence")


def compute_grid(pirep: pd.DataFrame) -> npt.NDArray:
    from consts import GRID_RANGE

    grid = np.zeros((GRID_RANGE["LAT"], GRID_RANGE["LON"], GRID_RANGE["ALT"]))

    from pirep.defs.report import Location, Altitude, Aircraft, Turbulence
    from pirep.consts import TURBULENCE_INDEXES

    loc: Location = pirep["Location"]
    alt: Altitude = pirep["Altitude"]
    aircraft: Aircraft = pirep["Aircraft"]
    turbulence: Turbulence = pirep["Turbulence"]

    if turbulence is Turbulence:
        if turbulence.intensity != Turbulence.Intensity.EXT:
            turbulence_index = TURBULENCE_INDEXES[aircraft][turbulence.intensity]
        else:
            turbulence_index = TURBULENCE_INDEXES[aircraft][Turbulence.Intensity.SEV]

        from pirep.consts import AREA_OF_EFFECT
        from utils.convert import convert_coord as convert

        grid[
            convert(loc.lat, "LAT")
            - AREA_OF_EFFECT : convert(loc.lat, "LAT")
            + AREA_OF_EFFECT,
            convert(loc.lon, "LON")
            - AREA_OF_EFFECT : convert(loc.lon, "LON")
            + AREA_OF_EFFECT,
            (alt.min // 500) : (alt.max // 500),
        ] = turbulence_index

    return grid
