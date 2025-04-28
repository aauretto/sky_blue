import datetime as dt

import numpy as np
import numpy.typing as npt
import pandas as pd
from consts import MAP_RANGE
from Logger import LOGGER


def url(date_s: dt.datetime, date_e: dt.datetime) -> str:
    from urllib import parse

    from pirep.sources import SRC_PIREPS

    # URL parameters
    params = {
        # Convert dates to Zulu time
        "sts": date_s.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "ets": date_e.astimezone(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artcc": "_ALL",  # Fetch from all ATCs
        "fmt": "csv",
    }

    # Build URL
    url = SRC_PIREPS + parse.urlencode(params)
    return url


def fetch(url: str) -> list[dict]:
    df = pd.read_csv(url)

    # Fix dataframe columns
    df = df[["VALID", "REPORT", "LAT", "LON"]]
    df = df.rename(
        columns={
            "VALID": "Timestamp",
            "REPORT": "Report",
            "LAT": "Lat",
            "LON": "Lon",
        }
    )
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y%m%d%H%M")
    return df.to_dict(orient="records")


def parse(row: dict) -> dict:
    import traceback

    from pirep.defs.altitude import Altitude, AltitudeError
    from pirep.defs.location import CodeError
    from pirep.defs.report import PilotReport, ReportError
    
    try:
        report = PilotReport.parse(row["Report"], timestamp=row["Timestamp"])
        row["Priority"] = report.priority
        row["Location"] = report.location
        row["Altitude"] = report.altitude
        row["Aircraft"] = report.aircraft

        # Fallback to PIREP altitude if turbulence altitude is unknown
        for flag in report.turbulence:
            if flag.altitude.err == Altitude.Error.UNKN:
                flag.altitude = report.altitude
        row["Turbulence"] = report.turbulence

        return row
    except (ReportError, CodeError, AltitudeError) as e:
        return row

    except Exception:
        LOGGER.debug(f"Failed to parse PIREP: {row['Report']}", exc_info=True)
        return row


def parse_all(table: list[dict], drop_no_turbulence: bool = True) -> list[dict]:

    reports: list[dict] = [parse(row) for row in table]
    reports = [
        {k: row[k] for k in row.keys() if k not in ["Lat", "Lon"]} for row in table
    ]
    exploded_reports = []
    for report in reports:
        if drop_no_turbulence and "Turbulence" not in report.keys():
            continue

        if not (
            MAP_RANGE["LAT"]["MIN"] <= report["Location"].lat <= MAP_RANGE["LAT"]["MAX"]
            and MAP_RANGE["LON"]["MIN"]
            <= report["Location"].lon
            <= MAP_RANGE["LON"]["MAX"]
        ):
            continue
        if not drop_no_turbulence and len(report["Turbulence"]) == 0:
            exploded_reports.append(report)
            continue

        for turbulence in report["Turbulence"]:
            new_report = report
            new_report["Turbulence"] = turbulence
            exploded_reports.append(new_report)
    return exploded_reports


# Temporary struct that holds some turb data
class PirepGrid():
    def __init__(self, lat_idx, lon_idx, alt_min_idx, alt_max_idx, turbulence_idx):
        self.lat_idx = lat_idx
        self.lon_idx = lon_idx
        self.alt_min_idx = alt_min_idx
        self.alt_max_idx = alt_max_idx
        self.turbulence_idx = turbulence_idx

def compute_grid(report: dict) -> npt.NDArray:
    from consts import GRID_RANGE, MAP_RANGE

    from pirep.defs.report import Aircraft, Altitude, Location, Turbulence

    loc: Location = report["Location"]
    alt: Altitude = report["Altitude"]
    aircraft: Aircraft = report["Aircraft"]
    turbulence: Turbulence = report["Turbulence"]

    if type(turbulence) is Turbulence:
        from pirep.consts import TURBULENCE_INDEXES

        intensity = turbulence.intensity
        if turbulence.intensity == Turbulence.Intensity.EXT:
            intensity = Turbulence.Intensity.SEV

        # TODO WARNING be careful of LGT default
        if aircraft == Aircraft.UNKN:
            aircraft = Aircraft.LGT

        turbulence_index = TURBULENCE_INDEXES[aircraft][intensity]

        from utils.convert import convert_coord as convert

        alt_min_idx = np.abs(np.array(MAP_RANGE["ALT"]["RANGE"]) - alt.min).argmin()
        if MAP_RANGE["ALT"]["RANGE"][alt_min_idx] > alt.min:
            alt_min_idx = max(alt_min_idx - 1, 0)

        alt_max_idx = np.abs(np.array(MAP_RANGE["ALT"]["RANGE"]) - alt.max).argmin()
        if MAP_RANGE["ALT"]["RANGE"][alt_max_idx] < alt.max:
            alt_max_idx = min(alt_max_idx + 1, len(MAP_RANGE["ALT"]["RANGE"]))

        # Return a 4-tuple with the center point and turb index in it
        grid = PirepGrid(
            lat_idx = convert(loc.lat, "LAT"),
            lon_idx = convert(loc.lon, "LON"),
            alt_min_idx = alt_min_idx,
            alt_max_idx = alt_max_idx + 1,
            turbulence_idx = turbulence_index,
            )

        return (grid, aircraft, intensity)
    else:
        return (grid, None, None)
