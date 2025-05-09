import datetime as dt

import numpy as np
import numpy.typing as npt
import pandas as pd
from consts import MAP_RANGE
from Logger import LOGGER
from pirep.spreading import add_pirep
from pirep.PirepGrid import PirepGrid

def url(date_s: dt.datetime, date_e: dt.datetime) -> str:
    """
    Creates a url to pull pireps from

    Parameters
    ----------
    date_s: dt.datetime
        The UTC datetime to start pulling pireps from
    date_e: dt.datetime
        The UTC datetime to pull pireps up to
    
    Returns
    -------
    url: str
        The url to pull pireps from
    """
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
    """
    Actually fetches the pireps from the url

    Parameters
    ----------
    url: str
        The url to fetch the PIREP from
    """
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
    """
    Updates a pirep to be parsed to actual Python objects

    Parameters
    ----------
    row: dict
        A singe row from the return of fetch
    
    Returns
    -------
    dict
        The row updated to include parsed information
    """
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
    """
    Parses all the PIREPs to uasbale python objects

    Parameters
    ----------
    table: list[dict]
        The return of fetch
    drop_no_turbulence: bool
        A flag for whether or not to drop any reports without turbulence

    Returns
    -------
    list(dict)
        The original table updated to include parsed information
    """

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


def compute_grid(report: dict):
    """
    Takes a single PIREP and creates the necessary information to 
    put it on the grid

    Parameters
    ----------
    report: dict
        The pirep report to actaully put on the grid
    
    Returns
    -------
    tuple(PirepGrid, Aircraft, Turbulence.Intensity)
        The information needed to gridify the pirep
    """
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

# Function that takes reports and spreads all PIREPS and smooshes everything together iteratively
def concatenate_all_pireps(reports: list[dict], background_risk: float):
    """
    Generates a grid representing a map of CONUS denoting risk of turbulence
    in a given area.

    Parameters
    ----------
    reports: list[dict]
        List of PIREPs reported over CONUS where turbulence was reported.
    background_risk: float
        The base risk of encountering turbulence

    Returns
    -------
    finalGrid: npt.NDarray
        A grid that represents turbulence risk over the entirity of CONUS
    """
    from consts import GRID_RANGE
    # Make final grid to spread all events onto
    finalGrid = np.full(
        (GRID_RANGE["LAT"], GRID_RANGE["LON"], GRID_RANGE["ALT"]), np.nan,
        dtype=np.float32
    )

    # Add all pireps to the grid one by one
    import pirep as pr
    for report in reports:
        prGridData, aircraft, intensity = compute_grid(report)
        # Add targeted pirep to grid
        try:
            add_pirep(finalGrid, prGridData, aircraft, intensity, background_risk)
        except Exception:
            LOGGER.error(f"Failed to add pirep {report}\n", exc_info=True)

    # Fill in bg risk everywhere we didnt previously fill in
    mask = np.isnan(finalGrid) | (finalGrid == -np.inf)
    finalGrid[mask] = np.random.uniform(
        1e-5, 1e-7, size=mask.sum()
    )  # TODO document magic numbers

    return finalGrid