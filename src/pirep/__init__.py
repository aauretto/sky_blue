import datetime as dt

import numpy as np
import numpy.typing as npt
import pandas as pd
from consts import MAP_RANGE


def url(date_s: dt.datetime, date_e: dt.datetime) -> str:
    from urllib import parse

    from pirep.sources import SRC_PIREPS

    # URL parameters
    params = {
        # Convert dates to Zulu time
        "sts": date_s.isoformat() + "Z",
        "ets": date_e.isoformat() + "Z",
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

    # print(f"\x1b[1K\r{row['Report']}", end="")

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
    except (ReportError, CodeError, AltitudeError):
        return row

    except Exception:
        print(traceback.format_exc())
        return row


def parse_all(table: list[dict], drop_no_turbulence: bool = True) -> list[dict]:
    reports: list[dict] = [parse(row) for row in table]
    reports = [{k: row[k] for k in row.keys() if k not in ["Lat", "Lon"]} for row in table]
    exploded_reports = []
    for report in reports:
        if drop_no_turbulence and "Turbulence" not in report.keys():
            continue

        if not (MAP_RANGE["LAT"]["MIN"] <= report["Location"].lat <= MAP_RANGE['LAT']['MAX'] and MAP_RANGE['LON']['MIN'] <= report["Location"].lon <= MAP_RANGE['LON']['MAX']):
            continue
        if not drop_no_turbulence and len(report["Turbulence"]) == 0:
            exploded_reports.append(report)
            continue

        for turbulence in report["Turbulence"]:
            new_report = report
            new_report["Turbulence"] = turbulence
            exploded_reports.append(new_report)
    return exploded_reports

# performs in place dropping of reports containing lats and/or lons outside of range
# def fetch_parse_and_drop_irrelevant(date_s: dt.datetime, date_e: dt.datetime) -> pd.DataFrame:
#     reports = parse_all(fetch(url(date_s, date_e)))
#     return reports[
#             reports["Location"].apply(lambda loc: MAP_RANGE["LAT"]["MIN"] <= loc.lat <= MAP_RANGE['LAT']['MAX'] and
#                                                 MAP_RANGE['LON']['MIN'] <= loc.lon <= MAP_RANGE['LON']['MAX'])
            # ]

def compute_grid(report: dict) -> npt.NDArray:
    from consts import GRID_RANGE, MAP_RANGE

    grid = np.full((GRID_RANGE["LAT"], GRID_RANGE["LON"], GRID_RANGE["ALT"]), np.nan)

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

        turbulence_index = TURBULENCE_INDEXES.get(
            aircraft, TURBULENCE_INDEXES[Aircraft.LGT]
        )[intensity]  # TODO WARNING be careful of LGT default

        from utils.convert import convert_coord as convert

        alt_min_idx = np.abs(np.array(MAP_RANGE["ALT"]["RANGE"]) - alt.min).argmin()
        if MAP_RANGE["ALT"]["RANGE"][alt_min_idx] > alt.min:
            alt_min_idx = max(alt_min_idx - 1, 0)

        alt_max_idx = np.abs(np.array(MAP_RANGE["ALT"]["RANGE"]) - alt.max).argmin()
        if MAP_RANGE["ALT"]["RANGE"][alt_max_idx] < alt.max:
            alt_max_idx = min(alt_max_idx + 1, len(MAP_RANGE["ALT"]["RANGE"]))

        grid[
            convert(loc.lat, "LAT") : convert(loc.lat, "LAT") + 1,
            convert(loc.lon, "LON") : convert(loc.lon, "LON") + 1,
            alt_min_idx : alt_max_idx + 1,
        ] = turbulence_index

        # TODO remove debugging
        if len(np.argwhere(~np.isnan(grid))) == 0:
            print('\n*************************************')
            print(report)
            print('Failed to input grid')
            print(f"{alt_min_idx=}")
            print(f"{alt_max_idx=}")
            print(f"{turbulence_index=}")
            print(f"{loc.lat=}")
            print(f"{convert(loc.lat, "LAT")=}")
            print(f"{loc.lon=}")
            print(f"{convert(loc.lon, "LON")=}")
            print('*************************************')

        return (grid, aircraft, intensity)
    else:
        return (grid, None, None)
