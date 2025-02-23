import datetime as dt
import pandas as pd
import numpy as np
import numpy.typing as npt


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


def fetch(url: str) -> pd.DataFrame:
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

    return df


def parse(row: pd.DataFrame) -> pd.DataFrame:
    from pirep.defs.report import PilotReport, ReportError
    from pirep.defs.location import CodeError
    from pirep.defs.altitude import Altitude, AltitudeError
    import traceback

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


def parse_all(table: pd.DataFrame, drop_no_turbulence: bool = True) -> pd.DataFrame:
    # TODO: use the drop_no_turbulence parameter

    reports: pd.DataFrame = table.apply(parse, axis=1)
    return (
        reports.drop(columns=["Lat", "Lon"])
        .explode(column="Turbulence")
        .dropna(subset=["Turbulence"])
    )


def compute_grid(report: pd.DataFrame) -> npt.NDArray:
    from consts import GRID_RANGE, MAP_RANGE

    grid = np.full((GRID_RANGE["LAT"], GRID_RANGE["LON"], GRID_RANGE["ALT"]), np.nan)

    from pirep.defs.report import Location, Altitude, Aircraft, Turbulence

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
        )[intensity]  # WARNING be careful of LGT default

        from pirep.consts import AREA_OF_EFFECT
        from utils.convert import convert_coord as convert

        # TODO fix gridding (no AOE and fix altitudes)
        alt_min_idx = np.abs(np.array(MAP_RANGE["ALT"]["RANGE"]) - alt.min).argmin()
        if MAP_RANGE["ALT"]["RANGE"][alt_min_idx] > alt.min:
            alt_min_idx = max(alt_min_idx - 1, 0)

        alt_max_idx = np.abs(np.array(MAP_RANGE["ALT"]["RANGE"]) - alt.max).argmin()
        if MAP_RANGE["ALT"]["RANGE"][alt_max_idx] < alt.max:
            alt_max_idx = min(alt_max_idx + 1, len(MAP_RANGE["ALT"]["RANGE"]))

        grid[
            convert(loc.lat, "LAT") - AREA_OF_EFFECT : convert(loc.lat, "LAT")
            + AREA_OF_EFFECT,
            convert(loc.lon, "LON") - AREA_OF_EFFECT : convert(loc.lon, "LON")
            + AREA_OF_EFFECT,
            alt_min_idx : alt_max_idx + 1,
        ] = turbulence_index

        # print(report)

        return (grid, aircraft, intensity)
    else:
        return (grid, None, None)
