from datetime import datetime, UTC
from pydantic import BaseModel
from enum import StrEnum
from utils import *
from defs import *


def convert_altitude(src: str) -> Altitude:
    match src:
        case "UNKN" | "DURC" | "DURD":
            return AltitudeError(src)

        case alt:
            return int(alt) * 100


def convert_altitude_range(src: str) -> tuple[Altitude, Altitude]:
    for pat in [ALT_MINABV, ALT_MAXBLO, ALT_RANGED, ALT_SINGLE]:
        if pat.match(src):
            m = pat.match(src).groupdict()
            alt1 = alt2 = AltitudeError.UNKN

            if "alt" in m.keys():
                alt1 = alt2 = convert_altitude(m["alt"])
            if "alt1" in m.keys():
                alt1 = convert_altitude(m["alt1"])
            if "alt2" in m.keys():
                alt2 = convert_altitude(m["alt2"])

            return (alt1, alt2)


def read_report(report: str, timestamp: datetime) -> Report:
    # Verify report is valid
    if not FULL.match(report):
        raise ValueError("Invalid PIREP provided:", report)

    # Retrieve default fields
    m = FULL.match(report).groupdict()

    # Parse default fields
    data = {}
    data["priority"] = Priority(m["priority"])
    data["location"] = Location.decode(m["location"], m["station"]).src
    data["aircraft"] = Aircraft.decode(m["aircraft"])
    data["altitude"] = Altitude.decode(m["altitude"])

    # Retrieve remaining flags
    for s in REST.finditer(m["rest"]):
        s = s.groupdict()
        match s["flag"]:
            # Turbulence
            case "TB":
                data["turbulence"] = []

                # Retrieve all turbulence entries
                for t in TURB.finditer(s["value"]):
                    t = t.groupdict()

                    # Parse fields
                    duration, intensity, type = (
                        None if t[key] is None else obj(t[key])
                        for key, obj in [
                            ("duration", Turbulence.Duration),
                            ("intensity", Turbulence.Intensity),
                            ("type", Turbulence.Type),
                        ]
                    )

                    # TODO: Parse altitude range
                    altitude = (
                        (data["altitude"], data["altitude"])
                        if t["altitude"] == ""
                        else convert_altitude_range(t["altitude"])
                    )

                    # Add turbulence objects
                    data["turbulence"].append(
                        Turbulence(
                            duration=duration,
                            intensity=intensity,
                            type=type,
                            altitude=altitude,
                        )
                    )

            # Remarks
            case "RM":
                continue

            # Unknown flag
            case _:
                print("Unknown flag:", s["flag"])

    # Create PIREP object
    return Report(**data)


rep = read_report(
    "KCMH UA /OV APE 230010/TM 1516/FL085/TP BE20/SK BKN065/WX FV03SM HZ FU/TA 20/TB LGT BLO 200",
    datetime.now(UTC),
)
print(rep)
