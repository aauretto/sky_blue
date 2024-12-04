from pydantic import BaseModel
from datetime import datetime, UTC
from enum import StrEnum
import re

from defs.location import Location
from defs.altitude import Altitude
from defs.aircraft import Aircraft
from defs.turbulence import Turbulence

FULL = re.compile(
    r"^(?P<station>[A-Z]{3,4})?"
    r"\s*(?P<priority>UA|UUA)"
    r"\s*/OV\s?(?P<location>[A-Z0-9\s]*)"
    r"\s*/TM\s?(?P<time>[0-9]{4})"
    r"\s*/FL\s?(?P<altitude>[0-9]{3}|DURC|DURD|UNKN)"
    r"\s*/TP\s?(?P<aircraft>[A-Z0-9]{3,4})"
    r"(?P<rest>[A-Z0-9\s/-]*)$"
)

REST = re.compile(
    r"/((?P<flag>SK|WX|TA|WV|TB|IC|RM)"
    r"\s?(?P<value>[A-Z0-9\s-]*"
    r"(?:/?(?!SK|WX|TA|WV|TB|IC|RM)[A-Z0-9\s\+-]*)*))*"
)


class PilotReport(BaseModel):
    class Priority(StrEnum):
        RTN = "UA"
        URG = "UUA"

    # Required fields
    timestamp: datetime = datetime.now(UTC)
    priority: Priority = Priority.RTN
    location: Location
    altitude: Altitude = Altitude()
    aircraft: Aircraft

    # Additional fields
    turbulence: list[Turbulence] = []

    @classmethod
    def parse(cls, report: str, timestamp: datetime = datetime.now(UTC)):
        # Verify report is valid
        if not re.match(FULL, report):
            raise ValueError("Invalid PIREP provided:", report)

        # Retrieve default fields
        m = re.match(FULL, report).groupdict()

        # Parse default fields
        priority = PilotReport.Priority(m["priority"])
        location = Location.parse(m["location"], m["station"])
        altitude = Altitude.parse(m["altitude"])
        aircraft = Aircraft.parse(m["aircraft"])
        data = {}

        for field in re.finditer(REST, m["rest"]):
            field = field.groupdict()
            flag = field["flag"]
            val = field["value"]

            match flag:
                # Turbulence
                case "TB":
                    entries = Turbulence.parse(val, altitude)
                    if "turbulence" in data.keys():
                        data["turbulence"].append(*entries)
                    else:
                        data["turbulence"] = entries

                # Unknown flag
                case _:
                    # print("Unknown flag:", flag)
                    pass

        return PilotReport(
            timestamp=timestamp,
            priority=priority,
            location=location,
            altitude=altitude,
            aircraft=aircraft,
            **data,
        )
