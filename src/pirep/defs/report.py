import re
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel

from pirep.defs.aircraft import Aircraft
from pirep.defs.altitude import Altitude
from pirep.defs.location import Location
from pirep.defs.turbulence import Turbulence

FULL = re.compile(
    r"^(?P<station>[A-Z]{3,4})?"
    r"\s*(?P<priority>UA|UUA)"
    r"\s*/OV\s?(?P<location>[A-Z0-9\s]*)"
    r"\s*/TM\s?(?P<time>[0-9]{4})"
    r"\s*/FL\s?(?P<altitude>[0-9]{3}|DURC|DURD|UNKN)"
    r"\s*/TP\s?(?P<aircraft>[A-Z0-9]{3,4})"
    r"(?P<rest>.*)$"
)

REST = re.compile(
    r"/((?P<flag>SK|WX|TA|WV|TB|IC|RM)"
    r"\s?(?P<value>[A-Z0-9\s-]*"
    r"(?:/?(?!SK|WX|TA|WV|TB|IC|RM)[A-Z0-9\s\+-]*)*))*"
)


# Error to be caught if we cant match format for pirep
class ReportError(Exception):
    def __init__(self, report):
        super().__init__(f"Invalid PIREP provided: {report}")


class PilotReport(BaseModel):
    """A class that encodes Pilot Reports (PIREPs)."""

    class Priority(StrEnum):
        """A class that enumerated PIREP priority classes."""
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

    @staticmethod
    def parse(report: str, timestamp: datetime = datetime.now(UTC)):
        """Parse a Pilot Report.

        ### Parameters
        report (str) : Full string corresponding to a single Pilot Report
        timestamp (datetime) : Timestamp associated with the report.

        ### Returns
        PilotReport object with the relevant fields populated

        ### Notes
        See subclasses for specific implementation of each field
        """

        # Verify report is valid
        if not re.match(FULL, report):
            raise ReportError(report)

        # Retrieve default fields
        m = re.match(FULL, report).groupdict()

        # Parse default fields
        priority = PilotReport.Priority(m["priority"])
        location = Location.parse(m["location"], m["station"])
        altitude = Altitude.parse(m["altitude"])
        aircraft = Aircraft.parse(m["aircraft"])

        # Parse optional fields
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

                # Unknown flag i.e. not a turbulence flag
                case _:
                    # Unimplemented
                    pass

        return PilotReport(
            timestamp=timestamp,
            priority=priority,
            location=location,
            altitude=altitude,
            aircraft=aircraft,
            **data,
        )
