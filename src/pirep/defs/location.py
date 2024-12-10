from pydantic import BaseModel
import re

LOC_LATLON = re.compile(
    r"\s*(?P<lat>[0-9]{2,4})(?P<latsign>[NS])"
    r"\s*(?P<lon>[0-9]{2,5})(?P<lonsign>[EW])"
)

LOC_TWOLOC = re.compile(r"(?P<loc1>[A-Z0-9]{3,4})\s?-\s?(?P<loc2>[A-Z0-9]{3,4})")

LOC_LOCDIR = re.compile(
    r".*?(?P<loc>[A-Z0-9]{3,4})\s?(?P<dir>[0-9]{3})(?P<dist>[0-9]{3})"
)

LOC_OFFSET = re.compile(
    r"(?P<dist>[0-9]{1,3})\s?"
    r"(?P<dir>NORTH|EAST|SOUTH|WEST|N|NNE|NE|ENE|E|ESE|"
    r"SE|SSE|S|SSW|SW|WSW|W|WNW|NW|NNW)\s+(OF )?(?P<loc>[A-Z0-9]{3,4})"
)


class Location(BaseModel):
    lat: float
    lon: float

    @classmethod
    def parse(cls, src: str, base: str):
        if re.match(LOC_LATLON, src):
            m = re.match(LOC_LATLON, src).groupdict()

            lat = m["lat"]
            lon = m["lon"]

            if len(lat) == 2 and len(lon) <= 3:
                # Degrees
                lat = int(lat)
                lon = int(lon)
            else:
                # Degrees + minutes
                lat_min = int(float(lat[-2:]) / 60.0 * 10000.0)
                lat = float(f"{lat[:-2]}.{lat_min:.0f}")
                lon_min = int(float(lon[-2:]) / 60.0 * 10000.0)
                lon = float(f"{lon[:-2]}.{lon_min:.0f}")

            if m["latsign"] == "S":
                lat *= -1
            if m["lonsign"] == "S":
                lon *= -1

            return Location(lat=lat, lon=lon)
        return Location(lat=0, lon=0)
