# NOTE: this has been designated as the entry point for Docker 

import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import pirep as pr
import satellite as st
from goes2go import GOES

# Retrieve PIREPs
reports: pd.DataFrame = pr.parse_all(
    pr.fetch(
        pr.url(
            date_s=dt.datetime(2024, 11, 6, 23, 54, 0, tzinfo=dt.timezone.utc),
            date_e=dt.datetime(2024, 11, 7, 0, 4, 0, tzinfo=dt.timezone.utc),
        )
    )
)

# Convert reports to grids
grids = pd.DataFrame(
    {
        "Timestamp": reports["Timestamp"],
        "Grid": reports.apply(pr.compute_grid, axis=1),
    }
)

sat_east = GOES(satellite=16, product="ABI", domain="C")
band = 14

data = st.fetch(dt.datetime(2024, 11, 6, 23, 59), sat_east)
band_data = st.fetch_band(data, band)
lats, lons = st.calculate_coordinates(data)
projected_data = st.project(lats, lons, band_data)

fig = plt.figure(figsize=(15, 12))
ax_east = fig.add_subplot(1, 1, 1)
ax_east.pcolormesh(projected_data)
plt.savefig("src/plots/main_figure.png")
