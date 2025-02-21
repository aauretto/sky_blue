#!/usr/bin/env python3

import datetime as dt
import pandas as pd
import pirep as pr
import satellite as st
from goes2go import GOES
from pirep.defs.spreading import vertical_spread
import numpy as np
from matplotlib import pyplot as plt


# from utils.compute_grid import compute_grid

if __name__ == "__main__":
    # Retrieve PIREPs
    reports: pd.DataFrame = pr.parse_all(
        pr.fetch(
            pr.url(
                date_s=dt.datetime(2024, 11, 6, 23, 54, 0, tzinfo=dt.timezone.utc),
                date_e=dt.datetime(2024, 11, 7, 0, 0, 0, tzinfo=dt.timezone.utc),
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

    grid, aircraft, intensity = grids["Grid"][1]
    vertical_spread(grid, intensity)
    fig = plt.figure()
    vals = np.argwhere(~np.isnan(grid))

    vals = vals[vals[:, 2].argsort()]
    points = [grid[*val] for val in vals]
    plt.plot(vals[:, 2], points)

    plt.savefig("./vertical_spread.png", dpi=300, bbox_inches="tight")
    print(reports[1])

    # # Initialize satellites
    # sat_east = GOES(satellite=16, product="ABI", domain="C")
    # bands = [8, 9, 10, 13, 14, 15]

    # # Fetch satellite data and project onto grid
    # data = st.fetch_range(
    #     dt.datetime(2025, 1, 19, 23, 59),
    #     dt.datetime(2025, 1, 20, 00, 14),
    #     sat_east,
    # )
    # band_data = st.fetch_bands(data, bands)
    # lats, lons = st.calculate_coordinates(data)
    # projected_data = st.smooth(st.project(lats, lons, band_data.data))
