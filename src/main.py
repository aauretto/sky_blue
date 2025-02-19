#!/usr/bin/env python3

import datetime as dt
import pandas as pd
import numpy as np
import pirep as pr
import satellite as st
from goes2go import GOES

if __name__ == "__main__":
    # Retrieve PIREPs
    # reports: pd.DataFrame = pr.parse_all(
    #     pr.fetch(
    #         pr.url(
    #             date_s=dt.datetime(2024, 11, 6, 23, 54, 0, tzinfo=dt.timezone.utc),
    #             date_e=dt.datetime(2024, 11, 7, 0, 24, 0, tzinfo=dt.timezone.utc),
    #         )
    #     )
    # )

    # Convert reports to grids
    # grids = pd.DataFrame(
    #     {
    #         "Timestamp": reports["Timestamp"],
    #         "Grid": reports.apply(pr.compute_grid, axis=1),
    #     }
    # )

    # Initialize satellites
    sat_east = GOES(satellite=16, product="ABI", domain="C")
    bands = [8, 9, 10, 13, 14, 15]

    # Fetch satellite data and project onto grid
    data = st.fetch_range(
        dt.datetime(2025, 1, 19, 23, 59),
        dt.datetime(2025, 1, 20, 00, 14),
        sat_east,
    )
    band_data = st.fetch_bands(data, bands)
    lats, lons = st.calculate_coordinates(data)
    projected_data = st.smooth(st.project(lats, lons, band_data.data))
