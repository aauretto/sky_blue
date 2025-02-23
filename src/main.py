#!/usr/bin/env python3

import datetime as dt
import pandas as pd
import pirep as pr
import satellite as st
from goes2go import GOES
from pirep.defs.spreading import vertical_spread, radial_spread
import numpy as np
from matplotlib import pyplot as plt


# from utils.compute_grid import compute_grid

if __name__ == "__main__":
    # Retrieve PIREPs
    reports: pd.DataFrame = pr.parse_all(
        pr.fetch(
            pr.url(
                date_s=dt.datetime(2024, 11, 6, 23, 54, 0),
                date_e=dt.datetime(2024, 11, 7, 0, 0, 0)
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

    grid, _aircraft, intensity = grids["Grid"].iloc[0]

    vertical_spread(grid, 'SEV')
    radial_spread(grid, 'SEV')

    x, y, z = np.indices(grid.shape)
    x, y, z, values = x.flatten(), y.flatten(), z.flatten(), grid.flatten()


    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=values, cmap='viridis', s=1, alpha=.5)  # Adjust alpha for transparency
    ax.set_facecolor('black')
    ax.tick_params(axis='x', colors='white')  
    ax.tick_params(axis='y', colors='white') 
    ax.tick_params(axis='z', colors='white')
    # ax.set_xlim(0, 1500)
    # ax.set_ylim(0, 2500)


    # Add color bar
    plt.colorbar(sc)

    plt.show()

    # fig = plt.figure()
    # vals = np.argwhere(~np.isnan(grid))

    # vals = vals[vals[:, 2].argsort()]
    # points = [grid[*val] for val in vals]
    # plt.plot(vals[:, 2], points)

    # plt.savefig("./vertical_spread.png", dpi=300, bbox_inches="tight")

    # Initialize satellites
    sat_east = GOES(satellite=16, product="ABI", domain="C")
    bands = [8, 9, 10, 13, 14, 15]

    # Fetch satellite data and project onto grid
    print("About to Fetch Range")
    data = st.fetch_range(
        dt.datetime(2024, 11, 6, 0, 30),
        dt.datetime(2024, 11, 6, 1, 0),
        sat_east,
    )
    #Work
    # data = st.fetch_range(
    #     dt.datetime(2025, 1, 19, 23, 59),
    #     dt.datetime(2025, 1, 20, 00, 14),
    #     sat_east,
    # )
    print("Fetched Range")
    band_data = st.fetch_bands(data, bands)
    print("Fetched Bands")
    lats, lons = st.calculate_coordinates(data)
    print("Calced Coords")
    unsmoothed_data = st.project(lats, lons, band_data.data)
    print(f"{unsmoothed_data.shape=}")
    datum = unsmoothed_data[0, :, :, 0]
    print(f"{datum.shape=}")




    projected_data = st.smooth(datum)
    fig = plt.figure(figsize=(15, 12))
    ax_east = fig.add_subplot(1, 1, 1)
    ax_east.pcolormesh(projected_data)
    plt.show()
    plt.clf()

    # print(f"{projected_data.shape=}")
    # num_bands = projected_data.shape[3]
    # for i in range(num_bands):
    #     fig = plt.figure(figsize=(15, 12))
    #     ax_east = fig.add_subplot(1, 1, 1)
    #     ax_east.pcolormesh(projected_data[0][:, :][i])
    #     plt.show()
    #     plt.clear()