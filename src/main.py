#!/usr/bin/env python3

import datetime as dt

import numpy as np
import pandas as pd
from goes2go import GOES
from matplotlib import pyplot as plt

import pirep as pr
import satellite as st
from consts import MAP_RANGE
from pirep.defs.spreading import concatenate_all_pireps, spread_pirep

if __name__ == "__main__":
    reports = pr.fetch_parse_and_drop_irrelevant(dt.datetime(2024, 11, 6, 23, 54, 0), dt.datetime(2024, 11, 7, 0, 0, 0))
    # Convert reports to grids
    grids = pd.DataFrame(
        {
            "Timestamp": reports["Timestamp"],
            "Grid": reports.apply(pr.compute_grid, axis=1),
        }
    )
    print("COMPUTED")
    # grid = concatenate_all_pireps(reports)
    see_vals = {"Timestamp" : [],
                "Places"    : []}
    for i in range(len(grids)):
        see_vals["Timestamp"].append(grids["Timestamp"].iloc[i])
        grid, aircraft, intensity = grids["Grid"].iloc[i]
        see_vals["Places"].append((aircraft, intensity, np.argwhere(~np.isnan(grid))))
    print("SEEN")
    see_vals = pd.DataFrame(see_vals)
    see_vals.to_csv('./grids.csv')
    grid, aircraft, intensity = grids["Grid"].iloc[0]
    spread_pirep(grid, aircraft, intensity)
    print(f"PIREP of {aircraft} aircraft and {intensity} intensity")
    print(f"has values in {np.argwhere(~np.isnan(grid))}")

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

    plt.savefig("/skyblue/spread.png")
    # fig = plt.figure()
    # vals = np.argwhere(~np.isnan(grid))

    # vals = vals[vals[:, 2].argsort()]
    # points = [grid[*val] for val in vals]
    # plt.plot(vals[:, 2], points)

    # plt.savefig("./vertical_spread.png", dpi=300, bbox_inches="tight")




###### Satellite Stuff ############

# print(f"{projected_data.shape=}")
# num_bands = projected_data.shape[3]
# for i in range(num_bands):
#     fig = plt.figure(figsize=(15, 12))
#     ax_east = fig.add_subplot(1, 1, 1)
#     ax_east.pcolormesh(projected_data[0][:, :][i])
#     plt.show()
#     plt.clear()
