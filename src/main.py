#!/usr/bin/env python3
from Logger import LOGGER
import datetime as dt

import numpy as np
import pandas as pd
from goes2go import GOES
from matplotlib import pyplot as plt

import pirep as pr
import satellite as st
from consts import MAP_RANGE, BACKGROUND_RISK
from pirep.defs.spreading import concatenate_all_pireps, spread_pirep

if __name__ == "__main__":
    reports = pr.parse_all(pr.fetch(pr.url(dt.datetime(2024, 11, 6, 23, 30, 0, tzinfo=dt.UTC), dt.datetime(2024, 11, 8, 0, 0, 0, tzinfo=dt.UTC))))
    print(f"Number of reports in specified range is {len(reports)}")

    grid = concatenate_all_pireps(reports, BACKGROUND_RISK)
    print(f'Concatenated reports')


    ### Old Single Pirep Stuff
    # grids = list(map(lambda row: {"Timestamp": row["Timestamp"], "Grid": pr.compute_grid(row)}, reports))
    # print("COMPUTED")
    # # grid = concatenate_all_pireps(reports)

    # # TO CSV the grids
    # see_vals = {"Timestamp" : [],
    #             "Places"    : []}
    # for i in range(len(grids)):
    #     see_vals["Timestamp"].append(grids[i]["Timestamp"])
    #     grid, aircraft, intensity = grids[i]["Grid"]
    #     see_vals["Places"].append((aircraft, intensity, np.argwhere(~np.isnan(grid))))
    # print("SEEN")
    # see_vals = pd.DataFrame(see_vals)
    # see_vals.to_csv('./grids.csv')


    # grid, aircraft, intensity = grids[2]["Grid"]
    # print(aircraft, intensity)
    # spread_pirep(grid, aircraft, intensity, BACKGROUND_RISK)
    # print(f"PIREP of {aircraft} aircraft and {intensity} intensity")
    # print(f"has values in {np.argwhere(~np.isnan(grid))}")

    # x, y, z = np.indices(grid.shape)
    # x, y, z, values = x.flatten(), y.flatten(), z.flatten(), grid.flatten()


    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # sc = ax.scatter(x, y, z, c=values, cmap='viridis', s=1, alpha=.5)  # Adjust alpha for transparency
    # ax.set_facecolor('black')
    # ax.tick_params(axis='x', colors='white')
    # ax.tick_params(axis='y', colors='white')
    # ax.tick_params(axis='z', colors='white')
    # ax.set_xlim(0, 1500)
    # ax.set_ylim(0, 2500)


    # # Add color bar
    # plt.colorbar(sc)

    # plt.savefig("/skyblue/spread.png")
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
