import datetime as dt
import pandas as pd
import pirep as pr

# Retrieve PIREPs
reports: pd.DataFrame = pr.parse_all(
    pr.fetch(
        pr.url(
            date_s=dt.datetime(2024, 11, 16, 0, 0, 0, tzinfo=dt.UTC),
            date_e=dt.datetime(2024, 11, 17, 0, 0, 0, tzinfo=dt.UTC),
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
