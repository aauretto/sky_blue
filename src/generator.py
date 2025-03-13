import datetime as dt
import numpy as np
import keras
import consts
from goes2go import GOES
import pirep as pr
import satellite as st
import xarray as xr
import datetime as dt

from pirep.defs.spreading import concatenate_all_pireps

class Generator(keras.utils.Sequence):
    def __init__(
        self,
        timestamps: list,
        width: int,
        stride: int,
        offset: int,
        background_risk: float,
        batch_size: int,
        sat: GOES = GOES(satellite=16, product="ABI", domain="C"),
        bands: list[int] = [8, 9, 10, 13, 14, 15],
        **kwargs,
    ): #TODO no more background risk
        super().__init__(**kwargs)
        self.times = timestamps
        self.width = width
        self.stride = stride
        self.offset = offset # TODO what are these value
        self.batch_size = batch_size
        self.background_risk = background_risk,
        self.sat = sat,
        self.bands = bands

    def __len__(self):  # this returns the length of the window
            # TODO PyDatset docstring says that len method should return the number of batches in the dataset rather than the number of samples
        return (
            (len(self.times) - self.offset) // (self.width - self.offset) // self.batch_size # TODO Document this math
        )


    def __retreive_x_frame(self, timestamps: list[dt.datetime]):
        """
        Retrieves the x (satellite) data for a single frame given the relevant timestamps for 

        Parameters
        ----------
        timestamps: list(dt.datetime)
            the list of timestamps the satellite images will be fetched
        
        Returns
        -------
        the satellite images and their corresponding, updated timestamps
        
        """
        xs = [st.fetch(t, self.sat) for t in timestamps]
        xs = xr.concat(xs, dim='t')
        lats, lons = st.calculate_coordinates(xs)
        xs = st.fetch_bands(xs, self.bands)
        updated_timestamps = [dt.datetime.fromtimestamp(ts.astype('datetime64[s]').astype(int), dt.UTC) for ts in np.array(xs.coords["t"], dtype=np.datetime64)]
        xs = st.smooth(st.project(lats, lons, xs.data))
        assert xs.shape[1:] == (
            consts.GRID_RANGE["LAT"],
            consts.GRID_RANGE["LON"],
            len(self.bands),
        )
        return xs, updated_timestamps


    def __retreive_y_frame(self, timestamps: list[dt.datetime]):

    
    def __getitem__(self, batch_index): #gets a batch
        batch_x = []
        batch_y = []

        for sample_index in range(self.batch_size):
            # TODO document these
            idx = batch_index * self.batch_size + sample_index

            ts = self.times[
                idx * self.offset : idx * self.offset + self.width : self.stride
            ]

            xs, updated_timestamps = self.__retreive_x_frame(ts)
            
            # xs = np.array([self.X[t] for t in ts])  # Shape (time, lat, lon, channels)
            print("SHAPE XS:", xs.shape)

            ys = np.array(
                [
                    concatenate_all_pireps(
                        [
                            frame
                            for t_frame, frame in self.Y.items()
                            if abs((t_frame - t) / dt.timedelta(minutes=1))
                            <= consts.PIREP_RELEVANCE_DURATION
                        ],
                        self.background_risk,
                    )
                    for t in ts
                ]
            )  # Shape (time, lat, lon, num_classes)
            print("SHAPE YS:", ys.shape)

            if xs.shape[0] == 0 or ys.shape[0] == 0:
                print(f"Skipping empty sequence at batch {batch_index}")
                continue  # Skip empty data

            batch_x.append(xs)
            batch_y.append(ys)

        # if len(batch_x) < self.batch_size:
        #     print(f"Skipping incomplete batch {index} with size {len(batch_x)}")
        #     return self.__getitem__((index + 1) % self.__len__())

        batch_x = np.array(batch_x)  # Shape (batch_size, time, lat, lon, channels)
        batch_y = np.array(batch_y)  # Shape (batch_size, time, lat, lon, num_classes)

        print(f"Batch {batch_index}: X.shape={batch_x.shape}, Y.shape={batch_y.shape}")
        return batch_x, batch_y
