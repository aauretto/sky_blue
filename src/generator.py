import datetime as dt
import numpy as np
import keras
import consts
from goes2go import GOES
import pirep as pr
import satellite as st
import xarray as xr
from numpy import random as rnd


from pirep.defs.spreading import concatenate_all_pireps
from concurrent.futures import ThreadPoolExecutor
import threading

class Generator(keras.utils.Sequence):
    def __init__(
        self,
        timestamps: list[list[dt.datetime]],
        batch_size: int,
        frame_size: int = 9,
        sat: GOES = GOES(satellite=16, product="ABI", domain="C"),
        bands: list[int] = [8, 9, 10, 13, 14, 15],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.rng = rnd.default_rng()
        self.timestamps = timestamps
        self.batch_timestamps = [
            timestamp
            for sequence in self.timestamps
            for timestamp in sequence[:len(sequence) - frame_size + 1]
        ]
        self.rng.shuffle(self.batch_timestamps)
        self.batch_size = batch_size  # the size of a batch
        self.frame_size = frame_size  # number of (satellite, pirep) instances in a frame should be 9 because 8hrs + now
        self.sat = sat
        self.bands = bands

    def __len__(self):  # this returns the length of the window
        """
        returns the number of batches in the dataset, NOT the number of samples
        """

        # Integer division handles extra frames in batch for case where it is not evenly divisible
        return len(self.batch_timestamps) // self.batch_size

    def __retrieve_x_frame(self, timestamps: list[dt.datetime]):
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
        
        def worker(ts):
            return st.fetch(ts, self.sat)
        
        numProcs = len(timestamps)
        with ThreadPoolExecutor(max_workers=numProcs) as exec:
            xs = exec.map(worker, timestamps)

        xs = xr.concat(xs, dim="t")
        lats, lons = st.calculate_coordinates(xs)
        xs = st.fetch_bands(xs, self.bands)

        updated_timestamps = [
            dt.datetime.fromtimestamp(
                timestamp.astype("datetime64[s]").astype(int), dt.UTC
            )
            for timestamp in np.array(xs.coords["t"], dtype=np.datetime64)
        ]
        xs = st.smooth(st.project(lats, lons, xs.data))
        assert xs.shape[1:] == (
            consts.GRID_RANGE["LAT"],
            consts.GRID_RANGE["LON"],
            len(self.bands),
        )
        return xs, updated_timestamps

    def __retrieve_y_frame(self, timestamps: list[dt.datetime]):
        delta_t = dt.timedelta(minutes=consts.PIREP_RELEVANCE_DURATION)
        return np.array(
            [
                concatenate_all_pireps(
                    pr.parse_all(pr.fetch(pr.url(t - delta_t, t + delta_t))),
                    background_risk=4e-5,
                )
                for t in timestamps
            ]
        )

    def __generate_frames(self, timestamp: dt.datetime) -> list[dt.datetime]:
        return [timestamp + dt.timedelta(hours=i) for i in range(self.frame_size)]

    def __getitem__(self, batch_index):  # gets a batch
        batch_x = []
        batch_y = []

        batch_times = self.batch_timestamps[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]
        frames_timestamps = [self.__generate_frames(t) for t in batch_times]            

        
        def get_frames_worker(frame_times):
            print(f"\n\n\n{frame_times=}\n")
            xs, updated_timestamps = self.__retrieve_x_frame(frame_times)
            print(f"\n{updated_timestamps=}\n")
            updated_timestamps.reverse()
            updated_timestamps = [uts + i * dt.timedelta(hours=self.frame_size) for i, uts in enumerate(updated_timestamps)]
            print(f"\n{updated_timestamps=}\n\n\n")

            ys = self.__retrieve_y_frame(updated_timestamps)
            return (xs, ys)

        with ThreadPoolExecutor(max_workers=len(frames_timestamps)) as exec:
            frame_pairs = exec.map(get_frames_worker, frames_timestamps)

        for (xs, ys) in frame_pairs:
            batch_x.append(xs)
            batch_y.append(ys)

        batch_x = np.array(batch_x)  # Shape (batch_size, time, lat, lon, bands)
        batch_y = np.array(batch_y)  # Shape (batch_size, time, lat, lon, alt)

        return batch_x, batch_y
