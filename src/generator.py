"""
File: generator.py
Description: A custom data generator for Keras that loads 
             and batches time-sequenced satellite imagery 
             and corresponding PIREP (pilot report) data for 
             training machine learning models.

Class:
    - Generator: A Keras-compatible data generator for time-sequenced satellite and PIREP data.
             
Usage: To use this generator, instantiate it with the required parameters and pass it to 
    the model's `.fit()` method in Keras.
    Example:
        generator = Generator(timestamps, y_times, y_reports, batch_size=32)
        model.fit(generator, epochs=5)
    Note: Generator needs to be instantiated for both x and y data
"""

from Logger import LOGGER
import datetime as dt
import numpy as np
import keras
import consts
from goes2go import GOES
import pirep as pr
import satellite as st
import xarray as xr
from numpy import random as rnd
from satellite_cache import retreive_satellite_data
from cacheWork_cacheReading import retrieve_from_pirep_cache
from concurrent.futures import ThreadPoolExecutor

class Generator(keras.utils.Sequence):
    """
    A custom data generator for Keras that loads and batches time-sequenced satellite imagery
    and corresponding PIREP (pilot report) data for training.

    This generator:
        - Prepares batches of multi-frame satellite data across specified bands.
        - Uses caching and parallel fetching (via ThreadPoolExecutor) to optimize data retrieval.
        - Generates matching labels using pre-cached or live PIREP data.
        - Supports shuffled batches for training and is compatible with Keras's fit methods.

    Input:
        timestamps (list[list[datetime.datetime]]): 
            nested list of datetime sequences representing time intervals available for sampling

        y_times (list[datetime.datetime]): 
            Timestamps corresponding to cached PIREP reports

        y_reports (list): 
            Cached parsed PIREP reports to be used when cache is enabled.

        batch_size (int): 
            Number of frames to return per batch.

        frame_size (int, optional): 
            Number of time steps per frame (default is 9, representing 8 hours + "now").

        sat (GOES, optional): 
            GOES satellite configuration for data retrieval. Defaults to GOES 16 CMPICABI product

        bands (list[int], optional): 
            List of satellite spectral bands to include in the dataset.
    """
    def __init__(
        self,
        timestamps: list[list[dt.datetime]],
        y_times,
        y_reports,
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
        self.y_times = y_times
        self.y_reports = y_reports

    def __len__(self):  # this returns the length of the window
        """
        returns the number of batches in the dataset, NOT the number of samples
        """

        # Integer division handles extra frames in batch for case where it is not evenly divisible
        return len(self.batch_timestamps) // self.batch_size

    def __retrieve_x_frame(self, timestamps: list[dt.datetime]):
        """
        Retrieves the x (satellite) data for a single frame (sequence of timestamps)

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

        # xs = [st.fetch(t, self.sat) for t in timestamps]

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
        """
        Fetches and combines live PIREP data for the specified time window.

        Input:
            timestamps: list[dt.datetime]
                the list of timestamps the satellite images will be fetched
        """
        delta_t = dt.timedelta(minutes=consts.PIREP_RELEVANCE_DURATION)
        return np.array(
            [
                pr.concatenate_all_pireps(
                    pr.parse_all(pr.fetch(pr.url(t - delta_t, t + delta_t))),
                    background_risk=4e-5,
                )
                for t in timestamps
            ]
        )

    def __cache_y_frame(self, timestamps: list[dt.datetime]):
        """
        Retrieves PIREP data from the in-memory cache if available.

        Input:
            timestamps: list[dt.datetime]
                the list of timestamps the satellite images will be fetched
        """
        delta_t = dt.timedelta(minutes=consts.PIREP_RELEVANCE_DURATION)
        frame = []
        for ts in timestamps:
            start_idx, end_idx = retrieve_from_pirep_cache(ts - delta_t, ts + delta_t, self.y_times)
            frame.append(pr.concatenate_all_pireps(self.y_reports[start_idx:end_idx], background_risk=4e-5))
        return np.array(frame)

    def __generate_frames(self, timestamp: dt.datetime) -> list[dt.datetime]:
        """
        Generates a list of timestamps covering one full frame window.
        Input:
            timestamps: list[dt.datetime]
                the list of timestamps the satellite images will be fetched
        """
        return [timestamp + dt.timedelta(hours=i) for i in range(self.frame_size)]

    def __getitem__(self, batch_index):  # Returns one batch of (X, Y) data at the specified batch index.
        """
        Gets a single batch of data 

        Input:
            batch_index: int
                The index of the batch to retrieve
    
        Output:
            (np.NDArray, np.NDArray)
                The x and y data of the batch
        """
        batch_x = []
        batch_y = []

        batch_times = self.batch_timestamps[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]
        frames_timestamps = [self.__generate_frames(t) for t in batch_times]            

        
        def get_frames_worker(frame_times):
            try:
                # get from cache
                xs, updated_timestamps = retreive_satellite_data(frame_times)
                updated_timestamps = [updated_timestamps[-1] + i * dt.timedelta(hours=1) for i in range(self.frame_size)]
                ys = self.__cache_y_frame(updated_timestamps)
                return (xs, ys)
            except Exception as e:
                LOGGER.debug(f"Got error {e} in get_frames_worker with timestamps {frame_times}", exc_info=True)

                xs, updated_timestamps = self.__retrieve_x_frame(frame_times)

                # We want the timestamps for the last xs (now) and the next 8 hours
                updated_timestamps = [updated_timestamps[-1] + i * dt.timedelta(hours=1) for i in range(self.frame_size)]

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
