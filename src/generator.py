import datetime as dt
import numpy as np
import keras
import consts
from pirep.defs.spreading import concatenate_all_pireps
import psutil
import os
import sys

def get_memory_info():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        "RSS (Resident Set Size)": f"{mem_info.rss / 1024 ** 2:.2f} MB",
        "VMS (Virtual Memory Size)": f"{mem_info.vms / 1024 ** 2:.2f} MB",
        "Shared Memory": f"{mem_info.shared / 1024 ** 2:.2f} MB",
        "Text (code)": f"{mem_info.text / 1024 ** 2:.2f} MB",
        "Data + Stack": f"{mem_info.data / 1024 ** 2:.2f} MB",
        "Dirty Pages": f"{mem_info.dirty / 1024 ** 2:.2f} MB",
        "PID": f"{os.getpid()}"
    }


class Generator(keras.utils.Sequence):
    def __init__(
        self,
        data: dict,
        labels: dict,
        timestamps: list,
        width: int,
        stride: int,
        offset: int,
        background_risk: float,
        batch_size: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.T = timestamps
        self.X, self.Y = data, labels

        self.width = width
        self.stride = stride
        self.offset = offset
        self.batch_size = batch_size

        self.background_risk = background_risk

    def __len__(self):  # this returns the length of the window
<<<<<<< Updated upstream
            # TODO PyDatset docstring says that len method should return the number of batches in the dataset rather than the number of samples
        return (
            (len(self.T) - self.offset) // (self.width - self.offset) // self.batch_size
        )

    def __getitem__(self, index):
        batch_x = []
        batch_y = []

        for batch_index in range(self.batch_size):
            idx = index * self.batch_size + batch_index

            ts = self.T[
                idx * self.offset : idx * self.offset + self.width : self.stride
            ]
=======
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
        print(f"\n\n\n\n SATELLITE DATA RETRIEVAL STARTING __retrieve_x_frame {get_memory_info()}")
        xs = [st.fetch(t, self.sat) for t in timestamps]
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
        print(f"\n\n\n\n SATELLITE DATA RETRIEVAL RETURNING __retrieve_x_frame {get_memory_info()}")
        return xs, updated_timestamps

    def __retrieve_y_frame(self, timestamps: list[dt.datetime]):
        print(f"\n\n\n\n PIREP DATA STARTING __retrieve_y_frame {get_memory_info()}")
        delta_t = dt.timedelta(minutes=consts.PIREP_RELEVANCE_DURATION)
        print(f"\n\n\n\n ABOUT TO CONCETENATE ALL BATCH PIREPS __retrieve_y_frame{get_memory_info()}")
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
        print("\n\n\n\n ABOUT TO GO INTO __getitem__")
        batch_x = []
        batch_y = []

        batch_times = self.batch_timestamps[
            batch_index * self.batch_size : (batch_index + 1) * self.batch_size
        ]
        print(f"\n\n\n\n TIMESTAMPS ABOUT TO RETRIEVE __getitem__{get_memory_info()} \n for a length of batch times of {len(batch_times)}")
        frames_timestamps = [self.__generate_frames(t) for t in batch_times]
        print(f"\n\n\n\n TIMESTAMPS RETRIEVED __getitem__{get_memory_info()}")

        # Warning: be careful of incomplete batches (len(batch_times) != batch_size )
        for frame_times in frames_timestamps:
            print(f"\n\n\n\n SATTELITE DATA ABOUT TO RETRIEVE __getitem__{get_memory_info()}")
            xs, updated_timestamps = self.__retrieve_x_frame(frame_times)
            print(f"\n\n\n\n SATTELITE DATA RETRIEVED __getitem__{get_memory_info()}")
            ys = self.__retrieve_y_frame(updated_timestamps)
            print(f"\n\n\n\n PIREP DATA RETRIEVED __getitem__{get_memory_info()}")
>>>>>>> Stashed changes

            xs = np.array([self.X[t] for t in ts])  # Shape (time, lat, lon, channels)
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
                print(f"Skipping empty sequence at batch {index}")
                continue  # Skip empty data

            batch_x.append(xs)
            print(f"\n\n\n\n BATCH X APPENDED __getitem__{get_memory_info()}")
            batch_y.append(ys)
            print(f"\n\n\n\n BATCH Y APPPENDED __getitem__{get_memory_info()}")

<<<<<<< Updated upstream
        # if len(batch_x) < self.batch_size:
        #     print(f"Skipping incomplete batch {index} with size {len(batch_x)}")
        #     return self.__getitem__((index + 1) % self.__len__())
=======
        print(f"\n\n\n\n GET ITEM FOR LOOP ENDED __getitem__{get_memory_info()}\n {sys.getsizeof(batch_x)=} {sys.getsizeof(batch_y)=}")
        batch_x = np.array(batch_x)  # Shape (batch_size, time, lat, lon, bands)
        print(f"\n\n\n\n NUMPYED BATCH X __getitem__{get_memory_info()} \n {sys.getsizeof(batch_x)=}")
        batch_y = np.array(batch_y)  # Shape (batch_size, time, lat, lon, alt)
        print(f"\n\n\n\n NUMPYED BATCH y __getitem__{get_memory_info()} \n {sys.getsizeof(batch_y)=}")
>>>>>>> Stashed changes

        batch_x = np.array(batch_x)  # Shape (batch_size, time, lat, lon, channels)
        batch_y = np.array(batch_y)  # Shape (batch_size, time, lat, lon, num_classes)

        print(f"Batch {index}: X.shape={batch_x.shape}, Y.shape={batch_y.shape}")
        return batch_x, batch_y
