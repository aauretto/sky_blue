import datetime as dt
import numpy as np
import keras
import consts
from pirep.defs.spreading import concatenate_all_pireps


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
            batch_y.append(ys)

        # if len(batch_x) < self.batch_size:
        #     print(f"Skipping incomplete batch {index} with size {len(batch_x)}")
        #     return self.__getitem__((index + 1) % self.__len__())

        batch_x = np.array(batch_x)  # Shape (batch_size, time, lat, lon, channels)
        batch_y = np.array(batch_y)  # Shape (batch_size, time, lat, lon, num_classes)

        print(f"Batch {index}: X.shape={batch_x.shape}, Y.shape={batch_y.shape}")
        return batch_x, batch_y
