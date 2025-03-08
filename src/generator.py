import datetime as dt
import numpy as np
import keras
import consts
from pirep.defs.spreading import concatenate_all_pireps


class Generator(keras.utils.PyDataset):
    def __init__(
        self,
        data: dict,
        labels: dict,
        timestamps: list,
        width: int,
        stride: int,
        offset: int,
        background_risk: float,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.T = timestamps
        self.X, self.Y = data, labels

        self.width = width
        self.stride = stride
        self.offset = offset

        self.background_risk = background_risk

    def __len__(self):
        return (len(self.T) - self.offset) // (self.stride * self.width)

    def __getitem__(self, index):
        print(f"time: {index} / {self.T}")
        ts = self.T[index * self.offset : index * self.offset + self.width]

        xs = np.array([self.X[t] for t in ts])

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
        )

        print(xs.shape, ys.shape)

        return (xs, ys)
