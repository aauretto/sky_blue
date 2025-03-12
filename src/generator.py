import datetime as dt
import numpy as np
import keras
import consts
from pirep.defs.spreading import concatenate_all_pireps

import psutil
import os
import gc

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss  # Resident Set Size (RSS) in bytes
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
        print(f"\n\n\n************************* __getitem__ in generator start run Mem = {get_memory_usage()}*************************\n\n")
        batch_x = []
        batch_y = []
        print(f"\n\n\n************************* __getitem__ batch_x and batch_y initialized empty Mem = {get_memory_usage()}*************************\n\n")
        for batch_index in range(self.batch_size):
            print(f"\n\n\n************************* __getitem__ start batch index Mem = {get_memory_usage()}*************************\n\n")
            idx = index * self.batch_size + batch_index
            
            ts = self.T[
                idx * self.offset : idx * self.offset + self.width : self.stride
            ]
            print(f"\n\n\n************************* __getitem__ ts created from self.T Mem = {get_memory_usage()}*************************\n\n")

            
            xs = np.array([self.X[t] for t in ts])  # Shape (time, lat, lon, channels)
            print("SHAPE XS:", xs.shape)

            print(f"\n\n\n************************* __getitem__ xs created from self.X Mem = {get_memory_usage()}*************************\n\n")

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
            print(f"\n\n\n************************* __getitem__ ys created from concatenate_all_pireps call for t in ts Mem = {get_memory_usage()}*************************\n\n")

            if xs.shape[0] == 0 or ys.shape[0] == 0:
                print(f"Skipping empty sequence at batch {index}")
                continue  # Skip empty data
            
            batch_x.append(xs)
            print(f"\n\n\n************************* __getitem__ batch x appended xs Mem = {get_memory_usage()}*************************\n\n")
            batch_y.append(ys)
            print(f"\n\n\n************************* __getitem__ batch y appended ys Mem = {get_memory_usage()}*************************\n\n")
            

        # if len(batch_x) < self.batch_size:
        #     print(f"Skipping incomplete batch {index} with size {len(batch_x)}")
        #     return self.__getitem__((index + 1) % self.__len__())
        print(f"\n\n\n************************* __getitem__ end of batching loop Mem = {get_memory_usage()}*************************\n\n")
        np_batch_x = np.array(batch_x)  # Shape (batch_size, time, lat, lon, channels)
        print(f"\n\n\n************************* __getitem__ batch x np array converted Mem = {get_memory_usage()}*************************\n\n")
        np_batch_y = np.array(batch_y)  # Shape (batch_size, time, lat, lon, num_classes) # TODO 0.8 GB SPIKE, should be in place
        print(f"\n\n\n************************* __getitem__ batch y np array converted Mem = {get_memory_usage()}*************************\n\n")
        del batch_x
        del batch_y
        print(f"\n\n\n************************* __getitem__ about to collect batch_x, batch_y converted Mem = {get_memory_usage()}*************************\n\n")
        gc.collect()
        print(f"\n\n\n************************* __getitem__ collected batch_x, batch_y converted Mem = {get_memory_usage()}*************************\n\n")

        print(f"Batch {index}: X.shape={np_batch_x.shape}, Y.shape={np_batch_y.shape}")
        print(f"\n\n\n************************* __getitem__  about to return np converted batchx, batchy Mem = {get_memory_usage()}*************************\n\n")
        return np_batch_x, np_batch_y
