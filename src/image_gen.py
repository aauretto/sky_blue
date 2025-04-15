#!/usr/bin/env python3

import datetime as dt

import numpy as np
from PIL import Image

import pirep as pr
from pirep.defs.spreading import concatenate_all_pireps

if __name__ == "__main__":
	reports = pr.parse_all(pr.fetch(pr.url(dt.datetime(2024, 11, 6, 23, 30, 0, tzinfo=dt.UTC), dt.datetime(2024, 11, 8, 0, 0, 0, tzinfo=dt.UTC))))
	grid = concatenate_all_pireps(reports, 4e-5)
	# TODO: add code for spreading PIREPs here?

	MAX = 1.0 # Set this to the maximum turbulence index/score

	# TODO: check that this breaks down the grid into each layer, and no more. There should be as many images as layers.
	for layer, idx in enumerate(grid):
		im = Image.fromarray(np.uint8(layer * 255 / MAX))
		im.save(f"layer{idx}.gif")
