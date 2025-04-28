#!/usr/bin/env python3

import datetime as dt

import numpy as np
from PIL import Image

import pirep as pr


if __name__ == "__main__":
	day_start = dt.datetime(2024, 2, 2, 0, tzinfo=dt.UTC)
	one_day = dt.timedelta(days=1)

	for day in range(9):
		day_end = day_start + one_day

		reports = pr.parse_all(pr.fetch(pr.url(day_start, day_end)))
		grid = pr.concatenate_all_pireps(reports, 4e-5)

		MAX = 1.0 # Set this to the maximum turbulence index/score

		# TODO: check that this breaks down the grid into each layer, and no more. There should be as many images as layers.
		print(f"Got {day_start} generating images")
		for idx in range(14):

			im = Image.fromarray(np.uint8(grid[:, :, idx] * 255 / MAX))
			im.save(f"/skyblue/images_for_demo/{day_start.strftime('%Y_%m_%d')}alt{idx:02}.gif")
		day_start = day_end
