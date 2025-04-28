"""
File: merge.py

Description:
    An aritifact of how we used to merge pireps. Left for later users to 
    consider for a refactor
"""

import numpy as np
from scipy.ndimage import maximum_filter

#TODO make sure float 32s stay the same when we get to this
def merge_max(pireps):
    """This is a naive merge function. It just takes the max value at each point"""
    return np.nanmax(pireps, axis=0)


# =================== Helper Functions for merge spread =======================
def compute_overlap(arrays):
    """Computes the number of nonzero contributors at each (x, y) position."""
    stacked = np.stack(arrays, axis=0)  # Shape: (N, X, Y, Z)
    overlap = np.sum(~np.isnan(stacked), axis=0)  # Count non-NaN values
    return overlap


def get_spread_radius(overlap):
    """Maps an overlap count to a spread radius. Radius = count - 1."""
    spread_map = np.zeros_like(overlap, dtype=int)
    for i in range(2, np.max(overlap)):
        spread_map[overlap >= i] = (
            i - 1
        )  # If at least 2 overlaps, use (i-1)x(i-1) spread
    return spread_map


def spread_max_values(merged, overlap_map, spread=1):
    """Applies spreading to max values based on overlap map."""
    spread_radius = get_spread_radius(overlap_map)
    safe_merged = np.where(np.isnan(merged), -np.inf, merged)
    output = safe_merged.copy()

    # Apply max filtering with different spread sizes
    for radius in np.unique(spread_radius):
        if radius > 0:
            mask = spread_radius == radius
            filtered = maximum_filter(
                safe_merged, size=(2 * spread * radius + 1, spread * radius + 1, 1)
            )
            output[mask] = filtered[mask]
    return output


# =================== Max Merge Spread (max merge 2.0) =======================
def merge_max_spread(pireps, spread=1):
    """Merge with a spread function that accounts for overlap.
    More overlap means the it will pull the max from a wider area.
    On purpose does not spread in the alititude dimension."""
    merged = merge_max(pireps)  # Step 1: Element-wise max
    overlap_map = compute_overlap(pireps)  # Step 2: Compute overlap
    if np.any(overlap_map > 1):  # Step 3: Spread if overlap exists
        return spread_max_values(merged, overlap_map, spread)
    else:
        print("No Overlap...")
        return merged


def merge_overlap_spread(max_map, overlap_map, spread=1):
    """Merge with a spread function that accounts for overlap.
    More overlap means the it will pull the max from a wider area.
    On purpose does not spread in the alititude dimension.
    Note: This does the same as above but takes the max and overlap array as
    inputs instead of calculating them."""
    if np.any(overlap_map > 1):
        return spread_max_values(max_map, overlap_map, spread)
    else:
        print("No Overlap...")
        return max_map


# ============================ Test Cases =====================================
# Testing and example usage of merge max:
test_merge_max = False
if test_merge_max:
    array1 = np.array(
        [
            [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
            [[1, 0, 0], [1, 0, 0], [1, 0, 0]],
        ]
    )
    array2 = np.array(
        [
            [[1, 2, 0], [1, 2, 0], [1, 2, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]
    )
    array3 = np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [3, 3, 3]],
            [[3, 3, 3], [3, 3, 3], [3, 3, 3]],
        ]
    )

    merged_array = merge_max([array1, array2, array3])
    print(merged_array)

# Testing and example usage of merge max spread.
test_merge_max_spread = False
if test_merge_max_spread:
    array1 = np.zeros((5, 4, 3))
    array2 = np.zeros((5, 4, 3))
    array3 = np.zeros((5, 4, 3))

    array1[2, 0, :] = 7
    array1[2, 1, :] = 5
    array1[2, 2, :] = 5
    array2[2, 2, :] = 3
    array3[2, 2, :] = 4  # Overlap at (2,2), spreading should activate

    array1[0, 0, :] = 1
    array2[0, 0, :2] = 1
    array3[0, 0, :1] = 1  # Overlap at (0,0), spreading should activate

    array1[0, 1, 0] = 2
    array1[0, 2, 0] = 3
    array1[0, 1, 1] = 2
    array1[0, 1, 2] = 5

    merged_result = merge_max_spread([array1, array2, array3])
    print(merged_result)
