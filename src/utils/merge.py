import numpy as np

def merge_max(pireps):
    # This is a naive merge function. It just takes the max value at each point
    return np.maximum.reduce(pireps)

def merge_max_spread(pireps):
    # TODO Merge with a spread function that accounts for overlap
    # More overlap means the it will pull the max from a wider area
    return merge_max(pireps)

# Example usage:
array1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
array2 = np.array([[[2, 1], [4, 3]], [[6, 5], [8, 7]]])
array3 = np.array([[[0, 3], [1, 5]], [[4, 2], [6, 9]]])

merged_array = merge_max([array1, array2, array3])
print(merged_array)
