import numpy as np

"""
How are we gonna spread?

1 fx for each severity {NONE, LIGHT, MOD, SEVERE} based on colorful graphs from paper that WxExt sent us (in dis)

Each function takes:
    (grid , aircraft type , severity) => implies compute_grid needs to return this tuple or aircraft/severity needs to come from elsewhere.

That funciton:
1) looks into the grid for the point that is nonzero, ie contains the risk val from the lookup table in
   the locaiton the pirep was reported.
2) Define some mathematical curve that depends on distance from the src point (from (1)) in x and z
3) apply that curve to each point on the grid


ALSO we want to define three regions in latitude that each have their own unit conversion from DEG => KM to account for wonky we live on a sphere stuff

"""

# Data Source: https://journals.ametsoc.org/view/journals/bams/aop/BAMS-D-23-0142.1/BAMS-D-23-0142.1.pdf

### Altitudinal Computations ### Taken from the page 14 graphs of the source above
NEG_RISKS_ALT = [
    [-4, -3.3, -2.5, -2, -1.5, 0, 1, 1.5, 2.25, 3.5, 6, np.inf],
    [1, 0.97, 0.95, 0.93, 0.91, 0.89, 0.91, 0.93, 0.95, 0.97, 0.99, 1],
]
LGT_RISKS_ALT = [
    [-4, -3, -1.6, 0.6, 2, 3.6, 6, np.inf],
    [0, 0.075, 0.15, 0.75, 0.15, 0.075, 0.035, 0],
]
MOD_RISKS_ALT = [
    [-4, -3, 1.5, 2.75, 4, 6, np.inf],
    [0, 0.15, 0.35, 0.15, 0.075, 0.035, 0],
]
SEV_RISKS_ALT = [
    [-4, -3.5, -1, -0.5, 1.25, 2.5, 3.5, 4.5, 6, np.inf],
    [0, 0.35, 0.75, 1.0, 0.75, 0.35, 0.15, 0.075, 0.035, 0],
]
