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


ALSO we want to define three regions in latitude that each have their own unit conversion from DEG => KM to account for wonky we live on a sphere bitch stuff

"""