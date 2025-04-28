"""
File: PirepGird.py

Purpose:
    To export a PirepGrid class which contains the necessary information for 
    putting a pirep on our grid (see concatenante_all_pireps in __init__)
"""
class PirepGrid():
    """
    Class that holds information about a single turbulence PIREP
    """
    def __init__(self, lat_idx, lon_idx, alt_min_idx, alt_max_idx, turbulence_idx):
        """
        Constructor

        Parameters
        ----------
        lat_idx: int
            The index in a lat-lon grid of the latitude coordinate
        lon_idx: int
            The index in a lat-lon grid of the longitude coordinate
        alt_min_idx: int
            The index in a lat-lon grid of the minimum altitude this PIREP 
            covers
        alt_max_idx: int
            The index in a lat-lon grid of the maximum altitude this PIREP 
            covers
        turbulence_idx: float
            The intensity value of the turbulence
        """
        self.lat_idx = lat_idx
        self.lon_idx = lon_idx
        self.alt_min_idx = alt_min_idx
        self.alt_max_idx = alt_max_idx
        self.turbulence_idx = turbulence_idx