from pirep.defs.aircraft import Aircraft
from pirep.defs.turbulence import Turbulence

# Turbulence Index Lookup Table
"""
The final mapping used by output data throughout the project

A certain type of turbulence reported from a certain aircraft represents a specific value

The values attempt to encapsulate 2 pieces of information:
    Intensity: the intensity of the turbulence
    Confidence: how sure we are a turbulence event is there 
"""
TURBULENCE_INDEXES = {
    Aircraft.LGT: {
        Turbulence.Intensity.NEG: 0.0,
        Turbulence.Intensity.LGT: 0.1,
        Turbulence.Intensity.MOD: 0.2,
        Turbulence.Intensity.SEV: 0.4,
    },
    Aircraft.MED: {
        Turbulence.Intensity.NEG: 0.1,
        Turbulence.Intensity.LGT: 0.2,
        Turbulence.Intensity.MOD: 0.4,
        Turbulence.Intensity.SEV: 0.8,
    },
    Aircraft.HVY: {
        Turbulence.Intensity.NEG: 0.2,
        Turbulence.Intensity.LGT: 0.4,
        Turbulence.Intensity.MOD: 0.8,
        Turbulence.Intensity.SEV: 1.0,
    },
}
