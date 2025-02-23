from pirep.defs.turbulence import Turbulence
from pirep.defs.aircraft import Aircraft

# Turbulence Index Lookup Table

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
