PIREP_RELEVANCE_DURATION = 15  # mins

# Map Range: the real lat/lon box we use to define CONUS
MAP_RANGE = {
    "LAT": {
        "MIN": 22,  # degrees N
        "MAX": 53,  # degrees N
    },
    "LON": {
        "MIN": -131,  # degrees E
        "MAX": -66,  # degrees E
    },
    "ALT": {
        "RANGE": [
            1000,
            3000,
            6000,
            9000,
            12000,
            15000,
            18000,
            21000,
            24000,
            27000,
            30000,
            36000,
            42000,
            48000,
        ]  # TODO maybe a 0
    },
}

# Grid Range: the dimensions of the grid we use to put data onto
GRID_RANGE = {
    "LAT": 1500,  # number of rows
    "LON": 2500,  # number of cols
    "ALT": len(MAP_RANGE["ALT"]["RANGE"]),  # number of z layers
}

BACKGROUND_RISK = 4e-5
