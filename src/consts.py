PIREP_RELEVANCE_DURATION = 15  # mins

# Map Range

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

# Grid Range

GRID_RANGE = {
    "LAT": 1500,  # number of rows
    "LON": 2500,  # number of cols
    "ALT": len(MAP_RANGE["ALT"]["RANGE"]),  # number of z layers
}
