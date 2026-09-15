DIRECTIONS = [
    "north",
    "south",
    "east",
    "west",
    "zero",
]

DIRECTION_MAP = {
    "south": 0,
    "north": 1,
    "east": 2,
    "west": 3,
    "zero": 4,
}

TRANSITIONS = [
    "noop",
    "cw",
    "ccw",
    "tilt_south",
    "tilt_north",
    "tilt_west",
    "tilt_east",
    "ground",
]

TRANSITION_MAP = {t: idx for (t, idx) in enumerate(TRANSITIONS)}

# DIRECTIONS is not in DIRECTION_MAP order, so it mislabels north/south when used to
# name classes. Use this where names have to line up with the tilt state index.
TILT_NAMES = sorted(DIRECTION_MAP, key=DIRECTION_MAP.__getitem__)

mT_TO_T = 1e-3

# Both 3-D sensors, flattened in the order `make_sensor_readings` returns them. Sensor 1
# alone leaves 9 of the 120 states closer to another state than to their own spread across
# units (worst pair `south/13` vs `east/18`, 0.23 mT apart against a 0.36 mT spread);
# adding sensor 2 takes that count to 0.
FIELD_COLUMNS = [f"B{axis}{sensor + 1}" for sensor in range(2) for axis in "xyz"]
