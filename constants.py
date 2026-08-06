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

# DIRECTIONS is not in DIRECTION_MAP order, so it mislabels north/south when used to
# name classes. Use this where names have to line up with the tilt state index.
TILT_NAMES = sorted(DIRECTION_MAP, key=DIRECTION_MAP.__getitem__)

mT_TO_T = 1e-3
