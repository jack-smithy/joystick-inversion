from dataclasses import dataclass

import numpy as np

from constants import DIRECTION_MAP

N_MAGNETS = 4
N_SENSORS = 2
N_TILTS = (
    len(DIRECTION_MAP) - 1
)  # south, north, east, west (ground is the un-tilted rest)

# Per-unit manufacturing tolerances, as 1-sigma. These are the knobs to turn when
# matching simulated units to measured hardware.
POSITION_TOLERANCE = 1e-4  # 0.1mm  | magnet and sensor positions
SIZE_TOLERANCE = 1e-4  # 0.1mm      | on a 5mm cube edge
ANGLE_TOLERANCE = 1.0  # 1deg       | magnet/sensor orientations and tilt angles
POLARIZATION_TOLERANCE = 15e-3  # 15mT


@dataclass
class Parameters:
    """
    One joystick's geometry, magnetics and travel.

    Magnet quantities are 4 parallel entries, one per magnet. Positions are in m,
    angles in deg, polarizations in T.

    Orientation convention: `phi` is applied about global z, then `theta` about global y,
    both anchored at the part itself. `theta` only ever carries a small misalignment
    (~ANGLE_TOLERANCE), so the choice of convention is a second-order effect.
    """

    magnet_position: np.ndarray  # (4, 3) x, y, z
    magnet_phi: np.ndarray  # (4,)
    magnet_theta: np.ndarray  # (4,)
    magnet_direction: np.ndarray  # (4,) index into `direction_from_index`
    magnet_polarization: np.ndarray  # (4,)
    magnet_size: float  # cube edge length
    sensor_position: np.ndarray  # (2, 3); only sensor 1 is used so far
    sensor_phi: float  # on top of the -45deg mounting rotation
    sensor_theta: float
    tilt_angle: np.ndarray  # (4,) in DIRECTION_MAP order: south, north, east, west

    def __post_init__(self) -> None:
        # a mis-shaped array would broadcast silently through magpylib and quietly
        # give the wrong physics, so check rather than trust
        assert self.magnet_position.shape == (N_MAGNETS, 3), self.magnet_position.shape
        assert self.sensor_position.shape == (N_SENSORS, 3), self.sensor_position.shape
        assert self.tilt_angle.shape == (N_TILTS,), self.tilt_angle.shape

        for name in (
            "magnet_phi",
            "magnet_theta",
            "magnet_direction",
            "magnet_polarization",
        ):
            assert getattr(self, name).shape == (N_MAGNETS,), name


def magnetization_values() -> np.ndarray:
    """
    Measured magnetization values of the 4 magnets
    """
    return np.asarray((1.2124, 1.204, 1.208, 1.196))


def parameter_factory(generator: np.random.Generator | None = None) -> Parameters:
    """
    (Peter) Create the system parameters for magnet and sensor locations and orientations
    This vector is obtained from a joystick design optimization routine, it includes positions and orientations for the 4 magnets as well as positions of the 2 sensors. The sensor orientations are not obtained, but predefined for fabrication reasons

    (Jack) We only use sensor 1 in the model so far, not sure how correct sensor 2 positions are.

    Args:
        generator (np.random.Generator | None, optional): When given, per-unit
            tolerances are drawn from it, i.e. one distinct joystick per generator
            state. Defaults to None, which gives the nominal design.

    Returns:
        Parameters: The full set of system parameters for one joystick.
    """
    # ------------------------------------------------
    magnet_position = np.array(
        [
            (0.0181, 0.0, 0.0217),  # magnet 1  | x, y, z
            (-0.0183, 0.0, 0.0218),  # magnet 2 |
            (0.0, 0.018, 0.0217),  # magnet 3   |
            (0.0, -0.0181, 0.0218),  # magnet 4 |
        ]
    )
    # ------------------------------------------------
    magnet_phi = np.array((357.0, 181.0, 284.0, 270.0))  # about z
    magnet_theta = np.zeros((N_MAGNETS,))  # about y, nominally aligned
    # ------------------------------------------------
    # magnets 1 and 2 polarize along +y/-y, magnets 3 and 4 along -z/+z
    magnet_direction = np.array((2, 3, 5, 4))
    magnet_polarization = magnetization_values()
    # ------------------------------------------------
    magnet_size = 5e-3
    # ------------------------------------------------
    sensor_position = np.array(
        [
            (0.017, -0.0135, 0.0162),  # sensor 1 | x, y, z
            (0.0135, 0.0171, 0.0162),  # sensor 2 |
        ]
    )
    sensor_phi = 0.0
    sensor_theta = 0.0
    # ------------------------------------------------
    tilt_angle = np.full((N_TILTS,), 4.0)  # south, north, east, west

    if generator is not None:
        magnet_position += generator.normal(
            scale=POSITION_TOLERANCE,
            size=magnet_position.shape,
        )
        magnet_phi += generator.normal(scale=ANGLE_TOLERANCE, size=(N_MAGNETS,))
        magnet_theta += generator.normal(scale=ANGLE_TOLERANCE, size=(N_MAGNETS,))
        magnet_polarization += generator.normal(
            scale=POLARIZATION_TOLERANCE,
            size=(N_MAGNETS,),
        )
        magnet_size += generator.normal(scale=SIZE_TOLERANCE)

        sensor_position += generator.normal(
            scale=POSITION_TOLERANCE,
            size=sensor_position.shape,
        )
        sensor_phi += generator.normal(scale=ANGLE_TOLERANCE)
        sensor_theta += generator.normal(scale=ANGLE_TOLERANCE)

        tilt_angle += generator.normal(scale=ANGLE_TOLERANCE, size=(N_TILTS,))

        # magnet_direction is categorical, so it is deliberately never perturbed

    return Parameters(
        magnet_position=magnet_position,
        magnet_phi=magnet_phi,
        magnet_theta=magnet_theta,
        magnet_direction=magnet_direction,
        magnet_polarization=magnet_polarization,
        magnet_size=magnet_size,
        sensor_position=sensor_position,
        sensor_phi=sensor_phi,
        sensor_theta=sensor_theta,
        tilt_angle=tilt_angle,
    )
