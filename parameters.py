import numpy as np


def parameter_factory(
    calibration: np.ndarray | None = None,
    *,
    generator: np.random.Generator | None = None,
) -> np.ndarray:
    """
    (Peter) Create the system parameters for magnet and sensor locations and orientations
    This vector is obtained from a joystick design optimization routine, it includes positions and orientations for the 4 magnets as well as positions of the 2 sensors. The sensor orientations are not obtained, but predefined for fabrication reasons

    (Jack) We only use sensor 1 in the model so far, not sure how correct sensor 2 positions are.

    Args:
        calibration (np.ndarray | None, optional): Measured calibration offsets. Defaults to None.
        uncertainty (bool): whether to add uncertainty to the parameters

    Returns:
        np.ndarray: Array describing the full set of system parameters
    """
    x = np.zeros((27,))
    # ------------------------------------------------
    x[0] = 0.0217  # z magnet 1   | magnet positions
    x[1] = 0.0218  # z magnet 2   |
    x[2] = 0.0217  # z magnet 3   |
    x[3] = 0.0218  # z magnet 4   |
    x[4] = 0.0181  # x magnet 1   |
    x[5] = -0.0183  # x magnet 2  |
    x[6] = 0.018  # y magnet 3    |
    x[7] = -0.0181  # y magnet 4  |
    # ------------------------------------------------
    x[8] = 0.017  # x sensor1     | sensor positions
    x[9] = -0.0135  # y sensor1   |
    x[10] = 0.0162  # z sensor1   |
    x[11] = 0.0135  # x sensor2   |
    x[12] = 0.0171  # y sensor2   |
    x[13] = 0.0162  # z sensor2   |
    # ------------------------------------------------
    x[14] = 357  # phi magnet 1   | magnet orientations
    x[15] = 181  # phi magnet 2   |
    x[16] = 284  # phi magnet 3   |
    x[17] = 270  # phi magnet 4   |
    # ------------------------------------------------
    x[18] = 2  # index magnet 1   | magnetization indices
    x[19] = 3  # index magnet 2   | converted to direction for 1 and 2
    # ------------------------------------------------
    x[20] = 0.0  # phi sensor 1   | sensor orientations
    x[21] = 0.0  # theta sensor 1 |
    # ------------------------------------------------
    x[22] = 4.0  # s              | tilt angles
    x[23] = 4.0  # n              |
    x[24] = 4.0  # e              |
    x[25] = 4.0  # w              |
    # ------------------------------------------------
    x[26] = 5e-3  #               | magnet dimensions
    # ------------------------------------------------

    if calibration is not None:
        x[8] += calibration[0]
        x[9] += calibration[1]
        x[10] += calibration[2]
        x[20] += calibration[3]
        x[21] += calibration[4]
        x[22] += calibration[5]
        x[23] += calibration[6]
        x[24] += calibration[7]
        x[25] += calibration[8]

    if generator is not None:
        # add uncertainty to positions
        position_uncertainty_scale = 1e-4  # 0.1mm

        # magnet positions
        x[:8] += generator.normal(
            loc=0,
            scale=position_uncertainty_scale,
            size=(8,),
        )

        # sensor positions
        x[8:14] += generator.normal(
            loc=0,
            scale=position_uncertainty_scale,
            size=(6,),
        )

        angle_uncertainty_scale = 0.1  # 1deg

        # magnet angles
        x[14:18] += generator.normal(
            loc=0,
            scale=angle_uncertainty_scale,
            size=(4,),
        )

        x[20:22] += generator.normal(
            loc=0,
            scale=angle_uncertainty_scale,
            size=(2,),
        )

    return x


def magnetization_values() -> np.ndarray:
    """
    Measured magnetization values of the 4 magnets
    """
    return np.asarray((1.2124, 1.204, 1.208, 1.196))


def calibration_values() -> np.ndarray:
    """
    Calibration offsets calculated by Peter for the real joystick.
    """
    x = np.zeros((9,))
    # ------------------------------------------------
    x[0] = -3.07410451e-05  # dx     | sensor position
    x[1] = 6.42540894e-04  # dy      |
    x[2] = 4.97023923e-04  # dz      |
    # ------------------------------------------------
    x[3] = 2.99978359e00  #  dphi    | sensor orientation
    x[4] = 2.87211782e00  # dtheta   |
    # ------------------------------------------------
    x[5] = -9.97912137e-01  # s      | tilt angles
    x[6] = -9.94383444e-01  # n      |
    x[7] = -5.00236025e-01  # e      |
    x[8] = -9.89713586e-01  # w      |
    # ------------------------------------------------
    return x
