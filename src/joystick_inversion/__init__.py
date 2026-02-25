from joystick_inversion.system import make_dataset


def main() -> None:
    df = make_dataset(2)
    print(df)
