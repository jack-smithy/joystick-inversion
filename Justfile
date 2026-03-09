run:
    uvx ty check && uv run joystick-inversion

type:
    uvx ty check

lint:
    uvx ruff check

fix:
    uvx ruff check --fix