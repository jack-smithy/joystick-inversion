# Copilot-instructions

## Coding guidelines
- Always follow python best practices
- Code should be fully typed with type hints
- Follow machine learning best practices
- All results should be reproducible

## Command running
- The entry point to the program should be in the function `main` in `src/joystick-inversion/__init__.py`
- This can be run using the command `just run`. This runs the type checker over the codebase as well.
- Alternatively you can run an individual script using `uv run /path/to/file.py`.
- Packages can be installed using `uv add package-name`.