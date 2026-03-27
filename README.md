# joystick-inversion

## Installation

Clone the repository

```bash
git clone <thingy>
```

### With pip

Create an activate a virtual environment in the project root

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install using

```bash
pip install -e .
```

You can train and evaluate the model by running

```bash
python scripts/run.py
```

### With uv

I like to use the uv package manager for reproducible builds and dependency resolution, <https://docs.astral.sh/uv/>. If you have uv installed, in the cloned repo run

```bash
uv sync
```

or

```bash
uv sync --dev
```

for some extra tooling. I like to usually use `ruff` as a linter and `ty` as a type checker. You can use these with

```bash
uvx ruff check && uvx ty check
```
