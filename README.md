# wound-composite

A Python package for wound composite structures.

## Requirements

- [uv](https://docs.astral.sh/uv/), which manages Python, the virtual environment and
  dependencies
- Python 3.14. You don't need to install it yourself: uv reads `.python-version` and
  downloads it if needed.
- `make`, to run the development tasks in the [Makefile](Makefile). This is optional if
  you prefer to run the underlying `uv run ...` commands directly.

## Installing uv

### macOS and Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On macOS you can also use Homebrew: `brew install uv`.

### Windows (PowerShell)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

You can also use `winget install --id=astral-sh.uv -e`.

Restart your terminal afterwards so `uv` is on your `PATH`, then check it works:

```bash
uv --version
```

## Getting started

Clone the repository and install the project with its development tools:

```bash
git clone git@github.com:fenics-dolfiny/wound-composite.git
cd wound-composite
uv sync
```

`uv sync` creates a `.venv` folder, installs the Python version from `.python-version`,
and installs the exact dependency versions pinned in `uv.lock`. These commands are the
same on macOS, Linux and Windows.

Run the command-line entry point:

```bash
uv run wound-composite
```

Run any Python script or tool inside the project environment with `uv run`:

```bash
uv run python my_script.py
```

You don't need to activate the virtual environment when you use `uv run`. If you'd like
to activate it anyway:

| Platform              | Command                       |
| --------------------- | ----------------------------- |
| macOS / Linux         | `source .venv/bin/activate`   |
| Windows (PowerShell)  | `.venv\Scripts\Activate.ps1`  |
| Windows (cmd)         | `.venv\Scripts\activate.bat`  |

## Running `make lint` (required)

Run `make lint` before every commit and pull request. It must finish without errors.

```bash
make lint
```

It runs, in order:

1. `uv sync`: brings your environment in line with `uv.lock`
2. `ruff check --fix`: lints the code with every ruff rule enabled (`select = ["ALL"]`)
   and fixes what it can automatically
3. `ruff format`: formats the code (line length 91)
4. `mypy`: static type checking. All code needs type annotations.
5. `bandit`: scans `wound_composite/` for common security problems
6. `pip-audit`: checks the installed dependencies for known vulnerabilities

`make lint` changes your files (fixes and formatting), so review and commit those changes
too. To run the same checks without changing anything, as you would in CI, use:

```bash
make check
```

### `make` on Windows

Windows doesn't ship with `make`. Install it with one of:

```powershell
winget install ezwinports.make
choco install make
scoop install make
```

Or run the commands from the `lint` target in the [Makefile](Makefile) by hand:

```powershell
uv sync
uv run ruff check --fix .
uv run ruff format .
uv run mypy .
uv run bandit -r wound_composite/
uv run pip-audit --skip-editable
```

## Other Makefile targets

| Target          | What it does                                               |
| --------------- | ---------------------------------------------------------- |
| `make install`  | `uv sync`                                                  |
| `make lint`     | Fixes, formats and checks the code (see above)             |
| `make check`    | Same checks as `lint`, without changing files              |
| `make coverage` | Runs the tests in `tests/` with coverage, writes `htmlcov/` |
| `make gitleaks` | Scans the repository for committed secrets (needs [gitleaks](https://github.com/gitleaks/gitleaks)) |

## Managing dependencies

```bash
uv add numpy            # runtime dependency
uv add --dev pytest     # development-only dependency
uv lock --upgrade       # upgrade all locked versions
```

Always commit `pyproject.toml` and `uv.lock` together.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
