If you would like to contribute to wound-composite, please read the following guidelines.

## Coding Guidelines

1. Comment your code. Explain why the code does what it does, especially for the
   mechanics and the maths. Every module, class and public function needs a docstring.
2. Keep it organized. Break large functions into smaller pieces, and give functions and
   variables descriptive names. The linter enforces a maximum cyclomatic complexity of 6
   and at most 8 arguments per function.
3. Keep it clean. The code follows PEP 8 with a line length of 91 characters, enforced by
   `ruff format`. Type-annotate all code; `mypy` checks it.

## Keys to Getting Your Contributions Merged Quickly

1. First and foremost, **KEEP IT SIMPLE**. Break your work into small, self-explanatory
   pieces. Send large features as a series of smaller pull requests that each add a piece
   of functionality, rather than one large change. Small pull requests are much easier to
   review.
2. Run `make lint` and make sure it passes before you open a pull request (see
   [Linting](#linting)).
3. Test and debug your code, and add tests for new functionality.
4. Include everything your change needs in the pull request: code, tests, and updates to
   `pyproject.toml` and `uv.lock` when you change dependencies.
5. Follow the coding guidelines above. This makes review much faster.

## Development Setup

wound-composite uses [uv](https://docs.astral.sh/uv/) to manage the development
environment and locked dependencies. Install uv (see the [README](README.md#installing-uv)
for macOS, Linux and Windows instructions), then create the environment:

```bash
uv sync
```

When you change project dependencies, update them with `uv add` / `uv remove`, or edit
`pyproject.toml` and regenerate the lockfile:

```bash
uv lock
```

Commit `pyproject.toml` and `uv.lock` together.

## Testing

Put tests in the `tests/` folder and run them with coverage:

```bash
make coverage
```

## Linting

**`make lint` must pass before you commit or open a pull request.** Pull requests that
don't pass it won't be merged.

```bash
make lint
```

It syncs the environment, then runs:

- [ruff](https://docs.astral.sh/ruff/) to lint the code with every rule enabled and fix
  what it can, then to format it
- [mypy](https://mypy.readthedocs.io/) for static type checking
- [bandit](https://bandit.readthedocs.io/) to find common security problems
- [pip-audit](https://pypi.org/project/pip-audit/) to find dependencies with known
  vulnerabilities

The rules live in `pyproject.toml` under `[tool.ruff]`. `make lint` rewrites files, so
review and commit its changes too. To run the checks without changing any files, use
`make check`.

If ruff reports a rule you believe doesn't apply, fix the code where you can. If you must
suppress a rule, use a targeted `# noqa: <CODE>` on that line with the rule code, never a
bare `# noqa`.

On Windows, `make` isn't installed by default. See the
[README](README.md#make-on-windows) for how to install it or run the commands directly.

Before pushing, you can also scan for accidentally committed secrets with `make gitleaks`
(requires [gitleaks](https://github.com/gitleaks/gitleaks)).
