# torch-power-weighted-projection

[![License](https://img.shields.io/pypi/l/torch-power-weighted-projection.svg?color=green)](https://github.com/jojoelfe/torch-power-weighted-projection/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-power-weighted-projection.svg?color=green)](https://pypi.org/project/torch-power-weighted-projection)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-power-weighted-projection.svg?color=green)](https://python.org)
[![CI](https://github.com/jojoelfe/torch-power-weighted-projection/actions/workflows/ci.yml/badge.svg)](https://github.com/jojoelfe/torch-power-weighted-projection/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jojoelfe/torch-power-weighted-projection/branch/main/graph/badge.svg)](https://codecov.io/gh/jojoelfe/torch-power-weighted-projection)

Create power weighted projections of volumes

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork jojoelfe/torch-power-weighted-projection --clone
# or just
# gh repo clone jojoelfe/torch-power-weighted-projection
cd torch-power-weighted-projection
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
