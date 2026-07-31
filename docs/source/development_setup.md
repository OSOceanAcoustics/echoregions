# Development setup

Thank you for your interest in contributing to **Echoregions**! This section describes how to set up a development environment, run tests, and build the documentation locally.

## Clone the repository

Fork the Echoregions repository, then clone your fork:

```console
# Clone your fork
git clone https://github.com/YOUR_GITHUB_USERNAME/echoregions.git

# Enter the repository
cd echoregions

# Add the upstream repository
git remote add upstream https://github.com/echostack-org/echoregions.git
```

## Development environment

Echoregions development can be done using either **Conda** or **uv**.

::::{tab-set}

:::{tab-item} Conda

Create and activate a development environment:

```console
conda create -c conda-forge -n echoregions-dev --yes python=3.12
conda activate echoregions-dev
```

Upgrade pip:

```console
python -m pip install --upgrade pip
```

Install Jupyter/IPython development support:

```console
conda install -c conda-forge ipykernel
```

Install Echoregions in editable mode with development and testing dependencies:

```console
python -m pip install -e . --group dev --group test
```

:::

:::{tab-item} UV

Install `uv` by following the
[uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

From the Echoregions repository directory:

```console
uv sync
```

This creates a virtual environment and installs Echoregions with the development dependencies.

To activate the environment:

```console
source .venv/bin/activate
```

:::

::::

:::{tip}
If you use Conda and experience slow dependency solving, consider using
[Mamba](https://mamba.readthedocs.io/en/latest/) as a faster alternative.
You can install a minimal Conda environment using
[Miniforge](https://conda-forge.org/download/).
:::

# Testing

The Echoregions test suite uses `pytest`.

::::{tab-set}

:::{tab-item} Conda

Run:

```console
pytest -vv
```

:::

:::{tab-item} UV

Run:

```console
uv run pytest -vv
```

:::

::::

To run a specific test file:

```console
pytest -vv path/to/test_file.py
```

# Code formatting and pre-commit hooks

Echoregions uses
[pre-commit](https://pre-commit.com/) to run automated checks before commits.

Install the hooks:

::::{tab-set}

:::{tab-item} Conda

```console
pre-commit install
```

:::

:::{tab-item} UV

```console
uv run pre-commit install
```

:::

::::

Run all checks manually:

::::{tab-set}

:::{tab-item} Conda

```console
pre-commit run --all-files
```

:::

:::{tab-item} UV

```console
uv run pre-commit run --all-files
```

:::

::::

# Documentation development

Echoregions documentation is built using **Jupyter Book 2** and the MyST Document Engine.

Install documentation dependencies:

::::{tab-set}

:::{tab-item} Conda

```console
python -m pip install --group docs
```

:::

:::{tab-item} UV

```console
uv sync --group docs
```

:::

::::

Build the documentation locally:

::::{tab-set}

:::{tab-item} Conda

```console
jupyter-book build docs
```

:::

:::{tab-item} UV

```console
uv run jupyter-book build docs
```

:::

::::

Documentation layout:

- Documentation source files: `docs/source`
- MyST configuration: `docs/source/myst.yml`
- Dependency groups: `pyproject.toml`
- Sidebar structure: `docs/source/myst.yml`
