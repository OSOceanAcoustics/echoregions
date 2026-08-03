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

```console
# Create and activate a development environment:
conda create -c conda-forge -n echoregions-dev --yes python=3.12
conda activate echoregions-dev

# Install Jupyter/IPython development support:
conda install -c conda-forge ipykernel

# Install Echoregions in editable mode with development and testing dependencies:
python -m pip install -e . --group dev --group test
```

If you use Conda and experience slow dependency solving, consider using
[Mamba](https://mamba.readthedocs.io/en/latest/) as a faster alternative.
You can install a minimal Conda environment using
[Miniforge](https://conda-forge.org/download/).

:::

:::{tab-item} uv

Install `uv` by following the
[uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

Install Echoregions from the local repository directory:
```console
uv sync
```
This creates a virtual environment and installs Echoregions with the development dependencies.
:::

::::

## Testing

The Echoregions test suite uses `pytest`.

::::{tab-set}

:::{tab-item} Conda

```console
# Run `Lines` class tests
pytest -vv -m mark.lines

# Run `Regions2d` class tests
pytest -vv -m mark.regions2d
```
:::

:::{tab-item} uv
```console
# Run `Lines` class tests
uv run pytest -vv -m mark.lines

# Run `Regions2d` class tests
uv run pytest -vv -m mark.regions2d
```
:::

::::


## Code formatting with pre-commit hooks

Echoregions uses
[pre-commit](https://pre-commit.com/) to run automated checks before commits.

Install the hooks and run the checks on all modified files:
::::{tab-set}

:::{tab-item} Conda
```console
# Install hooks
pre-commit install

# Stage modified files
git add .

# Run checks on staged files
pre-commit run
```
:::

:::{tab-item} uv
```console
# Install hooks
uv run pre-commit install

# Stage modified files
git add .

# Run checks on staged files
uv run pre-commit run
```
:::

::::

## Documentation development

Echoregions documentation is built using **Jupyter Book 2** and the MyST Document Engine.

Install documentation dependencies:

::::{tab-set}

:::{tab-item} Conda
```console
# Install documentation dependencies
python -m pip install --group docs

# Build and run the documentation locally
jupyter book
```
:::

:::{tab-item} uv
```console
# Install documentation dependencies
uv sync --group docs

# Build and run the documentation locally
uv run jupyter book
```
:::

::::

Documentation layout:

- Documentation source files: `docs/source`
- MyST configuration: `docs/source/myst.yml`
- Dependency groups: `pyproject.toml`
- Sidebar structure: `docs/source/myst.yml`
- CSS style: `docs/source/styles.css`
