# Justfile for ai-cr

MIN_PY_VER := "3.12"
MAIN_PY_VER := "3.14"
QA_ENV := 'UV_PROJECT_ENVIRONMENT=".venv_qa"'
TEST_ENV := 'UV_PROJECT_ENVIRONMENT=".venv_qa"'
BUILD_ENV := 'UV_PROJECT_ENVIRONMENT=".venv_qa"'
MAIN_ENV := 'UV_PROJECT_ENVIRONMENT=".venv"'

# Show available commands
list:
    @just --list

# Run all the formatting, linting, and testing commands
qa:
    {{QA_ENV}} uv run --python={{MAIN_PY_VER}} --extra dev ruff format . --check
    {{QA_ENV}} uv run --python={{MAIN_PY_VER}} --extra dev ruff check .
    {{QA_ENV}} uv run --python={{MAIN_PY_VER}} --extra dev ty check .
    {{QA_ENV}} uv run --python={{MAIN_PY_VER}} --extra dev basedpyright src tests
    {{QA_ENV}} uv run --python={{MAIN_PY_VER}} --extra dev mypy .

qa-fix:
    {{QA_ENV}} uv run --python={{MAIN_PY_VER}} --extra dev ruff format .
    {{QA_ENV}} uv run --python={{MAIN_PY_VER}} --extra dev ruff check . --fix

# Run all the tests for all the supported Python versions
testall:
    {{TEST_ENV}} uv run --python={{MIN_PY_VER}} --extra dev pytest
    {{TEST_ENV}} uv run --python={{MAIN_PY_VER}} --extra dev pytest

# Run all the tests, but allow for arguments to be passed
test *ARGS:
    @echo "Running with arg: {{ARGS}}"
    {{TEST_ENV}}  uv run --python={{MAIN_PY_VER}} --extra dev pytest {{ARGS}}

# Run all the tests, but on failure, drop into the debugger
pdb *ARGS:
    @echo "Running with arg: {{ARGS}}"
    {{TEST_ENV}} uv run --python={{MAIN_PY_VER}}  --extra dev pytest --pdb --maxfail=10 --pdbcls=IPython.terminal.debugger:TerminalPdb {{ARGS}}

# Run coverage, and build to HTML
coverage:
    {{TEST_ENV}} uv run --python={{MAIN_PY_VER}} --extra dev coverage run -m pytest .
    {{TEST_ENV}} uv run --python={{MAIN_PY_VER}} --extra dev coverage report -m
    {{TEST_ENV}} uv run --python={{MAIN_PY_VER}} --extra dev coverage html

# Build the project, useful for checking that packaging is correct
build:
    rm -rf build
    rm -rf dist
    {{BUILD_ENV}} uv build

setup_dev_venv:
    {{MAIN_ENV}} uv sync --python={{MAIN_PY_VER}} --extra dev

VERSION := `grep -m1 '^version' pyproject.toml | sed -E 's/version = "(.*)"/\1/'`

# Print the current version of the project
version:
    @echo "Current version is {{VERSION}}"

# Tag the current version in git and put to github
#tag:
#    echo "Tagging version v{{VERSION}}"
#    git tag -a v{{VERSION}} -m "Creating version v{{VERSION}}"
#    git push origin v{{VERSION}}

# remove all build, test, coverage and Python artifacts
clean:
	clean-build
	clean-pyc
	clean-test

# remove build artifacts
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

# remove Python file artifacts
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

# remove test and coverage artifacts
clean-test:
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache
