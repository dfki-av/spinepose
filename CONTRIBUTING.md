# Contributing

Thank you for helping improve SpinePose. This project is published on PyPI and
used as a production-facing open-source package, so changes should preserve the
public API unless the change is explicitly planned as a breaking release.

## Development Setup

Use Python 3.9-3.12 for development. The package may import on Python versions
outside that range in some environments, but the supported release target is the
range declared in `setup.cfg`.

```bash
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Linux or Windows, install the GPU runtime during development with:

```bash
python -m pip install -e ".[gpu,dev]"
```

On macOS, the default `onnxruntime` package is used for both CPU and available
Apple acceleration paths, so the `gpu` extra does not install
`onnxruntime-gpu`.

## Pre-commit Hooks

Install the pre-commit hook after installing the `dev` extra:

```bash
pre-commit install
```

The hook runs Ruff before each commit:

```bash
pre-commit run --all-files
```

The Ruff configuration in `pyproject.toml` is intentionally conservative. It
checks syntax-level pycodestyle errors and Pyflakes issues without forcing a
large style-only cleanup across the existing codebase.

## Running Tests

Run the full unit test suite with:

```bash
python -m unittest discover -s tests
```

The tests are written with the standard-library `unittest` framework. Keep new
tests deterministic and avoid network or model-download side effects. Mock ONNX
Runtime sessions, OpenCV I/O, and model downloads when testing code paths that
would otherwise require external files or hardware-specific providers.

## Linting

Run Ruff manually with:

```bash
ruff check .
```

Fix lint failures before opening a pull request. If you need to broaden the Ruff
rule set, do that in a separate cleanup-oriented change so functional patches do
not become hard to review.

## Code Guidelines

- Keep public APIs stable. If an argument or behavior needs to change, provide a
  deprecation path when practical.
- Prefer small, focused changes with tests that cover the behavior being changed.
- Avoid downloading models, opening GUI windows, or requiring GPU hardware in
  unit tests.
- Preserve existing package conventions for model wrappers, preprocessing,
  postprocessing, and inference helpers.
- Use concise Google-style docstrings for public classes and functions.
- Keep release-facing documentation in sync when changing CLI flags, Python API
  arguments, model defaults, or packaging behavior.

## Dependency Guidelines

Runtime dependencies belong in `install_requires`.

Optional dependency groups belong in `[options.extras_require]`:

- `gpu`: Linux/Windows GPU runtime support through `onnxruntime-gpu`.
- `dev`: developer tools such as Ruff, pre-commit, build, and twine.

Do not add heavyweight tooling or test-only packages to `install_requires`.
SpinePose should remain lightweight for users who only install it for inference.

## Changelog

Update `CHANGELOG.md` for user-visible changes. Keep new development work under
`[Unreleased]` until a release is tagged.

When preparing a release, move the relevant unreleased entries into a new version
section with the release date. Do not add unreleased branch work directly to
past release sections.

## Release Workflow

Typical release flow:

1. Develop changes on `dev` or a feature branch.
2. Keep tests and Ruff passing.
3. Update `CHANGELOG.md` under `[Unreleased]`.
4. When ready to release, merge into `main`.
5. Bump `src/spinepose/VERSION`.
6. Move changelog entries from `[Unreleased]` into the new version section.
7. Build the package:

```bash
python -m build
```

8. Inspect the generated distributions and upload through the project’s normal
   PyPI release process.
9. Create a GitHub release tag matching the version, for example `v2.0.3`.

The runtime version is read from `src/spinepose/VERSION` by
`src/spinepose/_version.py`, so keep that file as the single version source.

## Pull Request Checklist

Before submitting a pull request:

- Run `python -m unittest discover -s tests`.
- Run `ruff check .` or `pre-commit run --all-files`.
- Update tests for behavioral changes.
- Update `README.md` for user-facing CLI or API changes.
- Update `CHANGELOG.md` for user-visible changes.
- Confirm packaging changes do not force users to install mutually exclusive
  ONNX Runtime packages.

