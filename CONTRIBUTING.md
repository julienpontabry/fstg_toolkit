# Contributing to fSTG Toolkit

Thanks for your interest in fSTG Toolkit! Bug reports, feature ideas, documentation fixes and
code contributions are all welcome.

Everyone taking part in this project is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug** you ran into while using the library, the CLI or the dashboard.
- **Suggest a feature** or an improvement to an existing one.
- **Improve the documentation**: tutorials, docstrings, or the user guide.
- **Contribute code**: bug fixes, new metrics, new figures, new CLI options.

You do not need to be a Python expert to help: reporting a confusing error message or an unclear
paragraph of the documentation is a real contribution.

## Getting help

If you have a question about *using* the toolkit rather than a bug to report:

1. Check the documentation at <https://fstg-toolkit.readthedocs.io/>.
2. Check the built-in help: `python -m fstg_toolkit --help`, or `--help` on any subcommand
   (for instance `python -m fstg_toolkit graph build --help`).
3. Ask in [GitHub Discussions](https://github.com/julienpontabry/fstg_toolkit/discussions),
   in the **Q&A** category.

Please use Discussions rather than the issue tracker for usage questions: it keeps the answers
searchable for the next person with the same question.

## Reporting a bug

Open an issue with the [bug report form](https://github.com/julienpontabry/fstg_toolkit/issues/new/choose).
Before that, please search the [existing issues](https://github.com/julienpontabry/fstg_toolkit/issues)
in case the problem is already known.

A useful report contains:

- the version of the toolkit, obtained with `python -m fstg_toolkit --version`;
- your Python version and operating system;
- the optional extras you installed (`dashboard`, `plot`, `frequent`, or none);
- a **minimal reproducible example** — the shortest script or command that triggers the problem,
  with simulated data if your own data cannot be shared
  (see `python -m fstg_toolkit graph simulate --help`);
- what you expected to happen, what happened instead, and the full traceback if there is one.

## Development setup

Create an environment with the required Python and Poetry binaries. Using conda:

```shell
conda env create -n <env_name> -f environment.yml
conda activate <env_name>
```

Then, from the project root, install all the dependencies including the optional ones and the
development tools:

```shell
poetry install --all-extras
export PYTHONPATH="$PYTHONPATH:src"
```

The `frequent` extra additionally requires a working Docker installation, since frequent subgraph
pattern mining runs in a container.

## Running the tests

There is no linter configured for this project: the test suites are the quality gate. Both must
pass before a pull request can be merged.

```shell
# Unit tests
python -m unittest discover -s test -p "test_*.py"

# Doctests (the Examples sections of the docstrings)
python test/doctests_runner.py
```

You can run a single test file, class or method while working:

```shell
python -m unittest test.test_io
python -m unittest test.test_io.DataLoaderTestCase
python -m unittest test.test_io.DataLoaderTestCase.test_load_areas
```

And measure coverage:

```shell
coverage run -m unittest discover -s test -p "test_*.py" && coverage report
```

New behavior should come with a test. Tests live in `test/test_*.py` as `unittest.TestCase`
subclasses named `*TestCase`, using `setUp`/`tearDown` for fixtures. The Dash application layer
(`src/fstg_toolkit/app/`) is not covered by unit tests at the moment; changes there are checked
manually by running the dashboard:

```shell
python -m fstg_toolkit dashboard show my_graphs.zip
```

## Code style

Follow the conventions already used in `src/fstg_toolkit/`. Reading a neighboring module is the
fastest way to get them right.

- **Imports** grouped as standard library → third-party → local, separated by a blank line.
- **Naming**: `snake_case` for functions and variables, `PascalCase` for classes,
  `SCREAMING_SNAKE_CASE` for constants, a `__double_underscore` prefix for module-level private
  functions.
- **Type hints** on every function signature; use `Any` when the type is genuinely uncertain.
- **Docstrings** in NumPy/SciPy style, with `Parameters`, `Returns` and `Examples` sections. The
  code in `Examples` is executed by `test/doctests_runner.py`, so it must actually run.
- **Enums** are always decorated with `@unique` and use `auto()` values.

## License header

The toolkit is distributed under the [CeCILL Free Software Agreement v2.1](LICENSE). Every source
file carries the corresponding license header; copy it from the top of any existing file, for
instance `src/fstg_toolkit/graph.py`, and add it to any file you create.

By submitting a contribution, you agree that it is distributed under the CeCILL-2.1 license and
that you have the right to contribute it.

## Bundled third-party code

The sources under `src/fstg_toolkit/spminer/` are bundled from external projects and are **not**
covered by the CeCILL-2.1 license of the toolkit. Please do not modify them in a pull request; see
`src/fstg_toolkit/spminer/NOTICE.md` for their provenance and status. Changes to the way the
toolkit *calls* that code belong in `src/fstg_toolkit/frequent/` instead.

## Commit messages

Commits follow a conventional style: a lowercase type, a colon, then a short imperative sentence.

```
feat: add temporal betweenness metric.
fix: wrong path to spminer.
doc: clarify the temporal transitions algebra.
build: update the lock file.
```

Types used in this repository: `feat`, `fix`, `doc`, `build`, `license`, `paper`, `git`.

## Pull requests

1. Branch from `main`, and keep one topic per pull request.
2. Fill in the pull request template.
3. Make sure both test suites pass locally.
4. Add an entry to `docs/changelog.md` for anything a user would notice.
5. Push and open the pull request. The CI (`.github/workflows/ci.yml`) runs the unit tests and the
   doctests on Ubuntu and macOS with Python 3.12 and 3.13; it must be green before a merge.

Reviews are friendly and focused on the code. If a discussion stalls or you are unsure how to
proceed, say so in the pull request — it is always fine to ask.

## Credit

Contributions are credited in `docs/changelog.md`, and substantial ones are acknowledged in
`CITATION.cff`. Note that the author list of the software and the author list of the accompanying
article are distinct: contributing code does not automatically mean authorship of the paper.
