# POSGGym Contribution Guidelines

At this time we are currently accepting the current forms of contributions:

- Bug reports (keep in mind that changing environment behavior should be minimized as that requires releasing a new version of the environment and makes results hard to compare across versions)
- Pull requests for bug fixes
- Documentation improvements
- New environments

## Development

This section contains technical instructions & hints for the contributors.

### Installation

Clone the repo then you can install POSGGym locally using `pip`  by navigating to the `posggym` root directory (the one containing the `setup.py` file), and running:

```
pip install -e .
```

Or use the following to install `posggym` with all dependencies:

```
pip install -e .[all]
```

And the following to install dependencies for running tests:

```
pip install -e .[testing]
```


### Git hooks

The CI will run several checks on the new code pushed to the repository. These checks can also be run locally without waiting for the CI by following the steps below:

1. [install `pre-commit`](https://pre-commit.com/#install),
2. Install the Git hooks by running `pre-commit install`.

Once those two steps are done, the Git hooks will be run automatically at every new commit.
The Git hooks can also be run manually with `pre-commit run --all-files`, and if needed they can be skipped (not recommended) with `git commit --no-verify`.

**Note:** you may have to run `pre-commit run --all-files` manually a couple of times to make it pass when you commit, as each formatting tool will first format the code and fail the first time but should pass the second time.


### Type checking

This project uses [pyright](https://github.com/microsoft/pyright) for type checking. For instructions on installation see official [instructions](https://github.com/microsoft/pyright#command-line).
Once `pyright` is installed it can be run locally by running ``pyright --project pyproject.toml`` or using the pre-commit process ``pre-commit run --all-files`` from the root project directory. Alternatively, pyright is a built-in feature of VSCode that will automatically provide type hinting.


### Code style

For code style posggym-agents uses `black`. See the [black website](https://black.readthedocs.io/en/stable/) for install instructions. For everything else, posggym-agents uses [ruff](https://github.com/charliermarsh/ruff).


### Running tests

The project comes with a number of tests (see `tests` directory) using [pytest](https://docs.pytest.org/en/latest/getting-started.html#install-pytest). These will be run for the whole project during pull requests. They can also be run locally with `pytest` from the project root folder.


## Building the docs

Make sure that you have installed the requirements:

```shell
cd docs
pip install -r requirements.txt
```

Then run

```shell
python scripts/gen_mds.py
make dirhtml
```

Now, navigate to `_build/dirhtml` and open `index.html` in your browser.
