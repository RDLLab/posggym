# POSGGYm Contribution Guidelines

At this time we are currently accepting the current forms of contributions:

- Bug reports (keep in mind that changing environment behavior should be minimized as that requires releasing a new version of the environment and makes results hard to compare across versions)
- Pull requests for bug fixes
- Documentation improvements
- New environments

## Development

This section contains technical instructions & hints for the contributors.

### Type checking

This project uses `mypy` for type checking. For instructions on installation and running `mypy` locally see official [instruction](https://mypy.readthedocs.io/en/latest/getting_started.html#installing-and-running-mypy).

### Code style

For code style posggym uses `black`. See the [black website](https://black.readthedocs.io/en/stable/) for install instructions.

### Docstrings

For documentation this project uses `pydocstyle`.

### Running tests

The project comes with a number of tests using [pytest](https://docs.pytest.org/en/latest/getting-started.html#install-pytest). These can be run locally with `pytest` from the `posggym/tests` folder.

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
