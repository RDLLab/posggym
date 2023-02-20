# posggym testing

## Running the tests

```
$ pytest
```

To run in parallel:

1. Install the `pytest-xdist` package (i.e. with `pip install pytest-xdist`)
2. Run `pytest -n <num>` where `<num>` is the number of processes to run in parallel.
