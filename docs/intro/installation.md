---
layout: "contents"
title: Installation
firstpage:
---

# Installation

POSGGym supports and test for `Python>=3.8`. We recommend using a virtual environment to install POSGGym (e.g. [conda](https://docs.conda.io/projects/conda/en/latest/index.html), [venv](https://docs.python.org/3/library/venv.html)).

## Using pip

The latest release version of POSGGym can be installed using `pip` by running:

```bash
pip install posggym
```

This will install the base dependencies for running all the environments and download the agent models (so may take a few minutes). In order to minimise the number of unused dependencies installed the default install does not include dependencies for running many posggym agents (specifically PyTorch).

You can install dependencies for POSGGym agents using `pip install posggym[agents]` or to install dependencies for all environments and agents use `pip install posggym[all]`.

## Installing from source

To install POSGGym from source, first clone the [repository](https://github.com/RDLLab/posggym) then run:

```bash
cd posggym
pip install -e .
```

This will install the base dependencies and download the agent models (so may take a few minutes). You can optionally install extras as described above. E.g. to install all dependencies for all environments and agents use:

```bash
pip install -e .[all]
```

To run tests, install the test dependencies and then run the tests:

```bash
pip install -e .[test]
pytest
```

Or alternatively you can run one of the examples from the `examples` directory:

```bash
python examples/run_random_agent.py python run_random_agents.py --env_id Driving-v0 --num_episodes 10 --render_mode human
```

## Common issues

### Dependency error when using with rllib

You may run into some dependency issues when installing posggym and rllib. Specifically, posggym depends on ``gymnasium>=0.27``, while rllib (``>=2.3``) depends on ``gymnasium==0.26.3``.

The current best thing to do is:

```
pip install ray[rllib]>=2.3
pip install "gymnasium>=0.27"
```

This will leave you with rllib installed as well as ``gymnasium>=0.27``, and shouldn't cause any issues when running posggym or rllib, since ``gymnasium>=0.27`` is backwards compatible with ``gymnasium==0.26.3`` (to the best of our knowledge).

### libGL error: failed to open iris

This error can occur when trying to render an environment, depending on your OS and python setup. It isn't a problem with POSGGym, but rather with your system.

Here are some useful links for resolving the issue:

- <https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris>
- If your using conda: <https://stackoverflow.com/questions/71010343/cannot-load-swrast-and-iris-drivers-in-fedora-35/72200748#72200748>
