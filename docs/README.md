# POSGGym-docs

This folder contains the documentation for POSGGym.

## Build the Documentation

Assuming you have downloaded/cloned the repo and installed POSGGym. First install the required packages for building the documentation:

```bash
cd docs
pip install -r requirements.txt
```

Then build the documentation once:

```bash
cd docs
make dirhtml
```

To rebuild the documentation automatically every time a change is made:

```bash
cd docs
sphinx-autobuild -b dirhtml . _build
```

You should now be able to view the document by going to `http://127.0.0.1:8000` in your browser (or the address displayed in the stdout of the `sphinx-autobuild` command).

## Adding docs for new environment

Firstly ensure the docstrings for the environment are up-to-date and contains all the relevant information. If updating an existing environment make sure to update the version history in the docstring.

Then generate a GIF for the environment:

```bash
cd docs
python scripts/gen_gifs.py --env-id YourEnvID-v0
```

Next generate markdown file:

```bash
python scripts/gen_env_mds.py
```

Next update the relevant environmtn docs file in `docs/environments`, making sure to add the new environment `.md` file to the `{toctree}` and also adding the gif.

## Adding docs for new agents


Generate a markdown file for the agent:

```bash
cd docs
python scripts/gen_agent_mds.py
```
