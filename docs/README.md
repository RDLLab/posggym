# POSGGym-docs

This folder contains the documentation for POSGGym.

## Build the Documentation

Assuming you have downloaded/cloned the repo and installed POSGGym. First install the required packages for building the documentation:

```
cd docs
pip install -r requirements.txt
```

Then build the documentation once:

```
cd docs
make dirhtml
```

To rebuild the documentation automatically every time a change is made:

```
cd docs
sphinx-autobuild -b dirhtml . _build
```

## Adding docs for new environment

Firstly generate a GIF for the environment:

```
cd docs
python scripts/gen_gifs.py --env_id YourEnvID-v0
```

Next generate markdown file:

```
python scripts/gen_mds.py
```

Next update the relevant environmtn docs file in `docs/environments`, making sure to add the new environment `.md` file to the `{toctree}` and also adding the gif.
