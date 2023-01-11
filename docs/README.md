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
make dirhtml _build
```

To rebuild the documentation automatically every time a change is made:

```
cd docs
sphinx-autobuild -b dirhtml . _build
```
