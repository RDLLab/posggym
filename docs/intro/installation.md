---
layout: "contents"
title: Installation
firstpage:
---

# Installation

The easiest way to install POSGGym is using ``pip``:

```
pip install posggym
```

## Common issues

### libGL error: failed to open iris

This error can occur when trying to render an environment, depending on your python setup. It isn't a problem with POSGGym, but rather with your system.

Here are some useful links for resolving the issue:

- <https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris>
- If your using conda: <https://stackoverflow.com/questions/71010343/cannot-load-swrast-and-iris-drivers-in-fedora-35/72200748#72200748>
