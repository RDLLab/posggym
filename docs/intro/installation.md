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

### Dependency error when using with rllib

You may run into some dependency issues when installing posggym and rllib. Specifically, posggym depends on ``gymnasium>=0.27``, while rllib (``>=2.3``) depends on ``gymnasium==0.26.3``.

The current best thing to do is:

```
pip install ray[rllib]>=2.3
pip install "gymnasium>=0.27"
```

This will leave you with rllib installed as well as ``gymnasium>=0.27``, and shouldn't cause any issues when running posggym or rllib, since ``gymnasium>=0.27`` is backwards compatible with ``gymnasium==0.26.3`` (to the best of my knowledge).

### libGL error: failed to open iris

This error can occur when trying to render an environment, depending on your python setup. It isn't a problem with POSGGym, but rather with your system.

Here are some useful links for resolving the issue:

- <https://stackoverflow.com/questions/72110384/libgl-error-mesa-loader-failed-to-open-iris>
- If your using conda: <https://stackoverflow.com/questions/71010343/cannot-load-swrast-and-iris-drivers-in-fedora-35/72200748#72200748>
