---
layout: "contents"
title: Release Notes
---

# Release Notes

## 0.5.0 (currently in development)

*Release date TBD*

This release adds some more environments and agents, fixes some bugs, and tweaks some of the existing environments.

Major changes include:

- Added the `CooperativeReaching-v0` grid-world environment along with heuristic policies
- Added release notes section to the docs
- Updated and added more heuristic policies for `LevelBasedForaging-v2` grid-world environment
- Added the `AgentEnvWrapper` class which can be used to incorporate a `posggym.agents` policy as part of an environment
- Added the `StackEnv` wrapper for converting a `posggym.Env` into accepting and outputting stacked arrays as opposed to dictionaries.
- Cleaned up docstrings of a bunch of classes

---

## 0.4.0

*Released on Aug 14 2023*

This release is the first release of the full POSGGym (environments + agents).

Major changes include:

- Integration of the `posggym.agents` library (migrated from separate repo/library), with agents provided for all continuous environments, and 4/7 grid-world environments
- Addition of four continuous environments
- Major updates to documentation
- Improved support for installation via pip
- Many other improvements

---

## 0.3.2

*Released on Mar 13 2023*

- lowered gymnasium dependency version to >=0.26 so it's compatible with latest rllib version

## 0.3.1

*Released on Mar 13 2023*

Skipping v0.3.0 due to typo.

- Significant additions to the documentation
- Changed registered environments to have only a single default variation of each environment and then users can pass arguments to `posggym.make` to change the parameters of the environment
- updated wrappers including record video, rllib and pettingzoo wrappers
- updated all grid world environments to use pygame rendering
- updated keyboard agent to support 1D continuous actions
- migrated to ruff for linting
- made all environments `observation_first` (removed `observation_first` attribute from envs and models)
- added pre-commit support
- some other improvements, bug fixes, and tests


---

## 0.2.1

*Released on Jan 13 2023*

*Patch update*

- Updated install instructions to reflect availability of posggym on pypi

## 0.2.0

*Released on Jan 13 2023*

**Major update**

- Updated environment and model APIs to be more inline with Gymnasium and PettingZoo
- Adding extensive testing
- Updated all environments
- Added documentation

---

## 0.1.0

*Released on Jan 12 2023*

First release with classic, grid world, and lbf environment.

This is the release used for the BA-POSGMCP paper.

As of the time of the release POSGGym has been updated and this version is deprecated.
