---
layout: "contents"
title: Release Notes
---

# Release Notes

## Next Release

*Released on TBD*

- TODO

## 0.6.0

*Released on Apr 9 2024*

- fixed some awkward behaviour in `Driving-v1` and `DrivingGen-v1` shortest path policies.
- made `info` dictionary typing less strict. It can now contain no entries, or entries for keys other than agent IDs.
- fixed bug in `sample_initial_agent_state` function in `PursuitEvasion-v1` environment
- fixed docstring formatting across entire package
- updated documentation

## 0.5.1

*Released on Nov 22 2023*

Minor release to fix a bug with `setup.py` and the newer agent models being saved as cuda tensors.

---

## 0.5.0

*Released on Nov 19 2023*

This release adds some more environments and agents, fixes some bugs, and tweaks some of the existing environments.

Major changes include:

- Added release notes section to the docs
- Added the `AgentEnvWrapper` class which can be used to incorporate a `posggym.agents` policy as part of an environment
- Added the `StackEnv` wrapper for converting a `posggym.Env` into accepting and outputting stacked arrays as opposed to dictionaries.
- Added the `CooperativeReaching-v0` grid-world environment along with heuristic policies
- Updated the `posggym.agents.Policy` API to make it a bit less confusing
- Updated the `Driving` and `DrivingGen` envs to `v1` which includes a few bug fixes and adds a number of improvements (`Driving-v0` is no longer supported, including the `Driving-v0` agent policies.)
- Added shortest path based policies for all `Driving-v1` and `DrivingGen-v1` environments
- Updated RL policies for `Driving-v1`
- Updated `LevelBasedForaging` environment to `v3` which includes a number of small improvements, mainly around removing unused parameters (`LevelBasedForaging-v2` is no longer supported, including the `LevelBasedForaging-v2` agent policies.)
- Updated heuristic policies for `LevelBasedForaging-v2` grid-world environment and added some RL policies for two scenarios
- Updated `PursuitEvasion` environment to `v1` which removes unused parameters (`PursuitEvasion-v0` is no longer supported, including the `PursuitEvasion-v0` agent policies.)
- Updated agents for `PursuitEvasion-v1`
- Update agents for `PredatorPrey-v0` including adding some new RL policies and heuristic policies
- Tested agent diversity for most of the grid world environments
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
