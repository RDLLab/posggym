"""Contains list of all registered envs in posggym.

Reference:
- https://github.com/openai/gym/blob/master/tests/envs/spec_list.py
"""
from posggym import envs


spec_list = [
    spec
    for spec in sorted(envs.registry.all(), key=lambda x: x.id)
    if spec.entry_point is not None
]
