"""RETIRED - do not use. Kept only so that stale imports fail loudly.

WHY THIS FILE WAS RETIRED
-------------------------
The previous version of this module defined the EPO condition as

    mode="epo"  ->  conic_fov=True

That is wrong, and it is wrong in a way that would have quietly invalidated
the entire SMACv2 half of the dissertation.

The Extended Partial Observability challenge is defined by the benchmark's
own configuration files. Diffing the two official configs shipped with
SMACv2 --

    smacv2/examples/configs/sc2_gen_protoss.yaml       (standard)
    smacv2/examples/configs/sc2_gen_protoss_epo.yaml   (EPO)

-- the ENTIRE difference is two lines:

    prob_obs_enemy:  1.0   ->  0.0
    action_mask:     True  ->  False

and conic_fov is False in BOTH files. The conic field of view is a separate,
unrelated feature. It changes the observation and action spaces (adding FOV
rotation actions) but it does NOT create the meaningful partial observability
that EPO is about.

Concretely, the retired file's "EPO" mode ran with:
  * prob_obs_enemy left at its default of 1.0 -- every ally shares every
    enemy sighting, the exact OPPOSITE of EPO;
  * action_mask left at its default of True -- the available-action vector
    still tells each agent which enemies it can hit, which is precisely the
    free information the benchmark authors removed in order to force agents
    to communicate.

In other words it was the setting in which communication is least likely to
matter, labelled as the setting designed to require it.

WHAT TO USE INSTEAD
-------------------
    from scripts.trainers import ...   # not importable by number; use:
    make_env(mode, seed, n_units, n_enemies, map_name)

defined in scripts/trainers/14_mappo_smacv2_matched.py, which mirrors both
official YAML files exactly and is covered by
tests/test_smacv2_trainer.py::test_epo_config_matches_official_yaml.
"""

_MESSAGE = (
    "src/envs/smacv2_env.py is retired: its mode='epo' set conic_fov=True, "
    "which is NOT the Extended Partial Observability challenge. EPO is "
    "prob_obs_enemy=0.0 together with action_mask=False (conic_fov stays "
    "False). Use make_env() in scripts/trainers/14_mappo_smacv2_matched.py. "
    "See the docstring at the top of this file for the full explanation."
)


def __getattr__(name):
    raise ImportError(f"{_MESSAGE}\n(attempted to import '{name}')")


raise ImportError(_MESSAGE)
