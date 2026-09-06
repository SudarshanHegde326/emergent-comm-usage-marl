"""SMACv2 - graph attention network (GAT, multi-hop).

This file contains NO training logic. It fixes --comm=gnn and delegates to
the single shared implementation in 14_mappo_smacv2_matched.py.

WHY A WRAPPER AND NOT A COPY
----------------------------
The original 09-12 trainers were four independent implementations and they
drifted apart: the baseline ended up with a different network layout and
different optimiser settings from the communicating conditions, so the
comparison measured architecture as well as communication. Keeping ONE
implementation makes that class of error impossible -- every condition runs
exactly the same code and only the communication module differs.

Every flag accepted by 14_mappo_smacv2_matched.py is accepted here, including
--num-heads, --num-layers, --mode epo, --eval-episodes and
--no-comm-to-memory.

    python scripts/trainers/12_gnncomm_mappo_smacv2.py --mode standard --seed 0
    python scripts/trainers/12_gnncomm_mappo_smacv2.py --mode epo      --seed 0
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SHARED = Path(__file__).with_name("14_mappo_smacv2_matched.py")
if not _SHARED.exists():
    raise FileNotFoundError(
        f"Shared implementation not found at {_SHARED}. "
        "14_mappo_smacv2_matched.py must sit in the same folder."
    )

_spec = importlib.util.spec_from_file_location("_smacv2_shared", _SHARED)
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)

COMM_MODE = "gnn"

if __name__ == "__main__":
    # Fix this file's communication mode and refuse any override, so a launcher
    # script cannot silently run the wrong condition.
    if "--comm" in sys.argv:
        raise SystemExit(
            f"{Path(__file__).name} is fixed to --comm {COMM_MODE}. "
            "Use 14_mappo_smacv2_matched.py directly to choose a mode."
        )
    args = _shared.build_args(sys.argv[1:] + ["--comm", COMM_MODE])
    print(f"[{Path(__file__).stem}] shared implementation, "
          f"--comm {args.comm} --mode {args.mode} --seed {args.seed}", flush=True)
    _shared.train(args)
