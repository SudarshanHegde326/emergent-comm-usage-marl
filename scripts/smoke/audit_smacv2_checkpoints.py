
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Fields the current trainer writes into final.pt (see the save block at the
# end of train() in 14_mappo_smacv2_matched.py).
FINAL_REQUIRED = [
    "policy",
    "critic",
    "args",
    "steps",
    "final_test_win_rate",
    "final_train_win_rate",
]

# Extra fields present only in the full resumable checkpoint latest.pt.
LATEST_REQUIRED = [
    "policy", "critic", "opt_a", "opt_c",
    "obs_norm", "state_norm", "steps",
    "torch_rng", "numpy_rng",
]

# Scalar metric fields to range-check in final.pt (win rates must be in [0,1]).
FINAL_SCALARS_IN_UNIT_RANGE = ["final_test_win_rate", "final_train_win_rate"]


def audit_one(ckpt_path: Path, required: list[str],
              unit_range_fields: list[str]) -> tuple[bool, list[str]]:
    """Audit a single checkpoint file. Returns (is_ok, messages)."""
    msgs: list[str] = []
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        return False, [f"LOAD-ERROR: {e}"]

    if not isinstance(ckpt, dict):
        return False, [f"WRONG-TYPE: top-level is {type(ckpt).__name__}, expected dict"]

    missing = [f for f in required if f not in ckpt]
    if missing:
        msgs.append(f"MISSING-FIELDS: {missing}")

    for f in unit_range_fields:
        if f in ckpt and ckpt[f] is not None:
            try:
                v = float(ckpt[f])
            except (TypeError, ValueError):
                msgs.append(f"WRONG-TYPE: {f} is {type(ckpt[f]).__name__}")
                continue
            if v == v and not (0.0 <= v <= 1.0):
                msgs.append(f"OUT-OF-RANGE: {f}={v} not in [0,1]")

    if "steps" in ckpt:
        try:
            if int(ckpt["steps"]) < 0:
                msgs.append(f"OUT-OF-RANGE: steps={ckpt['steps']} < 0")
        except (TypeError, ValueError):
            msgs.append(f"WRONG-TYPE: steps is {type(ckpt['steps']).__name__}")

    is_ok = not any(
        m.startswith(("LOAD-ERROR", "WRONG-TYPE", "MISSING-FIELDS", "OUT-OF-RANGE"))
        for m in msgs
    )
    return is_ok, msgs


def find_experiments_dir(cli_value: str | None) -> Path | None:
    if cli_value is not None:
        return Path(cli_value)
    here = Path(__file__).resolve()
    for root in [here.parent.parent.parent, here.parent.parent, here.parent.parent.parent.parent]:
        if (root / "experiments").exists():
            return root / "experiments"
        if (root / "scripts" / "trainers" / "14_mappo_smacv2_matched.py").exists():
            return root / "experiments"
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", default=None,
                        help="Path to experiments/ dir. Auto-detected if omitted.")
    parser.add_argument("--check-latest", action="store_true",
                        help="also audit the resumable latest.pt files")
    args = parser.parse_args()

    experiments_dir = find_experiments_dir(args.experiments_dir)
    if experiments_dir is None:
        print("ERROR: cannot locate experiments/ dir. Pass --experiments-dir.",
              file=sys.stderr)
        return 1

    if not experiments_dir.exists():
        print(f"[audit] experiments dir does not exist yet: {experiments_dir}")
        print("[audit] nothing to audit (no runs have produced output).")
        return 0

    cond_dirs = sorted(d for d in experiments_dir.glob("smacv2_*") if d.is_dir())

    print(f"[audit] experiments_dir = {experiments_dir}")
    print(f"[audit] found {len(cond_dirs)} experiment dir(s)")
    print(f"[audit] required (final.pt) = {FINAL_REQUIRED}")
    if args.check_latest:
        print(f"[audit] required (latest.pt) = {LATEST_REQUIRED}")
    print()

    total = ok = 0
    bad: list[tuple[Path, list[str]]] = []

    def run_checks(pattern: str, required: list[str], scalars: list[str]) -> None:
        nonlocal total, ok
        for cond_dir in cond_dirs:
            for ckpt_path in sorted(cond_dir.glob(pattern)):
                total += 1
                is_ok, msgs = audit_one(ckpt_path, required, scalars)
                tag = "OK " if is_ok else "BAD"
                extra = "  " + "; ".join(msgs) if msgs else ""
                rel = ckpt_path.relative_to(experiments_dir)
                print(f"[audit] {tag} {str(rel):<48s}{extra}")
                if is_ok:
                    ok += 1
                else:
                    bad.append((ckpt_path, msgs))

    run_checks("seed*/final.pt", FINAL_REQUIRED, FINAL_SCALARS_IN_UNIT_RANGE)
    if args.check_latest:
        run_checks("seed*/latest.pt", LATEST_REQUIRED, [])

    print()
    print("=" * 60)
    print(f"[audit] total checked: {total}")
    print(f"[audit] OK:            {ok}")
    print(f"[audit] BAD:           {len(bad)}")
    print("=" * 60)

    if bad:
        print("\nFAILED checkpoints:")
        for path, msgs in bad:
            print(f"  {path}")
            for m in msgs:
                print(f"    - {m}")
        return 1

    if total == 0:
        print("[audit] No checkpoints found yet — nothing audited.")
        return 0

    print(f"[audit] All {total} checkpoint(s) pass the field-completeness check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
