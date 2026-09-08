# Emergent communication usage in cooperative MARL

Code for an MSc dissertation (Manchester Metropolitan University, 2026).  
The question is not only whether a communicating team scores higher than a silent team. It is whether the trained policy still needs the messages.

Four modules are compared **inside each domain**, not as one merged experiment:

- no communication  
- fully-connected pooling  
- multi-head attention  
- graph attention over **living** agents (fully connected among living teammates; dead units are masked)

Official scores are mean return on Simple Spread and greedy test win rate on SMACv2. Usage is checked with message ablation and MINE. Channel probes (noise, bandwidth, information bottleneck) are extra runs. They are not a second ranking.

Live repository (same commit as the Moodle ZIP):  
https://github.com/SudarshanHegde326/emergent-comm-usage-marl

Ethics: MMU EthOS **91944**. Only public simulators. No human participants.

---

## What this repo is, and is not

**Is:** trainers, communication modules, channel objects, analysis scripts, unit tests, and (if present) `.pt` checkpoints from the reported seeds.

**Is not:** a physical robot stack, a targeting system, or a BenchMARL training run. BenchMARL / TorchRL appear in notes and older pins. The numbers in Chapter 5 come from the custom MAPPO trainers under `scripts/trainers/`.

Checkpoints under `experiments/` are toy-simulator weights. SMACv2 files play against the scripted built-in opponent only.

---

## Layout

```
src/                  environments, comm modules, channels
scripts/trainers/     official training entry points
scripts/analysis/     ablation and MINE
scripts/smoke/        SMACv2 EPO flag check
tests/                pytest
experiments/          checkpoints (if you cloned a commit that includes .pt)
```

Main trainers:

| Domain | Script |
|---|---|
| Simple Spread, no-comm | `scripts/trainers/01_nocomm_mappo_simple_spread.py` |
| Simple Spread, FC | `scripts/trainers/02_fccomm_mappo_simple_spread.py` |
| Simple Spread, attention / GNN | matching `*attn*` / `*gnn*` trainers in the same folder |
| SMACv2 (Standard and EPO) | `scripts/trainers/15_mappo_smacv2_parallel.py` |
| Shared SMACv2 policy / env flags | `scripts/trainers/14_mappo_smacv2_matched.py` |

Do not import `src/envs/smacv2_env.py` for new runs. That module was retired so an old EPO definition cannot come back by accident.

---

## Two environments (do not mix them)

Chapter 4 pins two stacks. Use one conda env per domain.

### A. Simple Spread (laptop is enough)

Python 3.11. Packages that were used on the dissertation machine:

- PyTorch (CPU wheel is fine)  
- PettingZoo, mpe2, Gymnasium  
- NumPy, wandb (optional)

```bash
conda create -n py311 python=3.11 -y
conda activate py311
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install pettingzoo mpe2 gymnasium numpy wandb matplotlib pandas seaborn pytest
```

From the repo root:

```bash
export PYTHONPATH=$(pwd)          # Linux / HPC
set PYTHONPATH=%CD%               # Windows cmd
```

Smoke:

```bash
python -c "import torch, pettingzoo, mpe2; print('spread stack ok')"
python -m pytest tests -q
```

Paste the last pytest line you actually get. Do not invent a count.

### B. SMACv2 (StarCraft II required)

Needs the oxwhirl SMACv2 package **and** a StarCraft II install the wrapper can find. On the university HPC this already existed. On a laptop it usually does not.

```bash
conda activate marl          # name used on the HPC account
export PYTHONPATH=$(pwd)
pip install git+https://github.com/oxwhirl/smacv2.git
python -c "import smacv2, torch; print('smacv2 ok', 'cuda', torch.cuda.is_available())"
python scripts/smoke/test_smacv2_epo.py
```

The smoke script must print that EPO uses `prob_obs_enemy=0.0` and `action_mask=False`, Standard uses `1.0` and `True`, and `conic_fov=False` in both. Team size: Standard 5v5, EPO 6v5 unless you override it.

protobuf: SMACv2 historically wanted an old pin. If `import smacv2` works on your machine, leave that stack alone.

---

## Official Simple Spread recipe (Chapter 5)

Reported setup:

- 5 agents, 4 landmarks  
- landmark order randomised every episode  
- `--agent-neighbors 1 --landmark-neighbors 1`  
- `--num-obstacles 0`  
- message size **8**  
- **200 000** env steps  
- seeds **0, 1, 2**

Example, no-comm then FC, seed 0:

```bash
python scripts/trainers/01_nocomm_mappo_simple_spread.py \
  --seed 0 --total-steps 200000 \
  --n-agents 5 --num-landmarks 4 \
  --agent-neighbors 1 --landmark-neighbors 1 \
  --num-obstacles 0

python scripts/trainers/02_fccomm_mappo_simple_spread.py \
  --seed 0 --total-steps 200000 \
  --n-agents 5 --num-landmarks 4 \
  --agent-neighbors 1 --landmark-neighbors 1 \
  --num-obstacles 0
```

Repeat with the attention and GNN trainers and with `--seed 1` and `--seed 2`.

Add `--no-wandb` if you do not want Weights & Biases. Logging project name used in the dissertation: `msc-marl-comm`.

Checkpoints land under `experiments/<run>/seed<k>/final.pt`.

### Usage checks (Simple Spread)

Ablation and MINE in the dissertation used the **seed-0** checkpoints.

```bash
python scripts/analysis/spread_ablation.py \
  --checkpoint experiments/fccomm_mappo/seed0/final.pt --comm-type fc

python scripts/analysis/spread_mine.py \
  --checkpoint experiments/fccomm_mappo/seed0/final.pt --comm-type fc
```

Same for `attn` and `gnn` with their folders.  
A large ablation drop means the policy needed the channel. A high MINE value only means the outgoing vector was predictable from the sender’s observation.

---

## Official SMACv2 recipe (Chapter 5)

Entry point: `15_mappo_smacv2_parallel.py`.

Pinned flags used for the full campaign:

```text
--hidden 128
--msg-dim 64
--lr-actor 3e-4
--ent-coef 0.01
--num-envs 8
--total-steps 10000000
```

Per condition:

| Condition | Extra flags |
|---|---|
| no-comm | `--comm none --aux-coef 0` |
| FC | `--comm fc --aux-coef 0.2 --pool-type max` |
| attention | `--comm attn --aux-coef 0.2` |
| GNN | `--comm gnn --aux-coef 0.2` |

### EPO (harder observability)

```bash
python scripts/trainers/15_mappo_smacv2_parallel.py \
  --comm none --mode epo --seed 0 \
  --num-envs 8 --total-steps 10000000 \
  --hidden 128 --msg-dim 64 --lr-actor 3e-4 --ent-coef 0.01 \
  --aux-coef 0
```

Then the same line with `--comm fc --aux-coef 0.2 --pool-type max`, then `attn` / `gnn` with `--aux-coef 0.2`. Seeds 0, 1, 2.

EPO default team size in this code is **6v5**.

### Standard

Same flags, `--mode standard`. Standard team size is **5v5**. Action mask on; enemy features not forced to zero.

### Short check before a 10M job

```bash
python scripts/trainers/15_mappo_smacv2_parallel.py \
  --comm none --mode epo --seed 0 \
  --num-envs 4 --total-steps 20000 \
  --no-wandb --aux-coef 0
```

Expect win rate near zero at 20k steps. That is normal. The official number is greedy `test_win_rate` after the full budget, not the first log line.

SMACv2 usage scripts live under `scripts/analysis/` with names such as `spread_ablation.py` counterparts for SMAC if present in your commit. Official win-rate tables and later diagnostic evaluators were **not** pooled in the dissertation.

---

## Channel probes (not the main table)

These were extra:

- Gaussian noise at evaluation or during a probe train  
- low-rank / quantised bandwidth  
- information-bottleneck KL (`--beta` / `--kl-coef` on the GNN trainer when that flag exists)

Budgets were often shorter than 10M (2M on some SMACv2 probes). Do not treat those curves as a four-way robustness league table.

---

## Reproduce “the same result” honestly

You will not get bit-identical floats across machines. You should get the same **pattern**:

- Simple Spread: silent team much worse on return; the three communicating modules close on return; GNN the one that collapsed under ablation in the reported seed-0 check.  
- SMACv2: modest absolute win rates; small official gaps; usage and score can disagree.

If your FC run used `--pool-type mean` you are not on the reported FC protocol. The dissertation FC baseline used **max**.

If EPO smoke shows `prob_obs_enemy=1.0` or `action_mask=True`, you are not on the reported EPO.

---

## Tests

```bash
python -m pytest tests -q
```

Copy the last line the runner prints. Older machines in this project printed, on separate files:

- `tests/test_smacv2_trainer.py` — 53 passed  
- `tests/test_parallel_trainer.py` — 11 passed  

Those counts are historical. Use whatever your current tree prints.

---

## Citation

If you use the code, cite the dissertation:

Hegde, S. (2026). *Emergent communication in cooperative multi-agent reinforcement learning* (MSc dissertation). Manchester Metropolitan University. Supervisor: Dr Ayah Helal. EthOS 91944.

---

## Licence / use

Research code for the dissertation. Simulated coverage and simulated StarCraft only. No permission is implied for physical or weapons use.
