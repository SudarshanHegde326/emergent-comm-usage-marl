"""Vectorized SMACv2 environments via subprocess workers.

This runs N StarCraft II games in parallel, one per worker process, so the GPU
is not left idle waiting for a single CPU-bound game to step. It is the change
that takes throughput from ~50 fps (one env) toward 200-400 fps (many envs).

DESIGN (and why it is built this way)
-------------------------------------
* One OS process per environment. StarCraft II is heavy and holds process-wide
  state; threads would contend on the GIL and on SC2 itself, so we use real
  processes (multiprocessing with the 'spawn' start method, which is the only
  safe option with SC2 + CUDA in the parent).
* The parent sends a command tuple to each worker over a Pipe and reads back the
  result. Steps are dispatched to ALL workers, THEN collected, so the games run
  concurrently rather than one-after-another.
* AUTO-RESET: when an episode ends in a worker, the worker resets immediately
  and returns the FIRST observation of the next episode, plus the terminal
  info of the one that just ended. The trainer therefore always has a live
  observation for every env and never stalls. This is the standard vectorized-
  RL contract (as in Gym's SyncVectorEnv / SB3's SubprocVecEnv).
* Each worker gets a DISTINCT seed (base_seed + rank). SMACv2 is procedurally
  generated, so identical seeds would make all N workers play the same maps and
  destroy the diversity the benchmark exists to provide.

The parent process does all the neural-network work on the GPU; workers only
run the environment. No torch tensors cross the pipe -- only small numpy arrays
and Python scalars -- so there is no CUDA-in-subprocess hazard.
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Callable

import numpy as np


# =============================================================================
# Worker process
# =============================================================================
def _worker(remote, parent_remote, env_fn_pickle, rank: int, ep_limit: int):
    """Run one environment, forever, taking commands from the parent.

    The worker also enforces the episode time limit: if `ep_limit` steps elapse
    without natural termination, it treats that as a (truncated) episode end,
    auto-resets, and reports done=True with truncated=True in the info. This
    keeps ALL episode-boundary handling in one place and means the parent never
    has to do a mid-rollout targeted reset (which would perturb other envs).
    """
    import cloudpickle
    parent_remote.close()
    env = cloudpickle.loads(env_fn_pickle)()
    t_in_ep = 0

    def full_obs():
        return (np.asarray(env.get_obs(), dtype=np.float32),
                np.asarray(env.get_state(), dtype=np.float32),
                np.asarray(env.get_avail_actions(), dtype=np.float32))

    try:
        while True:
            cmd, data = remote.recv()

            if cmd == "step":
                reward, terminated, info = env.step(data)
                t_in_ep += 1
                truncated = (ep_limit > 0) and (t_in_ep >= ep_limit) and not terminated
                done = bool(terminated) or bool(truncated)
                if done:
                    term_info = dict(info)
                    term_info["truncated"] = bool(truncated)
                    env.reset()
                    t_in_ep = 0
                    obs, state, avail = full_obs()
                    remote.send((obs, state, avail, float(reward), True, term_info))
                else:
                    obs, state, avail = full_obs()
                    info = dict(info); info["truncated"] = False
                    remote.send((obs, state, avail, float(reward), False, info))

            elif cmd == "reset":
                env.reset()
                t_in_ep = 0
                remote.send(full_obs())

            elif cmd == "get_env_info":
                remote.send(env.get_env_info())

            elif cmd == "get_obs_layout":
                raw = getattr(env, "env", env)
                try:
                    move = int(raw.get_obs_move_feats_size())
                    en = raw.get_obs_enemy_feats_size()
                    enemy = int(en) if isinstance(en, int) else int(np.prod(en))
                    remote.send((move, enemy))
                except Exception:
                    remote.send(None)

            elif cmd == "close":
                try:
                    env.close()
                except Exception:
                    pass
                remote.send(True)
                break

            else:
                raise RuntimeError(f"unknown command to worker: {cmd}")
    except KeyboardInterrupt:
        pass
    finally:
        try:
            env.close()
        except Exception:
            pass
        remote.close()


# =============================================================================
# Parent-side vectorized controller
# =============================================================================
class SubprocVecSMAC:
    """Controls N SMACv2 workers. Presents a batched step/reset interface.

    Shapes returned (B = num_envs, N = n_agents, A = n_actions):
        obs    (B, N, obs_dim)
        state  (B, state_dim)
        avail  (B, N, A)
    step() also returns:
        reward (B,)          scalar team reward per env
        done   (B,)          True where an episode ended THIS step (natural
                             termination; the worker has already auto-reset and
                             the returned obs is the NEXT episode's first obs)
        infos  list[dict]    per-env info; contains 'battle_won' on done steps
    """

    def __init__(self, env_fns: list[Callable], ep_limit: int = 0):
        self.n = len(env_fns)
        self.closed = False
        ctx = mp.get_context("spawn")   # required for SC2 + CUDA-in-parent

        import cloudpickle
        self.remotes, self.work_remotes = zip(*[ctx.Pipe() for _ in range(self.n)])
        self.procs = []
        for rank, (work_remote, remote, fn) in enumerate(
                zip(self.work_remotes, self.remotes, env_fns)):
            p = ctx.Process(
                target=_worker,
                args=(work_remote, remote, cloudpickle.dumps(fn), rank, ep_limit),
                daemon=True,
            )
            p.start()
            self.procs.append(p)
            work_remote.close()   # parent does not use the worker end

        self.remotes[0].send(("get_env_info", None))
        self._env_info = self.remotes[0].recv()

    # ---- introspection ----
    def get_env_info(self) -> dict:
        return dict(self._env_info)

    def get_obs_layout(self):
        """(move_dim, enemy_dim) for the aux label, or None if unavailable."""
        self.remotes[0].send(("get_obs_layout", None))
        return self.remotes[0].recv()

    @property
    def num_envs(self) -> int:
        return self.n

    # ---- batched ops ----
    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        obs, state, avail = zip(*[r.recv() for r in self.remotes])
        return (np.stack(obs), np.stack(state), np.stack(avail))

    def step(self, actions: np.ndarray):
        """actions: (B, N) integer array. Dispatch to ALL workers, THEN gather.

        Dispatching before gathering is what makes the games run concurrently;
        gathering first would serialise them and destroy the speedup.
        """
        assert actions.shape[0] == self.n, \
            f"expected {self.n} action rows, got {actions.shape[0]}"
        for r, a in zip(self.remotes, actions):
            r.send(("step", list(int(x) for x in a)))
        results = [r.recv() for r in self.remotes]
        obs, state, avail, reward, done, infos = zip(*results)
        return (np.stack(obs), np.stack(state), np.stack(avail),
                np.asarray(reward, dtype=np.float32),
                np.asarray(done, dtype=bool),
                list(infos))

    def close(self):
        if getattr(self, "closed", True):
            return
        # getattr guards: if __init__ failed before setting remotes/procs,
        # __del__ still calls close() and must not raise a secondary error
        # that masks the real construction failure.
        for r in getattr(self, "remotes", []):
            try:
                r.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for r in getattr(self, "remotes", []):
            try:
                r.recv()
            except (EOFError, BrokenPipeError):
                pass
        for p in getattr(self, "procs", []):
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        self.closed = True

    def __del__(self):
        self.close()


def make_vec(env_fn_factory: Callable[[int], Callable], num_envs: int,
             ep_limit: int = 0):
    """Build a SubprocVecSMAC of `num_envs`, each from env_fn_factory(rank).

    env_fn_factory(rank) must return a zero-arg callable that builds ONE env
    with a rank-dependent seed (so the workers do not all play identical maps).
    ep_limit>0 makes workers auto-reset on time-limit truncation as well as on
    natural termination.
    """
    return SubprocVecSMAC([env_fn_factory(rank) for rank in range(num_envs)],
                          ep_limit=ep_limit)
