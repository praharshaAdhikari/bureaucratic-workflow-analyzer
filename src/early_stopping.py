"""
early_stopping.py
-----------------
SB3 callback that stops training when the policy has stabilised.

Tracks the per-interval reward *means* reported by TrainingLogger (not raw
episode rewards, which are too noisy).  Stability = the windowed mean has
not improved by more than ``min_delta`` for ``patience`` consecutive checks.

The ``max_std`` guard is intentionally removed — episode reward std of ~10
is normal for stochastic process envs and should not block early stopping.
"""

import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class EarlyStoppingCallback(BaseCallback):
    """
    Stop training when the smoothed reward mean has plateaued.

    Designed to work alongside TrainingLogger: call ``record_interval_mean``
    from TrainingLogger._flush_log() each interval, then this callback checks
    stability at ``check_freq`` timesteps.

    Parameters
    ----------
    window : int
        Number of interval means to smooth over (e.g. 5 = last 5 intervals).
    min_delta : float
        Minimum improvement in smoothed mean to count as progress.
    patience : int
        Consecutive checks with no improvement before stopping.
    check_freq : int
        Timestep interval between checks — should equal log_interval.
    verbose : int
        1 = print status lines.
    """

    def __init__(
        self,
        window: int = 5,
        min_delta: float = 0.5,
        patience: int = 5,
        check_freq: int = 10_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.window     = window
        self.min_delta  = min_delta
        self.patience   = patience
        self.check_freq = check_freq

        # Filled by record_interval_mean() called from TrainingLogger
        self._interval_means: deque[float] = deque(maxlen=window)
        self._patience_count  = 0
        self._last_check_step = 0
        self._best_mean       = -np.inf

    def record_interval_mean(self, mean: float) -> None:
        """Called by TrainingLogger each flush to register the interval mean."""
        self._interval_means.append(mean)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_check_step < self.check_freq:
            return True
        self._last_check_step = self.num_timesteps

        if len(self._interval_means) < self.window:
            if self.verbose >= 1:
                print(
                    f"  [EarlyStopping] step={self.num_timesteps}  "
                    f"warming up ({len(self._interval_means)}/{self.window} intervals)"
                )
            return True

        smoothed = float(np.mean(self._interval_means))
        improved = smoothed - self._best_mean

        if smoothed > self._best_mean:
            self._best_mean = smoothed

        if improved < self.min_delta:
            self._patience_count += 1
            if self.verbose >= 1:
                print(
                    f"  [EarlyStopping] step={self.num_timesteps}  "
                    f"smoothed={smoothed:+.2f}  best={self._best_mean:+.2f}  "
                    f"Δ={improved:+.3f}  patience={self._patience_count}/{self.patience}"
                )
            if self._patience_count >= self.patience:
                if self.verbose >= 1:
                    print(
                        f"  [EarlyStopping] Stopping at step {self.num_timesteps} — "
                        f"smoothed reward plateaued at {smoothed:+.2f} "
                        f"for {self.patience} consecutive checks."
                    )
                return False
        else:
            if self._patience_count > 0 and self.verbose >= 1:
                print(
                    f"  [EarlyStopping] step={self.num_timesteps}  "
                    f"smoothed={smoothed:+.2f}  Δ={improved:+.3f}  "
                    f"— progress, patience reset."
                )
            self._patience_count = 0

        return True
