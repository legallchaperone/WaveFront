"""Metric computation + fan-out for the trainer's observability layer.

Adapted in spirit from torchtitan's ``MetricsProcessor``: the trainer hands a
plain ``dict`` of scalars (the Recipe's losses, read off the
:class:`~minwm.engine.events.EventStorage`) to :meth:`MetricsProcessor.log`,
which augments it with throughput (steps/sec, ms/step) and — on CUDA — peak
memory, prints a STDOUT line, and fans the merged dict out to the configured
backends (TensorBoard / WandB).

The processor is **decoupled from EventStorage**: it takes a dict, so it is
unit-testable without the global bus and the trainer stays in control of what
gets merged. Real writers are built lazily (on the first :meth:`log`) and only
on the main process, so constructing a trainer never triggers ``wandb.init`` or
creates a TensorBoard run directory (important for dry runs / pytest).
"""

import os
import time

import torch

from minwm.config.schema import Monitor
from minwm.utils import comm
from minwm.utils.logger import init_logger

logger = init_logger(__name__)


class BaseWriter:
    """A metric sink. Subclasses fan one :meth:`log` call out to a backend."""

    def log(self, metrics: dict[str, float], step: int) -> None:
        """Write ``metrics`` at ``step`` (no-op in the base class)."""

    def close(self) -> None:
        """Flush / release backend resources (no-op in the base class)."""


class TensorBoardWriter(BaseWriter):
    """Fan metrics out to TensorBoard event files.

    Args:
        log_dir (str): directory for the event files.
    """

    def __init__(self, log_dir: str) -> None:
        from torch.utils.tensorboard import SummaryWriter

        os.makedirs(log_dir, exist_ok=True)
        self._writer = SummaryWriter(log_dir)
        logger.info("TensorBoard logging to %s", log_dir)

    def log(self, metrics: dict[str, float], step: int) -> None:
        for name, value in metrics.items():
            self._writer.add_scalar(name, value, step)

    def close(self) -> None:
        self._writer.close()


class WandBWriter(BaseWriter):
    """Fan metrics out to Weights & Biases.

    The run path (``entity/project/run_id``) is saved to
    ``<log_dir>/wandb_run_path.txt``. A resumed training run reads it back and keeps
    logging to the same W&B run; a fresh run starts a new W&B run and overwrites the
    file. ``tools/upload_hf.py`` reads it to attach checkpoint links to the run.

    Args:
        cfg (Monitor): W&B knobs. ``wandb_project`` falls back to ``WANDB_PROJECT``
            then ``"minwm"``; ``wandb_run_name`` falls back to ``WANDB_RUN_NAME``,
            then the last component of ``log_dir``.
        log_dir (str): the run's output dir; W&B run dir and home of the run-path file.
        config (dict | None): run hyperparameters, stored as the W&B run config.
        resume (bool): continue the W&B run recorded in ``log_dir``, if any.
    """

    RUN_PATH_FILE = "wandb_run_path.txt"

    def __init__(
        self,
        cfg: Monitor,
        log_dir: str,
        config: dict | None = None,
        resume: bool = False,
    ) -> None:
        import wandb

        self._wandb = wandb
        os.makedirs(log_dir, exist_ok=True)
        path_file = os.path.join(log_dir, self.RUN_PATH_FILE)
        run_id = None
        if resume and os.path.exists(path_file):
            with open(path_file) as f:
                run_id = f.read().strip().split("/")[-1] or None
        run = wandb.init(
            project=cfg.wandb_project or os.getenv("WANDB_PROJECT", "minwm"),
            name=(
                cfg.wandb_run_name
                or os.getenv("WANDB_RUN_NAME")
                or os.path.basename(os.path.normpath(log_dir))
            ),
            group=cfg.wandb_group,
            tags=list(cfg.wandb_tags) or None,
            notes=cfg.wandb_notes,
            dir=log_dir,
            config=config,
            id=run_id,
            resume="allow" if run_id else None,
        )
        with open(path_file, "w") as f:
            f.write(run.path)
        logger.info("WandB logging enabled (run id %s%s)", run.id, ", resumed" if run_id else "")

    def log(self, metrics: dict[str, float], step: int) -> None:
        self._wandb.log(metrics, step=step)

    def close(self) -> None:
        if self._wandb.run is not None:
            self._wandb.finish()


class WriterList(BaseWriter):
    """Fan one :meth:`log` / :meth:`close` out to a list of writers."""

    def __init__(self, writers: list[BaseWriter]) -> None:
        self._writers = writers

    def log(self, metrics: dict[str, float], step: int) -> None:
        for writer in self._writers:
            writer.log(metrics, step)

    def close(self) -> None:
        for writer in self._writers:
            writer.close()


def _build_writers(
    cfg: Monitor, log_dir: str, run_config: dict | None = None, resume: bool = False
) -> WriterList:
    """Construct the configured backend writers (main process only).

    Args:
        cfg (Monitor): the monitor config naming the backends.
        log_dir (str): base directory for backend output.
        run_config (dict | None): run hyperparameters for backends that store them.
        resume (bool): continue the previous backend run (W&B) recorded in ``log_dir``.

    Returns:
        WriterList: the fan-out container (empty when no backends configured).
    """
    writers: list[BaseWriter] = []
    for backend in cfg.backends:
        if backend == "tensorboard":
            writers.append(TensorBoardWriter(log_dir))
        elif backend == "wandb":
            writers.append(WandBWriter(cfg, log_dir, config=run_config, resume=resume))
        else:
            raise ValueError(f"unknown monitor backend {backend!r}; valid: tensorboard, wandb")
    return WriterList(writers)


class DeviceMemoryMonitor:
    """Tracks peak CUDA memory between log windows.

    Only constructed on CUDA; on CPU the :class:`MetricsProcessor` skips it
    entirely so no ``memory/*`` keys are emitted.
    """

    def __init__(self) -> None:
        self._capacity = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.reset_peak_memory_stats()

    def peak_stats(self) -> dict[str, float]:
        """Return peak-reserved GiB / % and OOM count since the last reset."""
        info = torch.cuda.memory_stats()
        max_reserved = info.get("reserved_bytes.all.peak", 0)
        return {
            "memory/max_reserved(GiB)": max_reserved / 1024**3,
            "memory/max_reserved(%)": 100 * max_reserved / self._capacity,
            "memory/num_ooms": info.get("num_ooms", 0),
        }

    def reset(self) -> None:
        torch.cuda.reset_peak_memory_stats()


class MetricsProcessor:
    """Compute throughput / memory, then fan metrics out to STDOUT + backends.

    Decoupled from :class:`~minwm.engine.events.EventStorage`: :meth:`log` takes a
    plain dict so it is unit-testable without the global bus, and the trainer
    stays in control of what it merges. Writers are built lazily on the first
    :meth:`log` (so constructing a trainer never triggers ``wandb.init`` or
    TensorBoard dir creation) and only on the main process.

    Args:
        cfg (Monitor): the monitor config (backends + W&B knobs).
        log_dir (str): base directory for backend output.
        run_config (dict | None): run hyperparameters handed to backends that store
            them (W&B run config).
        resume (bool): the trainer is resuming from a checkpoint, so backends
            continue their previous run instead of starting a new one.
    """

    def __init__(
        self,
        cfg: Monitor,
        log_dir: str,
        run_config: dict | None = None,
        resume: bool = False,
    ) -> None:
        self._cfg = cfg
        self._log_dir = log_dir
        self._run_config = run_config
        self._resume = resume
        self._writers: WriterList | None = None
        self._mem = DeviceMemoryMonitor() if torch.cuda.is_available() else None
        self._t_last = time.perf_counter()
        self._step_last = 0

    def start_window(self, step: int) -> None:
        """Reset the throughput window. Called at ``train()`` start, not init.

        Args:
            step (int): the step the loop is (re)starting from, so the first
                logged ``steps/sec`` excludes model-build / checkpoint-load time.
        """
        self._t_last = time.perf_counter()
        self._step_last = step
        if self._mem is not None:
            self._mem.reset()

    def log(self, step: int, metrics: dict[str, float]) -> None:
        """Merge perf + caller metrics, print to STDOUT, fan out to backends.

        Args:
            step (int): the current global step (post-increment, 1-based).
            metrics (dict[str, float]): caller-supplied scalars (losses, etc.).
        """
        if not comm.is_main_process():
            return
        if self._writers is None:
            self._writers = _build_writers(self._cfg, self._log_dir, self._run_config, self._resume)

        now = time.perf_counter()
        dt = now - self._t_last
        nsteps = step - self._step_last
        merged: dict[str, float] = dict(metrics)
        if dt > 0 and nsteps > 0:
            merged["steps/sec"] = nsteps / dt
            merged["ms/step"] = 1000 * dt / nsteps
        if self._mem is not None:
            merged.update(self._mem.peak_stats())
            self._mem.reset()

        body = "  ".join(f"{k}: {v:.4g}" for k, v in merged.items())
        logger.info("step %d  %s", step, body)
        self._writers.log(merged, step)

        self._t_last = now
        self._step_last = step

    def close(self) -> None:
        """Flush and close all writers (idempotent)."""
        if self._writers is not None:
            self._writers.close()
            self._writers = None
