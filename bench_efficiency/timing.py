"""Shared timing primitives for the runtime-efficiency benchmark.

Design notes
------------
* Every measurement is taken around a CUDA synchronisation barrier, otherwise
  PyTorch's async kernel launches make the timings meaningless.
* Nothing here re-implements a method's training loop.  We monkeypatch the
  method's own functions by name, so the numbers come from the exact code that
  produced the paper results.
* Warm-up is removed by the *two-point* trick: run the training function twice
  with a different `--max-train-steps` cap and take the slope.  Model loading,
  dataset construction and CUDA warm-up are identical in both runs and cancel.
"""

from __future__ import annotations

import functools
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def _torch():
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return None
    return torch


def sync() -> None:
    """Block until all queued GPU work has finished."""
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory() -> None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> Optional[float]:
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def count_params(model: Any) -> Dict[str, int]:
    """Total / trainable parameter counts for a torch or keras model."""
    if model is None:
        return {}
    if hasattr(model, "parameters"):  # torch
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"params_total": int(total), "params_trainable": int(trainable)}
    if hasattr(model, "count_params"):  # keras
        total = int(model.count_params())
        trainable = int(sum(int(w.shape.num_elements()) for w in model.trainable_weights))
        return {"params_total": total, "params_trainable": trainable}
    return {}


def environment() -> Dict[str, Any]:
    torch = _torch()
    env: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    if torch is not None:
        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["gpu_count"] = torch.cuda.device_count()
            env["cuda"] = torch.version.cuda
    try:
        import tensorflow as tf  # noqa: PLC0415

        env["tensorflow"] = tf.__version__
        env["tf_gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except ImportError:
        pass
    return env


class Phase:
    """Accumulates one named measurement region."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: List[Dict[str, Any]] = []

    @property
    def total_seconds(self) -> float:
        return sum(c["seconds"] for c in self.calls)

    @property
    def total_windows(self) -> int:
        return sum(int(c.get("windows") or 0) for c in self.calls)

    def summary(self, *, drop_first: bool = False) -> Dict[str, Any]:
        calls = self.calls[1:] if (drop_first and len(self.calls) > 1) else self.calls
        seconds = sum(c["seconds"] for c in calls)
        windows = sum(int(c.get("windows") or 0) for c in calls)
        out: Dict[str, Any] = {
            "calls": len(calls),
            "seconds": seconds,
            "windows": windows,
            "dropped_warmup_call": drop_first and len(self.calls) > 1,
        }
        if windows:
            out["ms_per_window"] = 1000.0 * seconds / windows
        return out


class PhaseTimer:
    """Collects timings for named phases and writes them out as JSON."""

    def __init__(self) -> None:
        self.phases: Dict[str, Phase] = {}
        self.extra: Dict[str, Any] = {}

    def phase(self, name: str) -> Phase:
        return self.phases.setdefault(name, Phase(name))

    def wrap(
        self,
        name: str,
        fn: Callable[..., Any],
        *,
        windows_at: Optional[int] = None,
    ) -> Callable[..., Any]:
        """Return `fn` wrapped so each call's wall time lands in phase `name`.

        `windows_at` is the tuple index of the forecast-window count in the
        return value (all `evaluate_subject` variants put it at index 3).
        """
        ph = self.phase(name)

        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            sync()
            start = time.perf_counter()
            out = fn(*args, **kwargs)
            sync()
            elapsed = time.perf_counter() - start

            record: Dict[str, Any] = {"seconds": elapsed}
            if windows_at is not None and isinstance(out, tuple) and len(out) > windows_at:
                try:
                    record["windows"] = int(out[windows_at])
                except (TypeError, ValueError):
                    pass
            ph.calls.append(record)
            return out

        inner.__bench_wrapped__ = True  # type: ignore[attr-defined]
        return inner

    def to_dict(self) -> Dict[str, Any]:
        return {
            "environment": environment(),
            "phases": {
                name: {"calls": ph.calls, "total_seconds": ph.total_seconds}
                for name, ph in self.phases.items()
            },
            **self.extra,
        }

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))


class Stopwatch:
    """Context manager for a one-off region (data loading, model load, ...)."""

    def __init__(self, timer: PhaseTimer, name: str) -> None:
        self.timer = timer
        self.name = name

    def __enter__(self) -> "Stopwatch":
        sync()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        sync()
        self.timer.phase(self.name).calls.append(
            {"seconds": time.perf_counter() - self._start}
        )
