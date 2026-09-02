"""Model-server and training dependency diagnostics.

The checks are intentionally import-based and do not download models or mutate
the environment.  A freshly cloned repository can therefore fail early with a
specific installation instruction instead of failing halfway through an
evaluation or training job.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import import_module
import platform
import shutil
import sys
from typing import Iterable, Literal


ReadinessProfile = Literal["core", "eval", "train"]


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    available: bool
    version: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ReadinessReport:
    profile: ReadinessProfile
    ready: bool
    python_version: str
    platform: str
    dependencies: tuple[DependencyStatus, ...]
    cuda_available: bool | None
    cuda_device: str | None
    ollama_executable: str | None

    def model_dump(self) -> dict[str, object]:
        return asdict(self)


_CORE_MODULES = ("numpy", "pydantic", "PIL", "pypdfium2", "softdoc")
_EVAL_MODULES = _CORE_MODULES + ("torch", "transformers", "sentence_transformers")
_TRAIN_MODULES = _EVAL_MODULES + ("accelerate", "datasets", "peft")


def _module_status(name: str) -> DependencyStatus:
    try:
        module = import_module(name)
    except Exception as exc:  # pragma: no cover - exact import failures vary
        return DependencyStatus(
            name=name,
            available=False,
            detail=f"{type(exc).__name__}: {exc}",
        )
    version = getattr(module, "__version__", None)
    if version is None and name == "PIL":
        version = getattr(module, "PILLOW_VERSION", None)
    return DependencyStatus(
        name=name,
        available=True,
        version=str(version) if version is not None else None,
    )


def _statuses(names: Iterable[str]) -> tuple[DependencyStatus, ...]:
    return tuple(_module_status(name) for name in names)


def check_server_readiness(profile: ReadinessProfile = "core") -> ReadinessReport:
    """Return deterministic environment diagnostics for one usage profile."""

    if profile == "core":
        module_names = _CORE_MODULES
    elif profile == "eval":
        module_names = _EVAL_MODULES
    elif profile == "train":
        module_names = _TRAIN_MODULES + (("bitsandbytes",) if platform.system() == "Linux" else ())
    else:  # pragma: no cover - guarded by CLI typing/argparse
        raise ValueError(f"Unknown readiness profile: {profile}")

    dependencies = _statuses(module_names)
    cuda_available: bool | None = None
    cuda_device: str | None = None
    torch_status = next((item for item in dependencies if item.name == "torch"), None)
    if torch_status is not None and torch_status.available:
        try:
            torch = import_module("torch")
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                cuda_device = str(torch.cuda.get_device_name(0))
        except Exception as exc:  # pragma: no cover - hardware-specific
            cuda_available = False
            cuda_device = f"CUDA inspection failed: {type(exc).__name__}: {exc}"

    python_supported = (3, 11) <= sys.version_info[:2] < (3, 14)
    dependencies_ready = all(item.available for item in dependencies)
    hardware_ready = profile != "train" or cuda_available is True
    return ReadinessReport(
        profile=profile,
        ready=python_supported and dependencies_ready and hardware_ready,
        python_version=platform.python_version(),
        platform=platform.platform(),
        dependencies=dependencies,
        cuda_available=cuda_available,
        cuda_device=cuda_device,
        ollama_executable=shutil.which("ollama"),
    )


def readiness_install_hint(profile: ReadinessProfile) -> str:
    if profile == "core":
        return 'python -m pip install -e ".[dev]"'
    if profile == "eval":
        return 'python -m pip install -e ".[dev,dense,visual-retrieval]"'
    return 'python -m pip install -e ".[dev,dense,visual-retrieval,train]"'
