"""Trainable LoRA factors on the LM's linear weights, stored as a ``zonos-accent-lora/1`` adapter.

A target module ``m`` with weight ``[..., in]`` (packed weights such as ``wkv`` [2, kv, in] count as their flat
``[out, in]`` product) gets ``down`` [rank, in] and ``up`` [out, rank]; the forward adds
``(alpha / rank) * (x @ down.T) @ up.T``. ``up`` starts at zero, so a fresh adapter leaves the logits unchanged and its
first step moves ``up`` only.

The saved file is the format ``zonos3.models.accent_format.load_adapter`` reads, with ``alpha / rank`` folded into
``up``: the reference and the servers apply it as an accent at strength 1.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
from zonos3.models.accent_format import FORMAT, TARGETS, lora_target

# Mixer and latent projections. The router, the experts and the output head stay frozen.
DEFAULT_TARGETS = (
    "attention.in_proj",
    "attention.wq",
    "attention.wkv",
    "attention.wo",
    "feed_forward.latent_down",
    "feed_forward.latent_up",
)


def target_shapes(lm, targets=DEFAULT_TARGETS) -> dict[str, tuple[int, ...]]:
    """Weight shape of every module of ``lm`` that ``targets`` (entries of accent_format.TARGETS) selects."""
    unknown = sorted(set(targets) - set(TARGETS))
    if unknown:
        raise ValueError(f"LoRA targets {unknown} are not adapter targets ({', '.join(TARGETS)})")
    shapes = {}
    for name, param in lm.named_parameters():
        module = name.removesuffix(".weight")
        if name.endswith(".weight") and lora_target(module) in targets:
            shapes[module] = tuple(param.shape)
    if not shapes:
        raise ValueError(f"no module of the model matches the LoRA targets {list(targets)}")
    return shapes


class LoraAdapter(nn.Module):
    def __init__(self, shapes: dict[str, tuple[int, ...]], rank: int, alpha: float, seed: int = 0):
        super().__init__()
        self.shapes, self.rank, self.alpha = dict(shapes), int(rank), float(alpha)
        generator = torch.Generator().manual_seed(seed)
        self.down, self.up = nn.ParameterList(), nn.ParameterList()
        for shape in shapes.values():
            down = torch.empty(self.rank, shape[-1])
            nn.init.kaiming_uniform_(down, a=math.sqrt(5), generator=generator)
            self.down.append(nn.Parameter(down))
            self.up.append(nn.Parameter(torch.zeros(math.prod(shape[:-1]), self.rank)))
        self._index = {module: i for i, module in enumerate(shapes)}

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    def linear(self, module: str, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        i = self._index.get(module)
        if i is None:
            return y
        delta = (x.to(self.down[i].dtype) @ self.down[i].T) @ self.up[i].T
        return y + (self.scale * delta).to(y.dtype)

    def export(self) -> dict:
        """The ``zonos-accent-lora/1`` file content."""
        factors = {
            module + ".weight": {
                "down": self.down[i].detach().float().cpu().clone(),
                "up": (self.scale * self.up[i].detach()).float().cpu(),
                "shape": list(self.shapes[module]),
            }
            for module, i in self._index.items()
        }
        return {"format": FORMAT, "factors": factors, "exact": {}, "lora": {"rank": self.rank, "alpha": self.alpha}}

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.export(), path)

    @classmethod
    def load(cls, path: str | Path) -> "LoraAdapter":
        doc = torch.load(path, map_location="cpu", weights_only=True)
        if doc.get("format") != FORMAT or "lora" not in doc or doc["exact"]:
            raise ValueError(f"{path}: not an adapter this trainer saved")
        shapes = {name.removesuffix(".weight"): tuple(pair["shape"]) for name, pair in doc["factors"].items()}
        adapter = cls(shapes, doc["lora"]["rank"], doc["lora"]["alpha"])
        with torch.no_grad():
            for module, i in adapter._index.items():
                pair = doc["factors"][module + ".weight"]
                adapter.down[i].copy_(pair["down"])
                adapter.up[i].copy_(pair["up"] / adapter.scale)
        return adapter
