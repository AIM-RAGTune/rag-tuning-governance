from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import nn
from torch.nn import functional as F

from square_sim.models.squaresim.zones import fixed_zone_masks


@dataclass(frozen=True)
class RegionSelection:
    masks: torch.Tensor
    crop_coords: torch.Tensor | None
    region_scores: torch.Tensor
    region_names: list[str]
    diagnostics: dict[str, Any] = field(default_factory=dict)


class RegionSelector(Protocol):
    def select(self, field: torch.Tensor, zone_masks: torch.Tensor | None = None) -> RegionSelection:
        ...


def _rect_masks(
    batch: int,
    grid_h: int,
    grid_w: int,
    coords: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    masks = torch.zeros(batch, coords.shape[1], grid_h, grid_w, device=device, dtype=dtype)
    for b in range(batch):
        for r in range(coords.shape[1]):
            y0, x0, y1, x1 = [int(v) for v in coords[b, r].tolist()]
            masks[b, r, y0:y1, x0:x1] = 1.0
    denom = masks.flatten(2).sum(-1).clamp_min(1.0).view(batch, coords.shape[1], 1, 1)
    return masks / denom


class FixedZoneSelector(nn.Module):
    def __init__(self, grid_size: int, num_regions: int = 2, include_overlap: bool = True):
        super().__init__()
        masks = fixed_zone_masks(grid_size, include_overlap)
        self.register_buffer("fixed_masks", masks[:num_regions])

    def select(self, field: torch.Tensor, zone_masks: torch.Tensor | None = None) -> RegionSelection:
        masks = (zone_masks[: self.fixed_masks.shape[0]] if zone_masks is not None else self.fixed_masks).to(
            device=field.device, dtype=field.dtype
        )
        batch_masks = masks.unsqueeze(0).expand(field.shape[0], -1, -1, -1)
        scores = (field.square().unsqueeze(1) * batch_masks.unsqueeze(2)).sum(dim=(-1, -2, -3))
        return RegionSelection(
            masks=batch_masks,
            crop_coords=None,
            region_scores=scores,
            region_names=[f"fixed_zone_{i}" for i in range(batch_masks.shape[1])],
        )


class OverlapZoneSelector(FixedZoneSelector):
    def __init__(self, grid_size: int, num_regions: int = 2):
        super().__init__(grid_size, num_regions=max(1, num_regions), include_overlap=True)

    def select(self, field: torch.Tensor, zone_masks: torch.Tensor | None = None) -> RegionSelection:
        selection = super().select(field, zone_masks)
        return RegionSelection(
            masks=selection.masks,
            crop_coords=selection.crop_coords,
            region_scores=selection.region_scores,
            region_names=[f"overlap_zone_{i}" for i in range(selection.masks.shape[1])],
            diagnostics={"selector": "overlap"},
        )


class WholeFieldSelector(nn.Module):
    def __init__(self, num_regions: int = 1):
        super().__init__()
        self.num_regions = num_regions

    def select(self, field: torch.Tensor, zone_masks: torch.Tensor | None = None) -> RegionSelection:
        batch, _channels, height, width = field.shape
        mask = torch.full(
            (batch, self.num_regions, height, width),
            1.0 / float(height * width),
            device=field.device,
            dtype=field.dtype,
        )
        score = field.square().mean(dim=(-1, -2, -3), keepdim=False).view(batch, 1)
        score = score.expand(-1, self.num_regions)
        return RegionSelection(mask, None, score, ["whole_field"] * self.num_regions, {"selector": "whole_field"})


class RandomRegionSelector(nn.Module):
    def __init__(self, patch_size: int | tuple[int, int] = 12, num_regions: int = 2, seed: int = 42):
        super().__init__()
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.num_regions = num_regions
        self.seed = seed

    def select(self, field: torch.Tensor, zone_masks: torch.Tensor | None = None) -> RegionSelection:
        batch, _channels, height, width = field.shape
        ph, pw = min(self.patch_size[0], height), min(self.patch_size[1], width)
        gen = torch.Generator(device=field.device)
        gen.manual_seed(self.seed)
        ys = torch.randint(0, max(height - ph + 1, 1), (batch, self.num_regions), generator=gen, device=field.device)
        xs = torch.randint(0, max(width - pw + 1, 1), (batch, self.num_regions), generator=gen, device=field.device)
        coords = torch.stack([ys, xs, ys + ph, xs + pw], dim=-1)
        masks = _rect_masks(batch, height, width, coords, field.device, field.dtype)
        scores = torch.rand(batch, self.num_regions, generator=gen, device=field.device, dtype=field.dtype)
        return RegionSelection(masks, coords, scores, [f"random_{i}" for i in range(self.num_regions)])


class _TopPatchSelector(nn.Module):
    def __init__(self, patch_size: int | tuple[int, int] = 12, num_regions: int = 2):
        super().__init__()
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.num_regions = num_regions

    def score_map(self, field: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def select(self, field: torch.Tensor, zone_masks: torch.Tensor | None = None) -> RegionSelection:
        batch, _channels, height, width = field.shape
        ph, pw = min(self.patch_size[0], height), min(self.patch_size[1], width)
        score = self.score_map(field)
        pooled = F.avg_pool2d(score, kernel_size=(ph, pw), stride=1)
        flat = pooled.flatten(1)
        k = min(self.num_regions, flat.shape[1])
        values, indices = torch.topk(flat, k=k, dim=1)
        out_w = pooled.shape[-1]
        ys = torch.div(indices, out_w, rounding_mode="floor")
        xs = indices % out_w
        coords = torch.stack([ys, xs, ys + ph, xs + pw], dim=-1)
        masks = _rect_masks(batch, height, width, coords, field.device, field.dtype)
        names = [f"{self.__class__.__name__.replace('Selector', '').lower()}_{i}" for i in range(k)]
        return RegionSelection(masks, coords, values, names)


class TopEnergySelector(_TopPatchSelector):
    def score_map(self, field: torch.Tensor) -> torch.Tensor:
        return field.square().mean(dim=1, keepdim=True)


class TopGradientSelector(_TopPatchSelector):
    def score_map(self, field: torch.Tensor) -> torch.Tensor:
        gy = F.pad(field[..., 1:, :] - field[..., :-1, :], (0, 0, 0, 1))
        gx = F.pad(field[..., :, 1:] - field[..., :, :-1], (0, 1, 0, 0))
        return (gx.square() + gy.square()).mean(dim=1, keepdim=True)


class LearnedMaskSelector(nn.Module):
    def __init__(self, channels: int, num_regions: int = 2, temperature: float = 1.0):
        super().__init__()
        self.num_regions = num_regions
        self.temperature = temperature
        self.net = nn.Sequential(
            nn.Conv2d(channels, max(4, channels * 2), kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(max(4, channels * 2), num_regions, kernel_size=1),
        )

    def select(self, field: torch.Tensor, zone_masks: torch.Tensor | None = None) -> RegionSelection:
        logits = self.net(field) / max(self.temperature, 1e-6)
        masks = torch.softmax(logits.flatten(2), dim=-1).view(field.shape[0], self.num_regions, *field.shape[-2:])
        scores = (field.square().unsqueeze(1) * masks.unsqueeze(2)).sum(dim=(-1, -2, -3))
        return RegionSelection(masks, None, scores, [f"learned_mask_{i}" for i in range(self.num_regions)])


def make_region_selector(
    mode: str,
    *,
    grid_size: int,
    channels: int,
    num_regions: int = 2,
    patch_size: int | tuple[int, int] = 12,
    seed: int = 42,
) -> nn.Module:
    mode = mode.lower()
    if mode == "fixed_zone":
        return FixedZoneSelector(grid_size, num_regions=num_regions)
    if mode == "overlap":
        return OverlapZoneSelector(grid_size, num_regions=num_regions)
    if mode == "top_energy":
        return TopEnergySelector(patch_size, num_regions=num_regions)
    if mode == "top_gradient":
        return TopGradientSelector(patch_size, num_regions=num_regions)
    if mode == "learned":
        return LearnedMaskSelector(channels, num_regions=num_regions)
    if mode == "whole_field":
        return WholeFieldSelector(num_regions=1)
    if mode == "random":
        return RandomRegionSelector(patch_size, num_regions=num_regions, seed=seed)
    if mode in {"uncertainty", "uncertainty_trigger"}:
        return TopGradientSelector(patch_size, num_regions=num_regions)
    raise ValueError(f"Unknown snapshot region selector: {mode}")
