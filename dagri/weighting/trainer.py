from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from ultralytics.data.build import InfiniteDataLoader, build_dataloader, seed_worker
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.torch_utils import torch_distributed_zero_first

from dagri.weighting.functions import scores_to_weight_map


class WeightedDetectionTrainer(DetectionTrainer):
    """Detection trainer that can sample training images by difficulty score."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None, weighted_config: dict[str, Any] | None = None):
        self.weighted_config = dict(weighted_config or {})
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def _resolve_sample_weight(self, img_path: str, score_map: dict[str, float]) -> float:
        path = Path(img_path)
        candidates = [str(path), path.name, path.stem]
        resolved = path.resolve()
        candidates.extend([str(resolved), resolved.name, resolved.stem])

        for candidate in candidates:
            if candidate in score_map:
                return float(score_map[candidate])

        return 1.0

    def _build_weighted_dataloader(self, dataset, batch_size: int, rank: int, mode: str):
        if rank != -1 and getattr(self.args, "world_size", 1) > 1:
            LOGGER.warning("Weighted sampling is currently only enabled for single-process training; falling back to default shuffle.")
            return build_dataloader(
                dataset,
                batch=batch_size,
                workers=self.args.workers if mode == "train" else self.args.workers * 2,
                shuffle=True,
                rank=rank,
                drop_last=self.args.compile and mode == "train",
            )

        raw_score_map = self.weighted_config.get("image_score_map") or {}
        if not raw_score_map:
            return build_dataloader(
                dataset,
                batch=batch_size,
                workers=self.args.workers if mode == "train" else self.args.workers * 2,
                shuffle=True,
                rank=rank,
                drop_last=self.args.compile and mode == "train",
            )

        weight_function_name = str(self.weighted_config.get("weight_function", "linear"))
        weight_gamma = float(self.weighted_config.get("weight_gamma", 1.0))
        normalize_scores = bool(self.weighted_config.get("normalize_scores", True))

        sample_score_map = {
            image_path: self._resolve_sample_weight(image_path, raw_score_map)
            for image_path in dataset.im_files
        }
        sample_weight_map = scores_to_weight_map(
            sample_score_map,
            function_name=weight_function_name,
            gamma=weight_gamma,
            normalize=normalize_scores,
        )
        sample_weights = torch.as_tensor(
            [sample_weight_map.get(image_path, 1.0) for image_path in dataset.im_files],
            dtype=torch.double,
        )

        if not np.isfinite(sample_weights.cpu().numpy()).all():
            raise ValueError("Weighted sampler received non-finite sample weights")

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        nd = torch.cuda.device_count()
        nw = min((os.cpu_count() or 1) // max(nd, 1), self.args.workers if mode == "train" else self.args.workers * 2)
        generator = torch.Generator()
        generator.manual_seed(6148914691236517205 + rank)

        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=nw,
            sampler=sampler,
            prefetch_factor=4 if nw > 0 else None,
            pin_memory=nd > 0,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=self.args.compile and mode == "train",
        )

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct and return a dataloader for the specified mode."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        if mode != "train" or not bool(self.weighted_config.get("enabled", False)):
            shuffle = mode == "train"
            if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
                LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
                shuffle = False
            return build_dataloader(
                dataset,
                batch=batch_size,
                workers=self.args.workers if mode == "train" else self.args.workers * 2,
                shuffle=shuffle,
                rank=rank,
                drop_last=self.args.compile and mode == "train",
            )

        return self._build_weighted_dataloader(dataset, batch_size, rank, mode)
