from pathlib import Path
from typing import Dict, List, Tuple

from dagri.interfaces import (
    BoundingBox,
    ImageDifficultyProperties,
    ObjectDifficultyProperties,
    ScorerInterface,
    ScoringConfig,
    ScoringResults,
)

class MinScorer(ScorerInterface):
    def __init__(self, scoring_config: ScoringConfig):
        self.scoring_config = scoring_config

    def score(
        self,
        optimal_conf_threshold_prediction_dir: str,
        low_conf_prediction_dir: str,
        images_dir: str,
        labels_dir: str,
    ) -> ScoringResults:
        """
        Score every image using object-level miss cost and image-level false-positive penalty.

        S_obj = min_pred(alpha*(1-conf) + beta*(1-IoU)) over predictions with IoU >= iou_threshold.
        S_img = w1*avg(S_obj) + w2*FP_rate.
        """
        if not labels_dir:
            raise ValueError("labels_dir is required for scoring")
        if not images_dir:
            raise ValueError("images_dir is required for scoring")

        low_dir = Path(low_conf_prediction_dir)
        opt_dir = Path(optimal_conf_threshold_prediction_dir)
        label_dir = Path(labels_dir)
        image_dir = Path(images_dir)

        if not label_dir.exists():
            raise FileNotFoundError(f"Ground-truth label directory not found: {label_dir}")
        if not low_dir.exists():
            raise FileNotFoundError(f"Low-confidence prediction directory not found: {low_dir}")
        if not opt_dir.exists():
            raise FileNotFoundError(f"Optimal-threshold prediction directory not found: {opt_dir}")
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        alpha = float(self.scoring_config.alpha)
        beta = float(self.scoring_config.beta)
        iou_thr = float(self.scoring_config.iou_threshold)
        w1 = float(self.scoring_config.object_weight)
        # Kept for backward compatibility with existing config key naming.
        configured_w2 = float(self.scoring_config.false_positive_weight)
        weight_mode = str(getattr(self.scoring_config, "weight_mode", "fixed")).strip().lower()
        if weight_mode not in {"fixed", "mean_match", "balance_correlation"}:
            raise ValueError(
                f"Unknown scoring weight_mode={weight_mode}. "
                "Use 'fixed', 'mean_match', or 'balance_correlation'."
            )

        per_image_rows: List[Dict] = []

        for gt_file in sorted(label_dir.glob("*.txt")):
            stem = gt_file.stem
            low_file = low_dir / f"{stem}.txt"
            opt_file = opt_dir / f"{stem}.txt"

            gt_boxes = self._read_yolo_file(gt_file, has_conf=False)
            low_preds = self._read_yolo_file(low_file, has_conf=True) if low_file.exists() else []
            opt_preds = self._read_yolo_file(opt_file, has_conf=True) if opt_file.exists() else []

            object_details: List[ObjectDifficultyProperties] = []
            obj_scores: List[float] = []

            for obj_id, gt in enumerate(gt_boxes):
                gt_cls, gt_xywh, _ = gt
                gt_xyxy = self._xywh_to_xyxy(gt_xywh)

                # Default to maximum difficulty when no matched prediction exists.
                # Since conf and IoU are each in [0, 1], max cost is alpha + beta.
                best_cost = alpha + beta
                for pred_cls, pred_xywh, pred_conf in low_preds:
                    if int(pred_cls) != int(gt_cls):
                        continue
                    iou = self._iou_xyxy(gt_xyxy, self._xywh_to_xyxy(pred_xywh))
                    if iou < iou_thr:
                        continue
                    cost = alpha * (1.0 - float(pred_conf)) + beta * (1.0 - iou)
                    if cost < best_cost:
                        best_cost = cost

                obj_scores.append(float(best_cost))
                bbox = BoundingBox(
                    x_center=float(gt_xywh[0]),
                    y_center=float(gt_xywh[1]),
                    width=float(gt_xywh[2]),
                    height=float(gt_xywh[3]),
                )
                object_details.append(
                    ObjectDifficultyProperties(
                        image_path=self._resolve_image_path(stem, images_dir),
                        object_id=obj_id,
                        class_id=int(gt_cls),
                        bounding_box=bbox,
                        difficulty_score=float(best_cost),
                    )
                )

            avg_obj = float(sum(obj_scores) / len(obj_scores)) if obj_scores else 0.0
            fp_rate = self._false_positive_rate(
                gt_boxes=gt_boxes,
                preds=opt_preds,
                iou_threshold=iou_thr,
            )
            missed_rate = self._missed_detections_rate(
                gt_boxes=gt_boxes,
                preds=opt_preds,
                iou_threshold=iou_thr,
            )
            image_path = self._resolve_image_path(stem, images_dir)
            per_image_rows.append(
                {
                    "image_path": image_path,
                    "num_objects": len(gt_boxes),
                    "objects_score": object_details,
                    "avg_object_score": avg_obj,
                    "false_positive_rate": fp_rate,
                    "missed_detections_rate": missed_rate,
                }
            )

        auto_w2 = self._select_false_positive_weight(
            image_rows=per_image_rows,
            object_weight=w1,
            fixed_false_positive_weight=configured_w2,
            weight_mode=weight_mode,
        )

        image_difficulties: List[ImageDifficultyProperties] = []
        for row in per_image_rows:
            s_img = float(
                w1 * float(row["avg_object_score"])
                + auto_w2 * float(row["false_positive_rate"])
            )
            image_difficulties.append(
                ImageDifficultyProperties(
                    image_path=str(row["image_path"]),
                    difficulty_score=s_img,
                    num_objects=int(row["num_objects"]),
                    objects_score=row["objects_score"],
                    false_positive_rate=float(row["false_positive_rate"]),
                    missed_detections_rate=float(row["missed_detections_rate"]),
                )
            )

        return ScoringResults(
            image_difficulties=image_difficulties,
            scoring_weight_mode=weight_mode,
            selected_object_weight=float(w1),
            selected_false_positive_weight=float(auto_w2),
        )

    def _select_false_positive_weight(
        self,
        image_rows: List[Dict],
        object_weight: float,
        fixed_false_positive_weight: float,
        weight_mode: str,
    ) -> float:
        if weight_mode == "fixed":
            return float(fixed_false_positive_weight)

        if not image_rows:
            return 0.0

        avg_obj = [float(row["avg_object_score"]) for row in image_rows]
        fp_rate = [float(row["false_positive_rate"]) for row in image_rows]

        mean_avg_obj = sum(avg_obj) / len(avg_obj) if avg_obj else 0.0
        mean_fp_rate = sum(fp_rate) / len(fp_rate) if fp_rate else 0.0
        base_w2 = float(object_weight * mean_avg_obj / mean_fp_rate) if mean_fp_rate > 0 else 0.0

        if weight_mode == "mean_match":
            return float(base_w2)

        # balance_correlation: minimize |corr(score, miss_rate) - corr(score, fp_rate)|,
        # then maximize corr(score, miss_rate) + corr(score, fp_rate).
        miss_rate = [float(row["missed_detections_rate"]) for row in image_rows]
        max_w2 = max(1.0, base_w2 * 5.0)

        best_w2 = float(base_w2)
        best_gap = float("inf")
        best_neg_corr_sum = float("inf")

        for step in range(401):
            w2 = float(max_w2 * step / 400.0)
            score = [float(object_weight) * a + w2 * f for a, f in zip(avg_obj, fp_rate)]
            corr_miss = self._pearson_corr(score, miss_rate)
            corr_fp = self._pearson_corr(score, fp_rate)
            gap = abs(corr_miss - corr_fp)
            neg_corr_sum = -1.0 * (corr_miss + corr_fp)

            if gap < best_gap or (gap == best_gap and neg_corr_sum < best_neg_corr_sum):
                best_w2 = w2
                best_gap = gap
                best_neg_corr_sum = neg_corr_sum

        return float(best_w2)

    @staticmethod
    def _pearson_corr(x: List[float], y: List[float]) -> float:
        if len(x) < 2 or len(y) < 2:
            return 0.0

        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        var_x = sum((xi - mean_x) ** 2 for xi in x)
        var_y = sum((yi - mean_y) ** 2 for yi in y)
        if var_x <= 0.0 or var_y <= 0.0:
            return 0.0

        cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        return float(cov / ((var_x ** 0.5) * (var_y ** 0.5)))

    @staticmethod
    def _read_yolo_file(path: Path, has_conf: bool) -> List[Tuple[int, Tuple[float, float, float, float], float]]:
        rows: List[Tuple[int, Tuple[float, float, float, float], float]] = []
        if not path.exists():
            return rows

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if has_conf:
                    if len(parts) < 6:
                        continue
                    cls = int(float(parts[0]))
                    x_c, y_c, w, h, conf = map(float, parts[1:6])
                else:
                    if len(parts) < 5:
                        continue
                    cls = int(float(parts[0]))
                    x_c, y_c, w, h = map(float, parts[1:5])
                    conf = 1.0

                rows.append((cls, (x_c, y_c, w, h), conf))
        return rows

    @staticmethod
    def _xywh_to_xyxy(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        x_c, y_c, w, h = box
        return (x_c - w / 2.0, y_c - h / 2.0, x_c + w / 2.0, y_c + h / 2.0)

    @staticmethod
    def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0.0:
            return 0.0
        return float(inter / union)

    def _false_positive_rate(
        self,
        gt_boxes: List[Tuple[int, Tuple[float, float, float, float], float]],
        preds: List[Tuple[int, Tuple[float, float, float, float], float]],
        iou_threshold: float,
    ) -> float:
        if not preds:
            return 0.0

        gt_xyxy = [(cls, self._xywh_to_xyxy(box)) for cls, box, _ in gt_boxes]
        pred_xyxy = [(cls, self._xywh_to_xyxy(box)) for cls, box, _ in preds]

        candidate_pairs: List[Tuple[float, int, int]] = []
        for p_idx, (p_cls, p_box) in enumerate(pred_xyxy):
            for g_idx, (g_cls, g_box) in enumerate(gt_xyxy):
                if int(p_cls) != int(g_cls):
                    continue
                iou = self._iou_xyxy(p_box, g_box)
                if iou >= iou_threshold:
                    candidate_pairs.append((iou, p_idx, g_idx))

        candidate_pairs.sort(reverse=True, key=lambda x: x[0])
        matched_preds = set()
        matched_gts = set()
        for _, p_idx, g_idx in candidate_pairs:
            if p_idx in matched_preds or g_idx in matched_gts:
                continue
            matched_preds.add(p_idx)
            matched_gts.add(g_idx)

        false_positives = len(preds) - len(matched_preds)
        # Normalize by prediction count so FP rate is bounded to [0, 1].
        denom = max(1, len(preds))
        return float(false_positives / denom)

    def _missed_detections_rate(
        self,
        gt_boxes: List[Tuple[int, Tuple[float, float, float, float], float]],
        preds: List[Tuple[int, Tuple[float, float, float, float], float]],
        iou_threshold: float,
    ) -> float:
        if not gt_boxes:
            return 0.0

        gt_xyxy = [(cls, self._xywh_to_xyxy(box)) for cls, box, _ in gt_boxes]
        pred_xyxy = [(cls, self._xywh_to_xyxy(box)) for cls, box, _ in preds]

        candidate_pairs: List[Tuple[float, int, int]] = []
        for p_idx, (p_cls, p_box) in enumerate(pred_xyxy):
            for g_idx, (g_cls, g_box) in enumerate(gt_xyxy):
                if int(p_cls) != int(g_cls):
                    continue
                iou = self._iou_xyxy(p_box, g_box)
                if iou >= iou_threshold:
                    candidate_pairs.append((iou, p_idx, g_idx))

        candidate_pairs.sort(reverse=True, key=lambda x: x[0])
        matched_preds = set()
        matched_gts = set()
        for _, p_idx, g_idx in candidate_pairs:
            if p_idx in matched_preds or g_idx in matched_gts:
                continue
            matched_preds.add(p_idx)
            matched_gts.add(g_idx)

        missed_detections = len(gt_boxes) - len(matched_gts)
        denom = max(1, len(gt_boxes))
        return float(missed_detections / denom)

    @staticmethod
    def _resolve_image_path(stem: str, train_images_dir: str | None) -> str:
        if not train_images_dir:
            return stem

        image_dir = Path(train_images_dir)
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            candidate = image_dir / f"{stem}{ext}"
            if candidate.exists():
                return str(candidate)
        return str(image_dir / stem)
