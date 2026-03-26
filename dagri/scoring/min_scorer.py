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

    def score(self, optimal_conf_threshold_prediction_dir: str, low_conf_prediction_dir: str, dataset_properties) -> ScoringResults:
        """
        Score every image using object-level miss cost and image-level false-positive penalty.

        S_obj = min_pred(alpha*(1-conf) + beta*(1-IoU)) over predictions with IoU >= iou_threshold.
        S_img = w1*avg(S_obj) + w2*FP_rate.
        """
        train_labels_dir = getattr(dataset_properties, "train_labels_dir", None)
        train_images_dir = getattr(dataset_properties, "train_images_dir", None)
        if not train_labels_dir:
            raise ValueError("dataset_properties.train_labels_dir is required for scoring")

        low_dir = Path(low_conf_prediction_dir)
        opt_dir = Path(optimal_conf_threshold_prediction_dir)
        label_dir = Path(train_labels_dir)

        if not label_dir.exists():
            raise FileNotFoundError(f"Ground-truth label directory not found: {label_dir}")
        if not low_dir.exists():
            raise FileNotFoundError(f"Low-confidence prediction directory not found: {low_dir}")
        if not opt_dir.exists():
            raise FileNotFoundError(f"Optimal-threshold prediction directory not found: {opt_dir}")

        alpha = float(self.scoring_config.alpha)
        beta = float(self.scoring_config.beta)
        iou_thr = float(self.scoring_config.iou_threshold)
        w1 = float(self.scoring_config.object_weight)
        # Kept for backward compatibility with existing config key naming.
        w2 = float(self.scoring_config.false_positive_weight)

        image_difficulties: List[ImageDifficultyProperties] = []

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
                        image_path=self._resolve_image_path(stem, train_images_dir),
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
            s_img = float(w1 * avg_obj + w2 * fp_rate)

            image_path = self._resolve_image_path(stem, train_images_dir)
            image_difficulties.append(
                ImageDifficultyProperties(
                    image_path=image_path,
                    difficulty_score=s_img,
                    num_objects=len(gt_boxes),
                    objects_score=object_details,
                    false_positive_rate=fp_rate,
                    missed_detections_rate=missed_rate,
                )
            )

        return ScoringResults(image_difficulties=image_difficulties)

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
        denom = max(1, len(gt_boxes))
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
