#!/usr/bin/env python3
"""Compute per-domain AP on GWHD train set using a trained YOLO model.

Outputs:
 - results/gwhd_train_domain_ap.csv
 - results/gwhd_train_domain_ap.png
 - results/gwhd_train_gt.json
 - results/gwhd_train_preds.json
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "global_wheat_head" / "yolo_format_v2"
TRAIN_IMG_DIR = DATA_ROOT / "train" / "images"
TRAIN_CSV = ROOT / "datasets" / "global_wheat_head" / "raw_v2" / "gwhd_2021" / "competition_train.csv"
WEIGHT = ROOT / "results" / "01_only_training_gwhd_2021" / "Step_2_Train_and_Evaluate_BASELINE_MODEL" / "train_results" / "best.pt"
OUT_DIR = ROOT / "results" / "gwhd_domain_ap"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def ensure_packages():
    import importlib
    needed = ["pycocotools", "ultralytics", "pandas", "tqdm", "matplotlib", "numpy", "Pillow"]
    to_install = []
    for p in needed:
        try:
            importlib.import_module(p)
        except Exception:
            to_install.append(p)
    if to_install:
        print("Installing missing packages:", to_install)
        os.system(f"pip install {' '.join(to_install)}")


def parse_csv_to_gt(csv_path, images_dir):
    import pandas as pd
    rows = pd.read_csv(csv_path)
    images = []
    annotations = []
    ann_id = 1
    img_id = 1
    img_name_to_id = {}
    for _, r in rows.iterrows():
        name = r['image_name']
        domain = r['domain'] if 'domain' in r else ''
        img_path = images_dir / name
        if not img_path.exists():
            continue
        from PIL import Image
        w,h = Image.open(img_path).size
        img = {"id": img_id, "file_name": name, "width": w, "height": h, "domain": domain}
        images.append(img)
        img_name_to_id[name] = img_id
        boxestr = r['BoxesString'] if not pd.isna(r['BoxesString']) else ''
        if isinstance(boxestr, str) and boxestr.lower().strip() not in ['', 'no_box']:
            parts = boxestr.split(';')
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                coords = [float(x) for x in p.split()]
                if len(coords) == 4:
                    x1,y1,x2,y2 = coords
                    bw = x2 - x1
                    bh = y2 - y1
                    ann = {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [x1, y1, bw, bh],
                        "area": bw*bh,
                        "iscrowd": 0,
                    }
                    annotations.append(ann)
                    ann_id += 1
        img_id += 1
    gt = {"images": images, "annotations": annotations, "categories": [{"id":1, "name":"wheat_head"}]}
    return gt, img_name_to_id


def run_inference(weight, image_paths):
    from ultralytics import YOLO
    model = YOLO(str(weight))
    # batch predict
    results = model.predict(source=list(map(str,image_paths)), conf=0.25, iou=0.45, max_det=1000, device=0)
    preds = []
    for res, img_path in zip(results, image_paths):
        img_name = Path(img_path).name
        dets = getattr(res, 'boxes', None)
        if dets is None or len(dets) == 0:
            continue
        for box in dets:
            xyxy = box.xyxy.tolist()[0]
            x1,y1,x2,y2 = xyxy
            w = x2 - x1
            h = y2 - y1
            score = float(box.conf.tolist()[0])
            preds.append({
                "image_id": img_name,
                "bbox": [float(x1), float(y1), float(w), float(h)],
                "score": score,
                "category_id": 1,
                "file_name": img_name,
            })
    return preds


def convert_preds_to_coco_format(preds, name_to_id):
    coco_preds = []
    for p in preds:
        name = p['image_id']
        if name not in name_to_id:
            continue
        coco_preds.append({
            "image_id": int(name_to_id[name]),
            "category_id": int(p['category_id']),
            "bbox": [p['bbox'][0], p['bbox'][1], p['bbox'][2], p['bbox'][3]],
            "score": float(p['score'])
        })
    return coco_preds


def evaluate_per_domain(gt_json, preds_json, images, out_csv, out_png):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import pandas as pd
    import matplotlib.pyplot as plt

    coco_gt = COCO(gt_json)
    coco_dt = coco_gt.loadRes(preds_json)
    domains = {}
    for img in images:
        dom = img.get('domain','')
        domains.setdefault(dom, []).append(img['id'])

    rows = []
    for dom, img_ids in domains.items():
        if len(img_ids) == 0:
            continue
        E = COCOeval(coco_gt, coco_dt, iouType='bbox')
        E.params.imgIds = img_ids
        E.evaluate()
        E.accumulate()
        stats = E.stats  # 12 metrics
        ap = float(stats[0]) if len(stats)>0 else 0.0
        rows.append((dom, ap, len(img_ids)))

    df = pd.DataFrame(rows, columns=['domain','AP','n_images']).sort_values('AP', ascending=False)
    df.to_csv(out_csv, index=False)

    # plot
    plt.figure(figsize=(10,6))
    plt.bar(df['domain'], df['AP'])
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('AP (mAP)')
    plt.ylim(0,1)
    plt.tight_layout()
    plt.savefig(out_png)
    print('Saved', out_csv, out_png)
    return df


def main():
    ensure_packages()
    import pandas as pd
    # parse GT
    gt, name_to_id = parse_csv_to_gt(TRAIN_CSV, TRAIN_IMG_DIR)
    gt_json = OUT_DIR / 'gwhd_train_gt.json'
    with open(gt_json, 'w') as f:
        json.dump(gt, f)

    # inference
    img_paths = [TRAIN_IMG_DIR / img['file_name'] for img in gt['images']]
    print('Running inference on', len(img_paths), 'images...')
    preds = run_inference(WEIGHT, img_paths)
    preds_json = OUT_DIR / 'gwhd_train_preds.json'
    coco_preds = convert_preds_to_coco_format(preds, name_to_id)
    with open(preds_json, 'w') as f:
        json.dump(coco_preds, f)

    out_csv = OUT_DIR / 'gwhd_train_domain_ap.csv'
    out_png = OUT_DIR / 'gwhd_train_domain_ap.png'
    df = evaluate_per_domain(str(gt_json), str(preds_json), gt['images'], str(out_csv), str(out_png))
    print(df)


if __name__ == '__main__':
    main()
