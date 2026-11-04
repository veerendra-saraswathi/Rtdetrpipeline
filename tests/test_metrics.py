"""Evaluator tests: hand-checked RCI metrics, and COCO-style AP against pycocotools.

Run: PYTHONPATH=python python tests/test_metrics.py
"""
import contextlib
import io
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from ird.eval.metrics import EvalData, coco_ap, rci_metrics  # noqa: E402

CATS = [{"id": 0, "name": "ship"}, {"id": 3, "name": "civil_vehicle"}]


def doc(images, anns):
    return {"categories": CATS, "images": [{"id": i, "width": 640, "height": 512} for i in images],
            "annotations": [dict(id=k + 1, image_id=i, category_id=c, bbox=b, area=b[2] * b[3],
                                 iscrowd=ig, ignore=ig) for k, (i, c, b, ig) in enumerate(anns)]}


def preds(images, dets):
    return {"image_ids": images, "detections": [dict(image_id=i, category_id=c, bbox=b, score=s)
                                                 for i, c, b, s in dets]}


def approx(a, b, tol=1e-9):
    assert abs(a - b) <= tol, (a, b)


def test_rci_hand_case():
    # Frame 1: two ships (40x40). One found exactly, one found with a 10 px shift -> IoU 30*40/(50*40)=0.6.
    # Frame 2: one ship missed; one false alarm far away; one tiny ship (10x10, ignored) with a detection on it.
    gt = doc([1, 2], [(1, 0, [100, 100, 40, 40], 0), (1, 0, [300, 300, 40, 40], 0),
                      (2, 0, [50, 50, 40, 40], 0), (2, 0, [400, 400, 10, 10], 0)])
    pr = preds([1, 2], [(1, 0, [100, 100, 40, 40], 0.9), (1, 0, [310, 300, 40, 40], 0.8),
                        (2, 0, [500, 100, 40, 40], 0.7), (2, 0, [400, 400, 10, 10], 0.95),
                        (2, 0, [50, 50, 40, 40], 0.3)])          # below tau 0.5: not counted
    r, _ = rci_metrics(EvalData(gt, pr, 16), tau=0.5, match_iou=0.5)
    assert r["n_gt"] == 3 and r["tp"] == 2 and r["fp"] == 1
    approx(r["pd"], 2 / 3)
    approx(r["ao"], (1.0 + 0.6) / 3)        # missed target counts 0
    approx(r["mean_iou_tp"], 0.8)
    approx(r["far_per_frame"], 0.5)
    approx(r["cs_mean_tp_score"], 0.85)
    assert r["by_size"]["32-96"]["n_gt"] == 3


def test_class_aware_vs_agnostic():
    gt = doc([1], [(1, 0, [100, 100, 40, 40], 0)])
    pr = preds([1], [(1, 3, [100, 100, 40, 40], 0.9)])          # right place, wrong class
    aw, _ = rci_metrics(EvalData(gt, pr), 0.5, 0.5, class_aware=True)
    ag, _ = rci_metrics(EvalData(gt, pr), 0.5, 0.5, class_aware=False)
    assert aw["tp"] == 0 and aw["fp"] == 1
    assert ag["tp"] == 1 and ag["fp"] == 0


def test_one_to_one():
    # Two detections on one target: one hit, one false alarm (no double counting).
    gt = doc([1], [(1, 0, [100, 100, 40, 40], 0)])
    pr = preds([1], [(1, 0, [100, 100, 40, 40], 0.9), (1, 0, [102, 100, 40, 40], 0.8)])
    r, _ = rci_metrics(EvalData(gt, pr), 0.5, 0.5)
    assert r["tp"] == 1 and r["fp"] == 1


def test_far_bound_zero_fp():
    gt = doc(list(range(1, 301)), [])
    r, _ = rci_metrics(EvalData(gt, preds(list(range(1, 301)), [])), 0.5, 0.5)
    approx(r["far_upper95"], 3 / 300)


def random_case(seed, n_img=40):
    rng = np.random.default_rng(seed)
    anns, dets = [], []
    for i in range(1, n_img + 1):
        for _ in range(rng.integers(0, 8)):
            c = int(rng.choice([0, 3]))
            w, h = rng.uniform(8, 120, 2)
            x, y = rng.uniform(0, 500), rng.uniform(0, 400)
            anns.append((i, c, [float(x), float(y), float(w), float(h)], 0))
            if rng.random() < 0.8:   # a noisy detection of it
                j = rng.normal(0, 0.15 * min(w, h), 4)
                dets.append((i, c if rng.random() < 0.9 else 3 - c,
                             [float(x + j[0]), float(y + j[1]), float(max(2, w + j[2])), float(max(2, h + j[3]))],
                             float(rng.uniform(0.05, 1))))
        for _ in range(rng.integers(0, 4)):  # false alarms
            dets.append((i, int(rng.choice([0, 3])), [float(rng.uniform(0, 500)), float(rng.uniform(0, 400)),
                                                      float(rng.uniform(8, 80)), float(rng.uniform(8, 80))],
                         float(rng.uniform(0.05, 1))))
    return doc(list(range(1, n_img + 1)), anns), preds(list(range(1, n_img + 1)), dets)


def test_ap_matches_pycocotools():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    for seed in range(5):
        gt, pr = random_case(seed)
        ours = coco_ap(EvalData(gt, pr, min_size=0.0))
        with contextlib.redirect_stdout(io.StringIO()):
            cg = COCO(); cg.dataset = gt; cg.createIndex()
            cd = cg.loadRes([dict(d) for d in pr["detections"]])
            E = COCOeval(cg, cd, "bbox"); E.evaluate(); E.accumulate(); E.summarize()
        approx(ours["AP"], E.stats[0], 1e-6)
        approx(ours["AP50"], E.stats[1], 1e-6)
        approx(ours["AP75"], E.stats[2], 1e-6)


if __name__ == "__main__":
    test_rci_hand_case(); test_class_aware_vs_agnostic(); test_one_to_one(); test_far_bound_zero_fp()
    test_ap_matches_pycocotools()
    print("all metric tests passed")
