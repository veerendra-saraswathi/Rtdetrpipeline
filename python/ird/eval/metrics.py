"""IRD metrics. One implementation for every arm and every backend (PyTorch, ONNX, TensorRT).

Definitions (engineering doc v0.3, section 10; RCI answers of 29-10-2026)
-----------------------------------------------------------------------
Size of a box: s = sqrt(w * h) in original pixels.
Ignored ground truth (GT): s < min_size (default 16) or ignore flag set. Out of scope for now.

RCI metrics, at one operating threshold tau (detections with score >= tau):
  Matching   one-to-one, maximising total IoU (Hungarian), only pairs with IoU >= match_iou
             (default 0.5). Class-aware: a detection only matches GT of its class.
             Class-agnostic results are also reported (RCI: classification not discussed).
             Unmatched detections that overlap an ignored GT (IoU >= match_iou) are ignored.
  Pd         matched GT / all non-ignored GT                                    (RCI: > 95%)
  AO         sum of IoU of matched GT / all non-ignored GT (missed GT count 0)  (RCI: > 90%)
  FAR        unmatched, non-ignored detections / frames                          (RCI: 1e-5 per frame)
             far_upper95: if FP = 0, 3 / frames (rule of three); else Poisson upper 95% bound.
  CS         mean score of matched detections (RCI: > 0.9) + calibration error (ECE, 10 bins).
  mean_iou_tp  mean IoU of matched GT.

COCO-style AP (for comparison with the literature), per class then averaged over classes with GT:
  greedy matching by score, IoU thresholds 0.50:0.05:0.95, 101-point interpolation, max 100
  detections per image. "all" range = s >= min_size; buckets 16-32, 32-96, 96+.
  GT outside the range are ignored; unmatched detections outside the range are ignored (COCO rule).
"""
import math
from collections import defaultdict

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None

IOU_THRS = np.round(np.arange(0.5, 0.951, 0.05), 2)
REC_THRS = np.linspace(0.0, 1.0, 101)
BUCKETS = {"16-32": (16, 32), "32-96": (32, 96), "96+": (96, math.inf)}


# ----------------------------------------------------------------------------- helpers
def box_size(b):
    return math.sqrt(max(b[2], 0.0) * max(b[3], 0.0))


def iou_matrix(d, g):
    """d: (N,4) xywh, g: (M,4) xywh -> (N,M) IoU."""
    if len(d) == 0 or len(g) == 0:
        return np.zeros((len(d), len(g)))
    d = np.asarray(d, float); g = np.asarray(g, float)
    dx1, dy1 = d[:, 0] + d[:, 2], d[:, 1] + d[:, 3]
    gx1, gy1 = g[:, 0] + g[:, 2], g[:, 1] + g[:, 3]
    iw = np.clip(np.minimum(dx1[:, None], gx1[None]) - np.maximum(d[:, 0][:, None], g[:, 0][None]), 0, None)
    ih = np.clip(np.minimum(dy1[:, None], gy1[None]) - np.maximum(d[:, 1][:, None], g[:, 1][None]), 0, None)
    inter = iw * ih
    union = (d[:, 2] * d[:, 3])[:, None] + (g[:, 2] * g[:, 3])[None] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def one_to_one(iou, thr):
    """Max-total-IoU one-to-one matching restricted to IoU >= thr. Returns list of (d, g)."""
    if iou.size == 0:
        return []
    valid = iou >= thr
    if not valid.any():
        return []
    if linear_sum_assignment is not None:
        cost = np.where(valid, -iou, 1e3)
        r, c = linear_sum_assignment(cost)
        return [(int(a), int(b)) for a, b in zip(r, c) if valid[a, b]]
    pairs = sorted(((iou[a, b], a, b) for a, b in zip(*np.nonzero(valid))), reverse=True)
    used_d, used_g, out = set(), set(), []
    for _, a, b in pairs:
        if a not in used_d and b not in used_g:
            used_d.add(a); used_g.add(b); out.append((int(a), int(b)))
    return out


def poisson_upper95(k, n):
    """Upper 95% bound of a per-frame rate with k events in n frames."""
    if n == 0:
        return float("nan")
    if k == 0:
        return 3.0 / n
    try:
        from scipy.stats import chi2
        return float(chi2.ppf(0.95, 2 * (k + 1)) / 2.0 / n)
    except ImportError:  # normal approximation
        return float((k + 1.645 * math.sqrt(k) + 1) / n)


# ----------------------------------------------------------------------------- data
class EvalData:
    """GT and detections indexed by image, restricted to the predicted image list."""

    def __init__(self, gt_doc, pred_doc, min_size=16.0):
        self.min_size = float(min_size)
        self.categories = {c["id"]: c["name"] for c in gt_doc["categories"]}
        gt_ids = {i["id"] for i in gt_doc["images"]}
        self.image_ids = [i for i in pred_doc.get("image_ids") or sorted(gt_ids)]
        missing = [i for i in self.image_ids if i not in gt_ids]
        if missing:
            raise ValueError(f"{len(missing)} predicted images are not in the GT file, e.g. {missing[:3]}")
        keep = set(self.image_ids)
        self.gt = defaultdict(list)
        for a in gt_doc["annotations"]:
            if a["image_id"] in keep:
                ign = bool(a.get("ignore", 0)) or box_size(a["bbox"]) < self.min_size
                self.gt[a["image_id"]].append((a["category_id"], list(a["bbox"]), ign))
        self.dt = defaultdict(list)
        for d in pred_doc["detections"]:
            if d["image_id"] in keep:
                self.dt[d["image_id"]].append((d["category_id"], list(d["bbox"]), float(d["score"])))
        self.gt_classes = sorted({c for v in self.gt.values() for c, _, ig in v if not ig})


# ----------------------------------------------------------------------------- RCI metrics
def match_image(gts, dts, tau, match_iou, class_aware=True, size_range=None):
    """Per-image matching at threshold tau. size_range limits which GT count (others ignored).
    Returns dict with n_gt, tp, iou_sum, fp, matched_scores, fp_scores, per-class/bucket parts."""
    dts = [d for d in dts if d[2] >= tau]
    gl = []
    for c, b, ig in gts:
        s = box_size(b)
        out = size_range is not None and not (size_range[0] <= s < size_range[1])
        gl.append((c, b, ig or out))
    groups = sorted({c for c, _, _ in gl} | {c for c, _, _ in dts}) if class_aware else [None]
    res = {"n_gt": 0, "tp": 0, "iou_sum": 0.0, "fp": 0, "fp_ge_min": 0, "tp_scores": [], "fp_scores": [],
           "per_class": defaultdict(lambda: [0, 0, 0.0, 0])}  # n_gt, tp, iou_sum, fp
    for cls in groups:
        G = [g for g in gl if cls is None or g[0] == cls]
        D = [d for d in dts if cls is None or d[0] == cls]
        Gn = [g for g in G if not g[2]]
        Gi = [g for g in G if g[2]]
        iou = iou_matrix([d[1] for d in D], [g[1] for g in Gn])
        pairs = one_to_one(iou, match_iou)
        md = {a for a, _ in pairs}
        iou_ign = iou_matrix([d[1] for d in D], [g[1] for g in Gi])
        key = cls if cls is not None else "all"
        pc = res["per_class"][key]
        pc[0] += len(Gn); pc[1] += len(pairs); pc[2] += float(sum(iou[a, b] for a, b in pairs))
        res["n_gt"] += len(Gn); res["tp"] += len(pairs)
        res["iou_sum"] += float(sum(iou[a, b] for a, b in pairs))
        res["tp_scores"] += [D[a][2] for a, _ in pairs]
        for k, d in enumerate(D):
            if k in md:
                continue
            if iou_ign.shape[1] and iou_ign[k].max() >= match_iou:
                continue
            res["fp"] += 1; pc[3] += 1
            res["fp_scores"].append(d[2])
            if box_size(d[1]) >= 16.0:
                res["fp_ge_min"] += 1
    return res


def rci_metrics(data, tau=0.5, match_iou=0.5, class_aware=True):
    per_img = {}
    tot = defaultdict(float)
    per_class = defaultdict(lambda: np.zeros(4))
    tp_scores, fp_scores = [], []
    for i in data.image_ids:
        r = match_image(data.gt[i], data.dt[i], tau, match_iou, class_aware)
        per_img[i] = (r["n_gt"], r["tp"], r["iou_sum"], r["fp"])
        for k in ("n_gt", "tp", "iou_sum", "fp", "fp_ge_min"):
            tot[k] += r[k]
        for k, v in r["per_class"].items():
            per_class[k] += np.array(v)
        tp_scores += r["tp_scores"]; fp_scores += r["fp_scores"]
    n_img = len(data.image_ids)
    out = summarize_counts(tot["n_gt"], tot["tp"], tot["iou_sum"], tot["fp"], n_img)
    out["fp_frames_ge16"] = tot["fp_ge_min"] / n_img if n_img else float("nan")
    out["cs_mean_tp_score"] = float(np.mean(tp_scores)) if tp_scores else float("nan")
    out["ece"] = ece(tp_scores, fp_scores)
    out["per_class"] = {
        (data.categories.get(k, str(k)) if k != "all" else "all"):
            summarize_counts(v[0], v[1], v[2], v[3], n_img) for k, v in per_class.items()}
    # by GT size bucket (detections never count as FP here: buckets are about Pd and AO)
    out["by_size"] = {}
    for name, rng in BUCKETS.items():
        n = tp = s = 0.0
        for i in data.image_ids:
            r = match_image(data.gt[i], data.dt[i], tau, match_iou, class_aware, size_range=rng)
            n += r["n_gt"]; tp += r["tp"]; s += r["iou_sum"]
        out["by_size"][name] = {"n_gt": int(n), "pd": tp / n if n else float("nan"),
                                "ao": s / n if n else float("nan")}
    return out, per_img


def summarize_counts(n_gt, tp, iou_sum, fp, n_img):
    return {"n_gt": int(n_gt), "tp": int(tp), "fp": int(fp), "frames": int(n_img),
            "pd": tp / n_gt if n_gt else float("nan"),
            "ao": iou_sum / n_gt if n_gt else float("nan"),
            "mean_iou_tp": iou_sum / tp if tp else float("nan"),
            "far_per_frame": fp / n_img if n_img else float("nan"),
            "far_upper95": poisson_upper95(int(fp), int(n_img))}


def ece(tp_scores, fp_scores, bins=10):
    s = np.array(list(tp_scores) + list(fp_scores), float)
    y = np.array([1] * len(tp_scores) + [0] * len(fp_scores), float)
    if s.size == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (s >= lo) & ((s < hi) if hi < 1 else (s <= hi))
        if m.any():
            e += m.mean() * abs(s[m].mean() - y[m].mean())
    return float(e)


def operating_curve(data, taus=None, match_iou=0.5, class_aware=True):
    taus = taus if taus is not None else [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
    rows = []
    for t in taus:
        n = tp = s = fp = 0.0
        for i in data.image_ids:
            r = match_image(data.gt[i], data.dt[i], t, match_iou, class_aware)
            n += r["n_gt"]; tp += r["tp"]; s += r["iou_sum"]; fp += r["fp"]
        rows.append({"tau": t, "pd": tp / n if n else float("nan"), "ao": s / n if n else float("nan"),
                     "far_per_frame": fp / len(data.image_ids)})
    return rows


# ----------------------------------------------------------------------------- COCO-style AP
def _eval_img_cls(gts, dts, rng, max_dets=100):
    """COCO evaluateImg for one image and class. gts: list (bbox, ignore). dts: list (bbox, score).
    Returns (scores, dtm[T,D] bool, dtIg[T,D] bool, n_non_ignored_gt)."""
    g_ign = np.array([ig or not (rng[0] <= box_size(b) < rng[1]) for b, ig in gts], bool)
    gord = np.argsort(g_ign, kind="mergesort")
    gb = [gts[k][0] for k in gord]; g_ign = g_ign[gord]
    dord = np.argsort([-s for _, s in dts], kind="mergesort")[:max_dets]
    db = [dts[k][0] for k in dord]; ds = np.array([dts[k][1] for k in dord])
    T, D, G = len(IOU_THRS), len(db), len(gb)
    dtm = np.zeros((T, D), bool); dtig = np.zeros((T, D), bool)
    ious = iou_matrix(db, gb)
    for ti, t in enumerate(IOU_THRS):
        gtm = np.zeros(G, bool)
        for d in range(D):
            best, m = min(t, 1 - 1e-10), -1
            for g in range(G):
                if gtm[g]:
                    continue
                if m > -1 and not g_ign[m] and g_ign[g]:
                    break
                if ious[d, g] < best:
                    continue
                best, m = ious[d, g], g
            if m == -1:
                continue
            dtig[ti, d] = g_ign[m]; dtm[ti, d] = True; gtm[m] = True
    d_out = np.array([not (rng[0] <= box_size(b) < rng[1]) for b in db], bool)
    dtig |= (~dtm) & d_out[None, :]
    return ds, dtm, dtig, int((~g_ign).sum())


def _ap_from(scores, dtm, dtig, npig):
    """Accumulate like COCOeval. Returns AP per IoU threshold (array T) or None if npig == 0."""
    if npig == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    dtm, dtig = dtm[:, order], dtig[:, order]
    tps = np.logical_and(dtm, ~dtig).cumsum(axis=1).astype(float)
    fps = np.logical_and(~dtm, ~dtig).cumsum(axis=1).astype(float)
    aps = np.zeros(len(IOU_THRS))
    for t in range(len(IOU_THRS)):
        tp, fp = tps[t], fps[t]
        if tp.size == 0:
            aps[t] = 0.0
            continue
        rc = tp / npig
        pr = tp / np.maximum(tp + fp, np.spacing(1))
        pr = pr.tolist()
        for k in range(len(pr) - 1, 0, -1):
            if pr[k] > pr[k - 1]:
                pr[k - 1] = pr[k]
        inds = np.searchsorted(rc, REC_THRS, side="left")
        q = np.zeros(len(REC_THRS))
        for ri, pi in enumerate(inds):
            if pi < len(pr):
                q[ri] = pr[pi]
        aps[t] = q.mean()
    return aps


def coco_ap(data, rng=None, per_image=False):
    """Returns dict with AP, AP50, AP75 averaged over classes with GT, and per class.
    With per_image=True also returns the per-image records needed for bootstrap."""
    rng = rng or (data.min_size, math.inf)
    per_cls, records = {}, defaultdict(dict)
    for c in data.gt_classes:
        S, M, I, N = [], [], [], 0
        for i in data.image_ids:
            gts = [(b, ig) for cc, b, ig in data.gt[i] if cc == c]
            dts = [(b, s) for cc, b, s in data.dt[i] if cc == c]
            s, m, ig, n = _eval_img_cls(gts, dts, rng)
            S.append(s); M.append(m); I.append(ig); N += n
            if per_image:
                records[c][i] = (s, m, ig, n)
        aps = _ap_from(np.concatenate(S) if S else np.zeros(0),
                       np.concatenate(M, axis=1) if M else np.zeros((len(IOU_THRS), 0), bool),
                       np.concatenate(I, axis=1) if I else np.zeros((len(IOU_THRS), 0), bool), N)
        if aps is not None:
            per_cls[data.categories.get(c, str(c))] = {"AP": float(aps.mean()), "AP50": float(aps[0]),
                                                        "AP75": float(aps[5]), "n_gt": N}
    out = {k: (float(np.mean([v[k] for v in per_cls.values()])) if per_cls else float("nan"))
           for k in ("AP", "AP50", "AP75")}
    out["per_class"] = per_cls
    return (out, records) if per_image else out


def ap_from_records(records, image_ids):
    """Class-mean AP and AP50 for a list of (possibly repeated) image ids. Used by bootstrap."""
    aps = []
    for c, per in records.items():
        S = [per[i][0] for i in image_ids]
        aps_c = _ap_from(np.concatenate(S),
                         np.concatenate([per[i][1] for i in image_ids], axis=1),
                         np.concatenate([per[i][2] for i in image_ids], axis=1),
                         sum(per[i][3] for i in image_ids))
        if aps_c is not None:
            aps.append(aps_c)
    if not aps:
        return float("nan"), float("nan")
    a = np.mean(aps, axis=0)
    return float(a.mean()), float(a[0])


# ----------------------------------------------------------------------------- bootstrap
def bootstrap(per_img, records, image_ids, n_boot=1000, seed=0, with_ap=True):
    """95% CIs over images for Pd, AO, FAR (and AP, AP50 if with_ap)."""
    rng = np.random.default_rng(seed)
    ids = list(image_ids)
    arr = np.array([per_img[i] for i in ids], float)  # n_gt, tp, iou_sum, fp
    keys = ["pd", "ao", "far_per_frame"] + (["AP", "AP50"] if with_ap else [])
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(ids), len(ids))
        s = arr[idx].sum(axis=0)
        samples["pd"].append(s[1] / s[0] if s[0] else np.nan)
        samples["ao"].append(s[2] / s[0] if s[0] else np.nan)
        samples["far_per_frame"].append(s[3] / len(ids))
        if with_ap:
            ap, ap50 = ap_from_records(records, [ids[k] for k in idx])
            samples["AP"].append(ap); samples["AP50"].append(ap50)
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in samples.items()}
