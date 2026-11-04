"""Locate the RT-DETRv2 code (official repo, Apache-2.0) and put it on sys.path.

Order: --rtdetr argument, then $RTDETR_ROOT, then third_party/RT-DETR/rtdetrv2_pytorch.
Get it with:  git clone https://github.com/lyuwenyu/RT-DETR third_party/RT-DETR
(record the commit you used in the experiment log).
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def add_rtdetr(path=None):
    cands = [path, os.environ.get("RTDETR_ROOT"),
             os.path.join(REPO, "third_party", "RT-DETR", "rtdetrv2_pytorch")]
    for c in cands:
        if c and os.path.isdir(os.path.join(c, "src", "zoo")):
            c = os.path.abspath(c)
            if c not in sys.path:
                sys.path.insert(0, c)
            # Windows DataLoader workers are new processes: pass the paths on through the environment.
            parts = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
            for extra in (c, os.path.join(REPO, "python")):
                if extra not in parts:
                    parts.insert(0, extra)
            os.environ["PYTHONPATH"] = os.pathsep.join(parts)
            return c
    sys.exit("RT-DETRv2 code not found. Clone https://github.com/lyuwenyu/RT-DETR into "
             "third_party/RT-DETR or set RTDETR_ROOT to its rtdetrv2_pytorch folder.")


def rtdetr_commit(root):
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"
