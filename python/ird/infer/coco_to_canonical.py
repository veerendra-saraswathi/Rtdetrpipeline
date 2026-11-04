"""E-00 helper: convert COCO val2017 annotations to the canonical format, so the released COCO
checkpoint can go through our export -> ONNX -> TensorRT -> evaluator path.

    python -m ird.infer.coco_to_canonical --coco-ann <coco>/annotations/instances_val2017.json \
        --img-dir val2017 -o data/coco_val2017_canonical.json

Category ids become 0..79 in the official RT-DETR label order (mscoco_category2label), matching
the model's class indices. Images go in files.vis (8-bit RGB; grey images are repeated to RGB).
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-ann", required=True)
    ap.add_argument("--img-dir", required=True, help="image folder relative to --data-root used later")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N images only (quick checks)")
    args = ap.parse_args()
    d = json.load(open(args.coco_ann))
    cat_ids = sorted(c["id"] for c in d["categories"])          # same order as mscoco_category2label
    remap = {cid: i for i, cid in enumerate(cat_ids)}
    names = {c["id"]: c["name"] for c in d["categories"]}
    images = d["images"][: args.limit] if args.limit else d["images"]
    keep = {i["id"] for i in images}
    out = {
        "info": {"source": "COCO val2017", "split": "val2017"},
        "categories": [{"id": remap[c], "name": names[c]} for c in cat_ids],
        "images": [{"id": i["id"], "file_name": f"{args.img_dir}/{i['file_name']}",
                    "files": {"ir": None, "vis": f"{args.img_dir}/{i['file_name']}"},
                    "width": i["width"], "height": i["height"], "bit_depth": {"vis": 8}} for i in images],
        "annotations": [{"id": a["id"], "image_id": a["image_id"], "category_id": remap[a["category_id"]],
                         "bbox": a["bbox"], "area": a["area"], "iscrowd": a.get("iscrowd", 0),
                         "ignore": a.get("iscrowd", 0)} for a in d["annotations"] if a["image_id"] in keep],
    }
    json.dump(out, open(args.output, "w"))
    print(f"{len(out['images'])} images, {len(out['annotations'])} annotations -> {args.output}")


if __name__ == "__main__":
    main()
