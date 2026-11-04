"""Write the frame list read by the C++ tool ird_infer (tab-separated, one frame per line).

    python -m ird.infer.make_list --ann data/airst_v0.1s/airst_test.json -o export/airst_test.tsv

Columns: image_id width height ir_path vis_path ir_bit_depth vis_bit_depth ('-' when absent).
Paths are relative to the dataset root given to ird_infer with --root.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()
    doc = json.load(open(args.ann))
    with open(args.output, "w", newline="\n") as fh:
        fh.write("# image_id\twidth\theight\tir_path\tvis_path\tir_bit_depth\tvis_bit_depth\n")
        for e in doc["images"]:
            f, bd = e["files"], e.get("bit_depth") or {}
            fh.write("\t".join(str(x) for x in [e["id"], e["width"], e["height"], f.get("ir") or "-",
                                               f.get("vis") or "-", bd.get("ir") or "-", bd.get("vis") or "-"]) + "\n")
    print(f"{len(doc['images'])} frames -> {args.output}")


if __name__ == "__main__":
    main()
