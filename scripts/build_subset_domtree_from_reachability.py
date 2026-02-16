# scripts/build_subset_domtree_from_reachability.py
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def copy_or_link(src: Path, dst: Path, prefer_hardlink: bool = True) -> None:
    if dst.exists():
        return
    ensure_dir(dst.parent)
    if prefer_hardlink:
        try:
            dst.hardlink_to(src)  # NTFS usually supports this
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset_reach", type=str, default="data/subset_reachability")
    ap.add_argument("--raw_domtree_labels", type=str,
                    default="data/deepdataflow_raw/labels_domtree_20.06.01/dataflow/labels/domtree")
    ap.add_argument("--out_subset_domtree", type=str, default="data/subset_domtree")
    ap.add_argument("--no_hardlink", action="store_true", help="use copy instead of hardlink")
    args = ap.parse_args()

    subset_reach = Path(args.subset_reach)
    raw_dom = Path(args.raw_domtree_labels)
    out_sub = Path(args.out_subset_domtree)
    prefer_hardlink = (not args.no_hardlink)

    if not subset_reach.exists():
        raise FileNotFoundError(f"subset_reach not found: {subset_reach}")
    if not raw_dom.exists():
        raise FileNotFoundError(f"raw domtree labels not found: {raw_dom}")

    splits = ["train", "val", "test"]
    ok_graphs = 0
    ok_labels = 0
    miss_labels = 0

    for sp in splits:
        in_g = subset_reach / sp / "graphs"
        in_l = subset_reach / sp / "labels"
        if not in_g.exists() or not in_l.exists():
            raise FileNotFoundError(f"missing in subset_reach: {in_g} or {in_l}")

        out_g = out_sub / sp / "graphs"
        out_l = out_sub / sp / "labels"
        ensure_dir(out_g); ensure_dir(out_l)

        graph_files = sorted(in_g.glob("*.ProgramGraph.pb"))
        if len(graph_files) == 0:
            raise RuntimeError(f"no graphs under {in_g}")

        for gpath in graph_files:
            # key = "<something>.ProgramGraph"
            stem = gpath.name
            if not stem.endswith(".ProgramGraph.pb"):
                continue
            key = stem[:-len(".ProgramGraph.pb")]  # remove suffix
            # expected domtree label file name:
            # <key>.ProgramGraphFeaturesList.pb
            dom_label = raw_dom / f"{key}.ProgramGraphFeaturesList.pb"
            if not dom_label.exists():
                miss_labels += 1
                continue

            # copy/link graph + label into subset_domtree
            copy_or_link(gpath, out_g / gpath.name, prefer_hardlink=prefer_hardlink)
            copy_or_link(dom_label, out_l / dom_label.name, prefer_hardlink=prefer_hardlink)

            ok_graphs += 1
            ok_labels += 1

    print("=== build_subset_domtree_from_reachability DONE ===")
    print(f"out_subset_domtree: {out_sub}")
    print(f"copied/linked pairs: {ok_graphs}")
    print(f"missing domtree labels: {miss_labels}")
    if miss_labels > 0:
        print("[WARN] Some graphs in subset_reachability do not have matching domtree labels; they were skipped.")

if __name__ == "__main__":
    main()
