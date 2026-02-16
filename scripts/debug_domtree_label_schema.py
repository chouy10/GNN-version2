# scripts/debug_domtree_label_schema.py
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PB2_ROOT = ROOT / "src" / "programl_pb2"
if str(PB2_ROOT) not in sys.path:
    sys.path.insert(0, str(PB2_ROOT))

from programl.proto.util_pb2 import ProgramGraphFeaturesList
from programl.proto.program_graph_pb2 import ProgramGraph

def _len_feature_list(fl) -> int:
    # tensorflow FeatureList: has feature[]; each feature can have int64_list/float_list/bytes_list
    return len(fl.feature)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset_domtree", type=str, default="data/subset_domtree")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--key", type=str, default="", help="optional key without suffix")
    args = ap.parse_args()

    subset = Path(args.subset_domtree)
    labels_dir = subset / args.split / "labels"
    graphs_dir = subset / args.split / "graphs"

    if args.key:
        lp = labels_dir / f"{args.key}.ProgramGraphFeaturesList.pb"
        gp = graphs_dir / f"{args.key}.ProgramGraph.pb"
    else:
        files = sorted(labels_dir.glob("*.ProgramGraphFeaturesList.pb"))
        lp = files[args.index % len(files)]
        key = lp.name.replace(".ProgramGraphFeaturesList.pb", "")
        gp = graphs_dir / f"{key}.ProgramGraph.pb"

    key = lp.name.replace(".ProgramGraphFeaturesList.pb", "")
    print(f"[label] {lp}")
    print(f"[graph] {gp}")

    g = ProgramGraph()
    g.ParseFromString(gp.read_bytes())
    N = len(g.node)
    E = len(g.edge)
    print(f"Graph sizes: N={N} E={E}")

    ylist = ProgramGraphFeaturesList()
    ylist.ParseFromString(lp.read_bytes())
    steps = list(ylist.graph)
    print(f"Steps: {len(steps)}  (type={steps[0].DESCRIPTOR.full_name if steps else 'N/A'})")

    for si, st in enumerate(steps[:3]):  # show first 3 steps
        print(f"\n--- step {si} ---")
        # node_features
        nf = getattr(st, "node_features", None)
        if nf is not None and hasattr(nf, "feature_list"):
            keys = list(nf.feature_list.keys())
            print(f"node_features keys ({len(keys)}): {keys[:50]}")
            # show lengths vs N
            for k in keys[:50]:
                L = _len_feature_list(nf.feature_list[k])
                if L in (N, N+1, N-1):
                    print(f"  * key '{k}' has FeatureList length {L} (≈N)")
        else:
            print("no node_features.feature_list")

        ef = getattr(st, "edge_features", None)
        if ef is not None and hasattr(ef, "feature_list"):
            keys = list(ef.feature_list.keys())
            print(f"edge_features keys ({len(keys)}): {keys[:50]}")
            for k in keys[:50]:
                L = _len_feature_list(ef.feature_list[k])
                if L in (E, E+1, E-1):
                    print(f"  * key '{k}' has FeatureList length {L} (≈E)")
        else:
            print("no edge_features.feature_list")

        # also show scalar context feature keys if exist
        ctx = getattr(st, "features", None)
        if ctx is not None and hasattr(ctx, "feature"):
            ckeys = list(ctx.feature.keys())
            print(f"context features keys ({len(ckeys)}): {ckeys[:50]}")

if __name__ == "__main__":
    main()
