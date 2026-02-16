# scripts/build_root_split_domtree.py
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
from typing import List, Dict, Any, Tuple

ROOT = Path(__file__).resolve().parents[1]
PB2_ROOT = ROOT / "src" / "programl_pb2"
if str(PB2_ROOT) not in sys.path:
    sys.path.insert(0, str(PB2_ROOT))

from programl.proto.util_pb2 import ProgramGraphFeaturesList

# ---- feature helpers (tensorflow FeatureLists) ----
def _featurelist_len(fl) -> int:
    return len(fl.feature)

def _feature_to_int(f) -> int:
    # f is tensorflow.Feature
    if f.HasField("int64_list") and len(f.int64_list.value) > 0:
        return int(f.int64_list.value[0])
    if f.HasField("float_list") and len(f.float_list.value) > 0:
        return int(f.float_list.value[0])
    if f.HasField("bytes_list") and len(f.bytes_list.value) > 0:
        # unlikely for labels
        try:
            return int(f.bytes_list.value[0])
        except Exception:
            return 0
    return 0

def _read_node_feature_list(step, key: str, N_hint: int | None = None) -> List[int]:
    """
    Returns per-node list of ints length N (usually).
    DeepDataFlow stores per-node features as FeatureList of length N.
    Each element is a tensorflow.Feature with one scalar value.
    """
    nfl = step.node_features.feature_list.get(key, None)
    if nfl is None:
        raise KeyError(f"Missing node feature_list key '{key}'")
    vals = [_feature_to_int(ft) for ft in nfl.feature]
    if N_hint is not None and len(vals) != N_hint:
        # keep going, but warn via exception context later if needed
        pass
    return vals

def _argmax_onehot(x: List[int]) -> int:
    best_i, best_v = 0, (x[0] if len(x) else 0)
    for i, v in enumerate(x):
        if v > best_v:
            best_v = v
            best_i = i
    return best_i

def collect_samples(label_pb_path: Path) -> List[Dict[str, Any]]:
    p = ProgramGraphFeaturesList()
    p.ParseFromString(label_pb_path.read_bytes())
    steps = list(p.graph)
    if len(steps) == 0:
        return []

    # determine N from first step's data_flow_root_node length
    # (works for your debug output)
    root0 = _read_node_feature_list(steps[0], "data_flow_root_node", N_hint=None)
    N = len(root0)

    root_to_steps: Dict[int, List[int]] = {}
    for sid, st in enumerate(steps):
        root_vec = _read_node_feature_list(st, "data_flow_root_node", N_hint=N)
        ridx = _argmax_onehot(root_vec)
        root_to_steps.setdefault(ridx, []).append(sid)

    out = []
    for ridx, sids in root_to_steps.items():
        for sid in sids:
            out.append({"root_idx": int(ridx), "step_id": int(sid)})
    return out

def build_split(subset_domtree: Path, split: str, holdout_ratio: float, seed: int, out_dir: Path):
    labels_dir = subset_domtree / split / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels dir: {labels_dir}")

    rng = random.Random(seed)
    main_list: List[Dict[str, Any]] = []
    holdout_list: List[Dict[str, Any]] = []

    label_files = sorted(labels_dir.glob("*.ProgramGraphFeaturesList.pb"))
    if len(label_files) == 0:
        raise RuntimeError(f"No label files under {labels_dir}")

    for lp in label_files:
        key = lp.name.replace(".ProgramGraphFeaturesList.pb", "")
        samples = collect_samples(lp)
        if len(samples) == 0:
            continue

        root_to_steps: Dict[int, List[int]] = {}
        for s in samples:
            root_to_steps.setdefault(int(s["root_idx"]), []).append(int(s["step_id"]))

        roots = list(root_to_steps.keys())
        if len(roots) == 1:
            for sid in root_to_steps[roots[0]]:
                main_list.append({"key": key, "step_id": int(sid)})
            continue

        rng.shuffle(roots)
        k_hold = max(1, int(round(len(roots) * holdout_ratio)))
        hold_roots = set(roots[:k_hold])
        main_roots = set(roots[k_hold:])

        for r in main_roots:
            for sid in root_to_steps[r]:
                main_list.append({"key": key, "step_id": int(sid)})
        for r in hold_roots:
            for sid in root_to_steps[r]:
                holdout_list.append({"key": key, "step_id": int(sid)})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{split}_main.json").write_text(json.dumps(main_list, indent=2), encoding="utf-8")
    (out_dir / f"{split}_holdout.json").write_text(json.dumps(holdout_list, indent=2), encoding="utf-8")

    print(f"[{split}] main={len(main_list)} holdout={len(holdout_list)} -> {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset_domtree", type=str, default="data/subset_domtree")
    ap.add_argument("--out_dir", type=str, default="data/subset_domtree/root_split")
    ap.add_argument("--holdout_ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    subset_domtree = Path(args.subset_domtree)
    out_dir = Path(args.out_dir)

    for sp in ["train", "val", "test"]:
        build_split(
            subset_domtree,
            sp,
            args.holdout_ratio,
            args.seed + (0 if sp == "train" else 1 if sp == "val" else 2),
            out_dir,
        )

if __name__ == "__main__":
    main()
