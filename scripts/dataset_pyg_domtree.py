# scripts/dataset_pyg_domtree.py
from __future__ import annotations
import json, sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
PB2_ROOT = ROOT / "src" / "programl_pb2"
if str(PB2_ROOT) not in sys.path:
    sys.path.insert(0, str(PB2_ROOT))

from programl.proto.program_graph_pb2 import ProgramGraph
from programl.proto.util_pb2 import ProgramGraphFeaturesList


# -----------------------------
# TF Feature helpers
# -----------------------------
def _feature_to_int(f) -> int:
    if f.HasField("int64_list") and len(f.int64_list.value) > 0:
        return int(f.int64_list.value[0])
    if f.HasField("float_list") and len(f.float_list.value) > 0:
        return int(f.float_list.value[0])
    if f.HasField("bytes_list") and len(f.bytes_list.value) > 0:
        try:
            return int(f.bytes_list.value[0])
        except Exception:
            return 0
    return 0


def _read_node_feature_list(step, key: str, N: int) -> List[int]:
    nfl = step.node_features.feature_list.get(key, None)
    if nfl is None:
        raise KeyError(f"Missing node feature_list key '{key}'")
    vals = [_feature_to_int(ft) for ft in nfl.feature]
    if len(vals) != N:
        raise ValueError(f"FeatureList '{key}' length mismatch: got {len(vals)} expected {N}")
    return vals


def _argmax_onehot(x: List[int]) -> int:
    best_i, best_v = 0, (x[0] if len(x) else 0)
    for i, v in enumerate(x):
        if v > best_v:
            best_v = v
            best_i = i
    return best_i


# -----------------------------
# Edge mapping
# -----------------------------
def _edge_flow_to_int(e) -> int:
    # ProgramGraph.Edge has "flow" enum (ProGraML)
    # keep as int index
    return int(getattr(e, "flow", 0))


def _node_type_to_int(n) -> int:
    # ProgramGraph.Node has "type" enum
    return int(getattr(n, "type", 0))


# -----------------------------
# Dataset
# -----------------------------
class DomTreePyGDataset(torch.utils.data.Dataset):
    """
    Task: DomTree (control-flow / backward) labels, but stored in the same
    ProgramGraphFeaturesList format with node_features:
      - data_flow_root_node (one-hot over nodes)
      - data_flow_value (0/1 per node)
    Root-conditional generalization:
      use root_split/{split}_{main|holdout}.json
      each entry chooses (key, step_id)
    """

    def __init__(
        self,
        subset_dir: Path,
        split: str = "train",
        root_mode: str = "main",  # "main" or "holdout"
        root_split_dir: Optional[Path] = None,
        max_samples: Optional[int] = None,
        seed: int = 0,
        control_flow_only: bool = False,
    ):
        self.subset_dir = Path(subset_dir)
        self.split = split
        assert root_mode in ["main", "holdout"]
        self.root_mode = root_mode

        self.graphs_dir = self.subset_dir / split / "graphs"
        self.labels_dir = self.subset_dir / split / "labels"

        if root_split_dir is None:
            root_split_dir = self.subset_dir / "root_split"
        self.root_split_dir = Path(root_split_dir)

        idx_path = self.root_split_dir / f"{split}_{root_mode}.json"
        if not idx_path.exists():
            raise FileNotFoundError(f"Missing root-split index: {idx_path}")

        items = json.loads(idx_path.read_text(encoding="utf-8"))
        # items: [{"key": "...", "step_id": 3}, ...]
        if max_samples is not None:
            # deterministic cut
            items = items[: int(max_samples)]

        self.items = items
        self.control_flow_only = control_flow_only

    def __len__(self) -> int:
        return len(self.items)

    def _load_graph(self, key: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gp = self.graphs_dir / f"{key}.ProgramGraph.pb"
        if not gp.exists():
            raise FileNotFoundError(f"Missing graph pb: {gp}")

        g = ProgramGraph()
        g.ParseFromString(gp.read_bytes())

        node_type = torch.tensor([_node_type_to_int(n) for n in g.node], dtype=torch.long)

        # edges
        src = []
        dst = []
        flow = []
        for e in g.edge:
            ef = _edge_flow_to_int(e)
            if self.control_flow_only:
                # Keep only control-flow-ish edges.
                # In ProGraML, flow enum values depend on schema; we keep a conservative rule:
                # include edges whose flow name contains "CONTROL" if enum provides name,
                # else include all (fallback).
                # If your pb2 has enum names, this works. Otherwise control_flow_only=False.
                try:
                    enum_desc = e.DESCRIPTOR.fields_by_name["flow"].enum_type
                    name = enum_desc.values[ef].name.upper()
                    if "CONTROL" not in name:
                        continue
                except Exception:
                    pass

            src.append(int(e.source))
            dst.append(int(e.target))
            flow.append(int(ef))

        if len(src) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_flow = torch.zeros((0,), dtype=torch.long)
        else:
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_flow = torch.tensor(flow, dtype=torch.long)

        return node_type, edge_index, edge_flow

    def _load_label_step(self, key: str, step_id: int, N: int) -> Tuple[int, torch.Tensor]:
        lp = self.labels_dir / f"{key}.ProgramGraphFeaturesList.pb"
        if not lp.exists():
            raise FileNotFoundError(f"Missing label pb: {lp}")

        ylist = ProgramGraphFeaturesList()
        ylist.ParseFromString(lp.read_bytes())

        steps = list(ylist.graph)
        if step_id < 0 or step_id >= len(steps):
            raise IndexError(f"step_id out of range: {step_id} (steps={len(steps)}) key={key}")

        st = steps[step_id]
        root_vec = _read_node_feature_list(st, "data_flow_root_node", N)
        y_vec = _read_node_feature_list(st, "data_flow_value", N)

        root_idx = _argmax_onehot(root_vec)
        y = torch.tensor(y_vec, dtype=torch.float32)
        return root_idx, y

    def __getitem__(self, i: int) -> Dict[str, Any]:
        it = self.items[i]
        key = it["key"]
        step_id = int(it["step_id"])

        node_type, edge_index, edge_flow = self._load_graph(key)
        N = int(node_type.numel())

        root_idx, y = self._load_label_step(key, step_id, N)

        return {
            "node_type": node_type,
            "edge_index": edge_index,
            "edge_flow": edge_flow,
            "root_idx": root_idx,
            "y": y,
            "key": key,
            "step_id": step_id,
            "split": self.split,
        }
