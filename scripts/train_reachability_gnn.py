# scripts/train_reachability_gnn.py
# - Keeps original reachability train/predict + CSV/DOT/PNG export
# - Adds DomTree task support (same schema keys in node_features feature_list)
# - Adds Root-conditional generalization evaluation: main vs holdout roots
#
# Usage:
#   Reachability (original):
#     python .\scripts\train_reachability_gnn.py --mode train --task reachability
#     python .\scripts\train_reachability_gnn.py --mode predict --task reachability --split test --index 0 --out_png outputs\pred0.png
#
#   DomTree:
#     python .\scripts\train_reachability_gnn.py --mode train --task domtree --subset data/subset_domtree --domtree_root_mode main
#     python .\scripts\train_reachability_gnn.py --mode predict --task domtree --subset data/subset_domtree --domtree_root_mode holdout --split test --index 0 --out_png outputs\dom_pred0.png
#
#   (Train mode reports val_main and val_holdout + test_main/test_holdout at end)

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parents[1]

_root_str = str(ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

# keep pb2 importable if any script needs it
_prog_pb2 = str(ROOT / "src" / "programl_pb2")
if _prog_pb2 not in sys.path:
    sys.path.insert(0, _prog_pb2)

from scripts.dataset_pyg_reachability import ReachabilityPyGDataset
from scripts.dataset_pyg_domtree import DomTreePyGDataset


def pick_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        _ = torch.empty(1, device="cuda") + 1
        return "cuda"
    except Exception as e:
        print(f"[WARN] CUDA available but not runnable ({type(e).__name__}: {e}). Falling back to CPU.")
        return "cpu"


# -----------------------------
# Collate -> make a single big graph batch
# -----------------------------
def collate_graph_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    node_offsets = []
    total_nodes = 0
    for b in batch:
        node_offsets.append(total_nodes)
        total_nodes += b["node_type"].numel()

    node_type = torch.cat([b["node_type"] for b in batch], dim=0)
    y = torch.cat([b["y"] for b in batch], dim=0).float()

    root_flag = torch.zeros(total_nodes, dtype=torch.float32)
    for i, b in enumerate(batch):
        off = node_offsets[i]
        root_flag[off + int(b["root_idx"])] = 1.0

    edge_index_list = []
    edge_flow_list = []
    for i, b in enumerate(batch):
        off = node_offsets[i]
        ei = b["edge_index"] + off
        edge_index_list.append(ei)
        edge_flow_list.append(b["edge_flow"])
    edge_index = torch.cat(edge_index_list, dim=1) if edge_index_list else torch.zeros((2, 0), dtype=torch.long)
    edge_flow = torch.cat(edge_flow_list, dim=0) if edge_flow_list else torch.zeros((0,), dtype=torch.long)

    return {
        "edge_index": edge_index,
        "node_type": node_type,
        "root_flag": root_flag,
        "edge_flow": edge_flow,
        "y": y,
        "meta": [(b.get("key", None), b.get("step_id", None), b.get("split", None)) for b in batch],
        "num_graphs": len(batch),
        "num_nodes": total_nodes,
        "num_edges": int(edge_flow.numel()),
    }


# -----------------------------
# Simple Edge-aware GraphSAGE
# -----------------------------
class EdgeSAGEConv(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.w_self = nn.Linear(hidden, hidden, bias=True)
        self.w_nei = nn.Linear(hidden, hidden, bias=False)
        self.w_edge = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            return F.relu(self.w_self(x))

        src, dst = edge_index[0], edge_index[1]
        msg = self.w_nei(x[src]) + self.w_edge(edge_attr)

        N = x.size(0)
        out = torch.zeros((N, x.size(1)), device=x.device, dtype=x.dtype)
        out.index_add_(0, dst, msg)

        deg = torch.zeros((N,), device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones((dst.numel(),), device=x.device, dtype=x.dtype))
        deg = deg.clamp_min(1.0).unsqueeze(1)
        out = out / deg

        return F.relu(self.w_self(x) + out)


class ReachabilityGNN(nn.Module):
    def __init__(self, num_node_types: int, num_edge_flows: int, hidden: int = 64, layers: int = 3):
        super().__init__()
        self.node_emb = nn.Embedding(num_node_types, hidden)
        self.edge_emb = nn.Embedding(num_edge_flows, hidden)
        self.root_proj = nn.Linear(1, hidden, bias=False)

        self.convs = nn.ModuleList([EdgeSAGEConv(hidden) for _ in range(layers)])
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_type: torch.Tensor, root_flag: torch.Tensor, edge_index: torch.Tensor, edge_flow: torch.Tensor):
        x = self.node_emb(node_type)
        x = x + self.root_proj(root_flag.unsqueeze(1))
        e = self.edge_emb(edge_flow) if edge_flow.numel() > 0 else torch.zeros((0, x.size(1)), device=x.device, dtype=x.dtype)

        for conv in self.convs:
            x = conv(x, edge_index, e)

        logits = self.mlp(x).squeeze(1)
        return logits


# -----------------------------
# Metrics
# -----------------------------
@torch.no_grad()
def compute_metrics(logits: torch.Tensor, y: torch.Tensor, thresh: float = 0.0) -> Dict[str, float]:
    pred = (logits >= thresh).to(torch.long)
    y_i = y.to(torch.long)

    tp = int(((pred == 1) & (y_i == 1)).sum().item())
    tn = int(((pred == 0) & (y_i == 0)).sum().item())
    fp = int(((pred == 1) & (y_i == 0)).sum().item())
    fn = int(((pred == 0) & (y_i == 1)).sum().item())

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)
    iou = tp / max(1, tp + fp + fn)

    return {"acc": acc, "f1": f1, "iou": iou, "tp": tp, "fp": fp, "fn": fn}


def estimate_type_counts(subset_dir: Path, task: str, domtree_root_mode: str, root_split_dir: Optional[Path]) -> Dict[str, int]:
    if task == "reachability":
        ds = ReachabilityPyGDataset(subset_dir, split="train", max_graphs=200, seed=0)
    else:
        ds = DomTreePyGDataset(subset_dir, split="train", root_mode=domtree_root_mode, root_split_dir=root_split_dir)

    if len(ds) == 0:
        raise RuntimeError(f"Empty train split under {subset_dir}. Cannot estimate type counts.")

    max_node_type = 0
    max_edge_flow = 0
    for i in range(min(len(ds), 500)):
        ex = ds[i]
        max_node_type = max(max_node_type, int(ex["node_type"].max().item()))
        if ex["edge_flow"].numel() > 0:
            max_edge_flow = max(max_edge_flow, int(ex["edge_flow"].max().item()))
    return {"num_node_types": max_node_type + 1, "num_edge_flows": max_edge_flow + 1}


# -----------------------------
# "Visible application" helpers
# -----------------------------
def _get_meta(ex: Dict[str, Any]) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    return (ex.get("key", None), ex.get("step_id", None), ex.get("split", None))


@torch.no_grad()
def predict_one(
    model: ReachabilityGNN,
    ex: Dict[str, Any],
    device: str,
    prob_thresh: float = 0.5,
) -> Dict[str, Any]:
    model.eval()

    node_type = ex["node_type"].to(device)
    edge_index = ex["edge_index"].to(device)
    edge_flow = ex["edge_flow"].to(device)
    y = ex["y"].float().to(device)
    root_idx = int(ex["root_idx"])

    root_flag = torch.zeros(node_type.numel(), device=device, dtype=torch.float32)
    root_flag[root_idx] = 1.0

    logits = model(node_type, root_flag, edge_index, edge_flow)
    prob = torch.sigmoid(logits)

    pred = (prob >= prob_thresh).to(torch.long)
    y_i = y.to(torch.long)

    tp = int(((pred == 1) & (y_i == 1)).sum().item())
    fp = int(((pred == 1) & (y_i == 0)).sum().item())
    fn = int(((pred == 0) & (y_i == 1)).sum().item())
    iou = tp / max(1, tp + fp + fn)
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)

    m_ref = compute_metrics(logits, y, thresh=0.0)

    pred_idx = pred.nonzero(as_tuple=False).squeeze(1).detach().cpu().tolist()
    true_idx = y_i.nonzero(as_tuple=False).squeeze(1).detach().cpu().tolist()

    topk = min(10, prob.numel())
    top_prob, top_i = torch.topk(prob, k=topk)
    top_list = [(int(i.item()), float(p.item())) for p, i in zip(top_prob.detach().cpu(), top_i.detach().cpu())]

    return {
        "root_idx": root_idx,
        "num_nodes": int(node_type.numel()),
        "num_edges": int(edge_flow.numel()),
        "prob_thresh": prob_thresh,
        "metrics_prob_thresh": {"iou": iou, "f1": f1, "prec": prec, "rec": rec, "tp": tp, "fp": fp, "fn": fn},
        "metrics_logit0_thresh": m_ref,
        "pred_reachable_nodes": pred_idx,
        "true_reachable_nodes": true_idx,
        "top_prob_nodes": top_list,
        "prob": prob.detach().cpu(),
        "pred": pred.detach().cpu(),
        "y": y_i.detach().cpu(),
        "edge_index": ex["edge_index"].detach().cpu(),
        "edge_flow": ex["edge_flow"].detach().cpu(),
        "node_type": ex["node_type"].detach().cpu(),
        "meta": _get_meta(ex),
    }


def export_csv(out_path: Path, result: Dict[str, Any]) -> None:
    prob = result["prob"].numpy()
    pred = result["pred"].numpy()
    y = result["y"].numpy()
    node_type = result["node_type"].numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("node_id,node_type,y_true,y_pred,prob\n")
        for i in range(len(prob)):
            f.write(f"{i},{int(node_type[i])},{int(y[i])},{int(pred[i])},{float(prob[i]):.6f}\n")


def export_dot(out_path: Path, result: Dict[str, Any]) -> None:
    edge_index = result["edge_index"].numpy()
    edge_flow = result["edge_flow"].numpy()
    y = result["y"].numpy()
    pred = result["pred"].numpy()
    node_type = result["node_type"].numpy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("digraph G {\n")
        f.write('  rankdir=LR;\n')
        f.write('  node [shape=box, fontname="Consolas"];\n')

        for i in range(len(node_type)):
            if pred[i] == 1 and y[i] == 1:
                color = "palegreen"
            elif pred[i] == 1 and y[i] == 0:
                color = "gold"
            elif pred[i] == 0 and y[i] == 1:
                color = "tomato"
            else:
                color = "lightgray"

            label = f"{i} | t={int(node_type[i])} | y={int(y[i])}/p={int(pred[i])}"
            f.write(f'  n{i} [label="{label}", style=filled, fillcolor="{color}"];\n')

        for e in range(edge_index.shape[1]):
            s = int(edge_index[0, e])
            d = int(edge_index[1, e])
            ef = int(edge_flow[e]) if edge_flow.size > 0 else 0
            f.write(f'  n{s} -> n{d} [label="f={ef}", fontsize=10];\n')

        f.write("}\n")


def export_png(out_path: Path, result: Dict[str, Any], max_nodes: int = 300) -> None:
    edge_index = result["edge_index"].numpy()
    y = result["y"].numpy()
    pred = result["pred"].numpy()

    n = len(y)
    keep_n = min(n, max_nodes)

    G = nx.DiGraph()
    G.add_nodes_from(range(keep_n))
    for e in range(edge_index.shape[1]):
        s = int(edge_index[0, e])
        d = int(edge_index[1, e])
        if s < keep_n and d < keep_n:
            G.add_edge(s, d)

    colors = []
    for i in range(keep_n):
        if pred[i] == 1 and y[i] == 1:
            colors.append("tab:green")
        elif pred[i] == 1 and y[i] == 0:
            colors.append("tab:orange")
        elif pred[i] == 0 and y[i] == 1:
            colors.append("tab:red")
        else:
            colors.append("tab:gray")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pos = nx.spring_layout(G, seed=0)

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=220, alpha=0.9)
    nx.draw_networkx_edges(G, pos, arrows=True, width=0.8, alpha=0.35)

    if keep_n <= 80:
        nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(
        f"Node-set Prediction (showing {keep_n}/{n} nodes)\n"
        f"TP=green FP=orange FN=red TN=gray   prob_thresh={result['prob_thresh']}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def load_checkpoint(model: ReachabilityGNN, ckpt_path: Path, device: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    return ckpt


def _make_dataset(task: str, subset: Path, split: str, domtree_root_mode: str, root_split_dir: Optional[Path], seed: int):
    if task == "reachability":
        return ReachabilityPyGDataset(subset, split=split, seed=seed)
    else:
        return DomTreePyGDataset(subset, split=split, root_mode=domtree_root_mode, root_split_dir=root_split_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--task", choices=["reachability", "domtree"], default="reachability")

    parser.add_argument("--subset", type=str, default="")
    parser.add_argument("--ckpt", type=str, default="")

    # domtree options
    parser.add_argument("--domtree_root_mode", choices=["main", "holdout"], default="main")
    parser.add_argument("--root_split_dir", type=str, default="", help="optional: data/subset_domtree/root_split")

    # predict options
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--prob_thresh", type=float, default=0.5)
    parser.add_argument("--out_csv", type=str, default="")
    parser.add_argument("--out_dot", type=str, default="")
    parser.add_argument("--out_png", type=str, default="")
    parser.add_argument("--max_nodes", type=int, default=300)

    args = parser.parse_args()

    task = args.task
    device = pick_device()

    if args.subset:
        subset = Path(args.subset)
    else:
        subset = Path(ROOT / "data" / ("subset_reachability" if task == "reachability" else "subset_domtree"))

    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_name = f"{task}_gnn.pt" if task == "reachability" else f"{task}_gnn_{args.domtree_root_mode}.pt"
        ckpt_path = Path(ROOT / "checkpoints" / ckpt_name)

    root_split_dir = Path(args.root_split_dir) if args.root_split_dir else None

    counts = estimate_type_counts(subset, task=task, domtree_root_mode=args.domtree_root_mode, root_split_dir=root_split_dir)
    num_node_types = counts["num_node_types"]
    num_edge_flows = counts["num_edge_flows"]

    model = ReachabilityGNN(num_node_types=num_node_types, num_edge_flows=num_edge_flows, hidden=64, layers=3).to(device)

    # -----------------------------
    # PREDICT
    # -----------------------------
    if args.mode == "predict":
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        load_checkpoint(model, ckpt_path, device)

        ds = _make_dataset(task, subset, split=args.split, domtree_root_mode=args.domtree_root_mode, root_split_dir=root_split_dir, seed=0)
        if len(ds) == 0:
            raise RuntimeError(f"Empty split: {args.split} under {subset}")

        idx = args.index % len(ds)
        ex = ds[idx]
        result = predict_one(model, ex, device=device, prob_thresh=args.prob_thresh)

        key, step_id, split = result["meta"]
        print("\n================= PREDICT (VISIBLE OUTPUT) =================")
        print(f"task={task}  device={device}")
        print(f"ckpt={ckpt_path}")
        if task == "domtree":
            print(f"domtree_root_mode={args.domtree_root_mode}  root_split_dir={root_split_dir}")
        print(f"sample: split={args.split} index={idx} meta(key={key}, step_id={step_id}, split={split})")
        print(f"graph: nodes={result['num_nodes']} edges={result['num_edges']} root_idx={result['root_idx']}")
        print(f"threshold: prob_thresh={result['prob_thresh']}")

        print("\n--- Metrics @ prob_thresh ---")
        m2 = result["metrics_prob_thresh"]
        print(
            f"iou={m2['iou']:.3f} f1={m2['f1']:.3f} prec={m2['prec']:.3f} rec={m2['rec']:.3f}  "
            f"tp={m2['tp']} fp={m2['fp']} fn={m2['fn']}"
        )

        print("\n--- Predicted nodes (first 50) ---")
        pr = result["pred_reachable_nodes"]
        print(pr[:50], ("...(truncated)" if len(pr) > 50 else ""))
        print(f"pred_count={len(pr)}")

        print("\n--- True nodes (first 50) ---")
        tr = result["true_reachable_nodes"]
        print(tr[:50], ("...(truncated)" if len(tr) > 50 else ""))
        print(f"true_count={len(tr)}")

        print("\n--- Top prob nodes (id, prob) ---")
        print(result["top_prob_nodes"])

        if args.out_csv:
            out_csv = Path(args.out_csv)
            export_csv(out_csv, result)
            print(f"\n[Saved] CSV: {out_csv}")

        if args.out_dot:
            out_dot = Path(args.out_dot)
            export_dot(out_dot, result)
            print(f"[Saved] DOT: {out_dot}")
            print("        (View DOT with GraphViz: dot -Tpng file.dot -o file.png)")

        if args.out_png:
            out_png = Path(args.out_png)
            export_png(out_png, result, max_nodes=args.max_nodes)
            print(f"[Saved] PNG: {out_png}")

        print("============================================================\n")
        return

    # -----------------------------
    # TRAIN
    # -----------------------------
    if task == "reachability":
        train_ds = ReachabilityPyGDataset(subset, split="train", seed=0)
        val_ds = ReachabilityPyGDataset(subset, split="val", seed=1)
        test_ds = ReachabilityPyGDataset(subset, split="test", seed=2)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_graph_batch, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_graph_batch, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, collate_fn=collate_graph_batch, num_workers=0)

        extra_eval_loaders = {}  # none for reachability
    else:
        # domtree root-conditional generalization:
        # train on train_main only, eval on val_main + val_holdout, test_main + test_holdout
        train_ds = DomTreePyGDataset(subset, split="train", root_mode="main", root_split_dir=root_split_dir)
        val_main = DomTreePyGDataset(subset, split="val", root_mode="main", root_split_dir=root_split_dir)
        val_hold = DomTreePyGDataset(subset, split="val", root_mode="holdout", root_split_dir=root_split_dir)
        test_main = DomTreePyGDataset(subset, split="test", root_mode="main", root_split_dir=root_split_dir)
        test_hold = DomTreePyGDataset(subset, split="test", root_mode="holdout", root_split_dir=root_split_dir)

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_graph_batch, num_workers=0)
        val_loader = DataLoader(val_main, batch_size=8, shuffle=False, collate_fn=collate_graph_batch, num_workers=0)
        test_loader = DataLoader(test_main, batch_size=8, shuffle=False, collate_fn=collate_graph_batch, num_workers=0)

        extra_eval_loaders = {
            "val_holdout": DataLoader(val_hold, batch_size=8, shuffle=False, collate_fn=collate_graph_batch, num_workers=0),
            "test_holdout": DataLoader(test_hold, batch_size=8, shuffle=False, collate_fn=collate_graph_batch, num_workers=0),
        }

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    # compute pos_weight from a few train batches
    pos = 0.0
    neg = 0.0
    for k, batch in enumerate(train_loader):
        yy = batch["y"]
        pos += float((yy == 1).sum().item())
        neg += float((yy == 0).sum().item())
        if k >= 20:
            break
    pos_weight = torch.tensor([neg / max(1.0, pos)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"task={task} device={device}  num_node_types={num_node_types}  num_edge_flows={num_edge_flows}  pos_weight={pos_weight.item():.2f}")
    if task == "domtree":
        print(f"root_split_dir={root_split_dir}  train_on=train_main  eval=val_main+val_holdout  test=test_main+test_holdout")
    print(f"train_samples={len(train_ds)}  val_samples={len(val_loader.dataset)}  test_samples={len(test_loader.dataset)}")

    def run_epoch(loader, train: bool):
        model.train() if train else model.eval()

        total_loss = 0.0
        total = {"acc": 0.0, "f1": 0.0, "iou": 0.0}
        steps = 0

        for batch in loader:
            node_type = batch["node_type"].to(device)
            root_flag = batch["root_flag"].to(device)
            edge_index = batch["edge_index"].to(device)
            edge_flow = batch["edge_flow"].to(device)
            yy = batch["y"].to(device)

            logits = model(node_type, root_flag, edge_index, edge_flow)
            loss = loss_fn(logits, yy)

            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            m = compute_metrics(logits.detach(), yy.detach(), thresh=0.0)
            total_loss += float(loss.item())
            total["acc"] += m["acc"]
            total["f1"] += m["f1"]
            total["iou"] += m["iou"]
            steps += 1

        for kk in total:
            total[kk] /= max(1, steps)

        return total_loss / max(1, steps), total

    best_val_iou = -1.0
    best_path = ckpt_path
    best_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, 11):
        tr_loss, tr_m = run_epoch(train_loader, train=True)
        va_loss, va_m = run_epoch(val_loader, train=False)

        msg = (
            f"[{epoch:02d}] train loss={tr_loss:.4f} acc={tr_m['acc']:.3f} f1={tr_m['f1']:.3f} iou={tr_m['iou']:.3f} | "
            f"val_main loss={va_loss:.4f} acc={va_m['acc']:.3f} f1={va_m['f1']:.3f} iou={va_m['iou']:.3f}"
        )

        # extra evals (val_holdout)
        if task == "domtree":
            hv_loss, hv_m = run_epoch(extra_eval_loaders["val_holdout"], train=False)
            msg += (
                f" | val_holdout loss={hv_loss:.4f} acc={hv_m['acc']:.3f} f1={hv_m['f1']:.3f} iou={hv_m['iou']:.3f}"
            )

        print(msg)

        if va_m["iou"] > best_val_iou:
            best_val_iou = va_m["iou"]
            torch.save(
                {"model": model.state_dict(), "num_node_types": num_node_types, "num_edge_flows": num_edge_flows, "task": task},
                best_path,
            )

    # test with best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    te_loss, te_m = run_epoch(test_loader, train=False)
    print(f"[TEST main] loss={te_loss:.4f} acc={te_m['acc']:.3f} f1={te_m['f1']:.3f} iou={te_m['iou']:.3f}")

    if task == "domtree":
        th_loss, th_m = run_epoch(extra_eval_loaders["test_holdout"], train=False)
        print(f"[TEST holdout] loss={th_loss:.4f} acc={th_m['acc']:.3f} f1={th_m['f1']:.3f} iou={th_m['iou']:.3f}")

    print(f"Saved best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
