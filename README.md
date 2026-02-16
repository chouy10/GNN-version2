# GNN-version2

https://zenodo.org/records/4247595

data:Need to download labels_domtree_20.06.01.tar.bz2, graphs_20.06.01.tar.bz2 labels_reachability_20.06.01.tar.bz2 vocab_20.06.01.tar.bz2 from above link 

Project Overview: Root-Conditioned Node Prediction on Program Graphs (Reachability + DomTree)
Purpose (What problem are we solving?)

This project builds a Graph Neural Network (GNN) system to predict node-level properties conditioned on a given root node in a program graph.

Reachability task: Given a program graph and a starting point (“root”), predict which nodes become reachable / active.

DomTree task (new): Using the same feature schema style, predict dominator-tree related node labels (implemented as a second task option in the same training/prediction pipeline).

This is useful as an end-to-end learning exercise for:

program-graph representations (ProGraML / ProgramGraph protobuf)

graph machine learning (edge-aware message passing)

converting compiler graphs into training-ready datasets

evaluating generalization under different root distributions (new)

Data & Labels (What are the inputs and outputs?)

Input graph: ProgramGraph.pb

Nodes include fields like type, text, etc.

Edges include a flow category (e.g., data-flow vs control-flow types)

Label file: ProgramGraphFeaturesList.pb
Each graph contains multiple “steps” (e.g., len(y.graph)=7). Each step provides:

data_flow_root_node: one-hot root indicator

data_flow_value: per-node binary label (0/1) indicating active/reachable

graph-level statistics (sanity-checkable), e.g.

data_flow_active_node_count

data_flow_step_count

Sanity checks confirmed:

label node count matches graph node count

sum(data_flow_value==1) matches data_flow_active_node_count

New: Multi-task dataset support

The training script now supports two datasets/tasks via --task:

reachability → ReachabilityPyGDataset

domtree → DomTreePyGDataset

DomTree uses the same schema keys style in the node_features feature_list, allowing the same model + pipeline to be reused with minimal changes.

Model Design (How does the model work?)

We use a lightweight edge-aware GraphSAGE-style GNN.

Node features

node_type → learned embedding

root_flag (1 if node is root else 0) → projected and added into node embeddings (root-conditioned prediction)

Edge features

edge_flow → learned embedding

edge embeddings are injected into message passing and aggregated by mean over incoming edges

Prediction

an MLP outputs one logit per node

sigmoid(logit) gives probability of being positive (reachable/active or DomTree-positive depending on task)

Loss

BCEWithLogitsLoss(pos_weight=...) to handle class imbalance (positive nodes usually fewer)

Pipeline & Functionality (What can the code do?)

The script supports Train and Predict (visible demo) for both tasks.

A) Train mode

trains the GNN for 10 epochs

evaluates on validation set

saves best checkpoint based on IoU (Jaccard)

Reachability

python .\scripts\train_reachability_gnn.py --mode train --task reachability


DomTree (new)

python .\scripts\train_reachability_gnn.py --mode train --task domtree --subset data/subset_domtree --domtree_root_mode main


New: Root-conditional generalization evaluation (DomTree)
For DomTree, training/eval is explicitly split by root distributions:

Train on: train_main

Validate on: val_main and val_holdout

Test on: test_main and test_holdout

Train logs include:

val_main and val_holdout
End-of-training reports include:

test_main and test_holdout

This directly measures how well the model generalizes when the root node comes from unseen/holdout root sets.

B) Predict mode (visible demo)

loads a trained checkpoint

runs inference on one sample graph

prints:

graph size (nodes, edges)

root index + metadata (key/step/split)

metrics (IoU, F1, precision, recall)

predicted node list vs ground truth

top probability nodes

optionally exports:

CSV (node-wise predictions)

DOT (GraphViz visualization)

PNG (networkx + matplotlib visualization)

Reachability

python .\scripts\train_reachability_gnn.py --mode predict --task reachability --split test --index 0 --out_png outputs\pred0.png


DomTree (new) — including holdout root evaluation

python .\scripts\train_reachability_gnn.py --mode predict --task domtree --subset data/subset_domtree --domtree_root_mode holdout --split test --index 0 --out_png outputs\dom_pred0.png

Results (What did we observe?)

For a sample test graph:

~134 nodes and ~225 edges

Only a subset of nodes are positive (e.g., 18/134 ≈ 13% in one step)

Predict mode reports:

IoU / F1 / precision / recall

plus interpretable outputs: predicted node IDs, true node IDs, top-probability nodes

Visualization outputs (DOT/PNG) highlight:

True positives / False positives / False negatives / True negatives
This makes the project easy to demo, debug, and iterate.

New for DomTree:
Because we evaluate on main vs holdout roots, results can be interpreted as:

in-distribution performance (main roots)

out-of-distribution / generalization performance (holdout roots)

Key Takeaways

Converted ProGraML protobuf graphs into training-ready PyTorch datasets.

Built a root-conditioned GNN that predicts node-level labels on program graphs.

Added DomTree as a second supported task using the same pipeline.

Added root-conditional generalization evaluation (main vs holdout roots) to explicitly test robustness to unseen root distributions.

Predict mode provides strong demo value via printed metrics + CSV/DOT/PNG visualization.
