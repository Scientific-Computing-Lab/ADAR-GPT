#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics

# Keep TF quiet (CPU fine)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from tensorflow.keras.models import model_from_json

# -----------------------------
# JSONL parsing (keeps L/R)
# -----------------------------
L_AR_STRUCT_RE = re.compile(
    r'^L:(?P<L>.*?),\s*A:(?P<center>[ACGTUacgtu]),\s*R:(?P<R>.*)Alu Vienna Structure:(?P<struct>[\.\(\)]+)$',
    re.DOTALL
)

def read_jsonl(p: Path):
    with p.open("r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def sanitize_seq(seq: str) -> str:
    s = seq.upper().replace("U","T")
    return re.sub(r'[^ACGTN]', '', s)

def parse_adargpt_jsonl(p: Path) -> pd.DataFrame:
    rows, skipped = [], 0
    for i, rec in enumerate(read_jsonl(p)):
        try:
            user = rec["messages"][1]["content"]
            lab  = rec["messages"][2]["content"].strip().lower()
            label = 1 if lab == "yes" else 0 if lab == "no" else None
            m = L_AR_STRUCT_RE.match(user)
            if not m or label is None:
                skipped += 1; continue
            L_raw, center, R_raw = m.group("L"), m.group("center"), m.group("R")
            if center.upper() != "A":
                skipped += 1; continue
            Ls = sanitize_seq(L_raw); Rs = sanitize_seq(R_raw)
            rows.append({"id": i, "L": Ls, "R": Rs, "label": label})
        except Exception:
            skipped += 1
    if not rows:
        raise RuntimeError(f"No usable rows parsed from {p}")
    df = pd.DataFrame(rows).reset_index(drop=True)
    if skipped:
        print(f"[parse] Skipped {skipped} lines in {p} (format/label mismatch).")
    return df

# -----------------------------
# Centered window & encoding
# -----------------------------
def make_centered_window(L: str, R: str, target_len: int, pad_char: str = "N") -> str:
    left_len  = target_len // 2
    right_len = target_len - left_len - 1
    left_ctx  = L[-left_len:] if len(L) >= left_len else (pad_char * (left_len - len(L))) + L
    right_ctx = R[:right_len] if len(R) >= right_len else R + (pad_char * (right_len - len(R)))
    return left_ctx + "A" + right_ctx

def onehot_encode_batch(seqs, mapping={'A':0,'C':1,'G':2,'T':3}) -> np.ndarray:
    B = len(seqs); L = len(seqs[0])
    x = np.zeros((B, L, 4, 1), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s):
            k = mapping.get(ch, None)
            if k is not None: x[i, j, k, 0] = 1.0
    return x

def revcomp(s: str) -> str:
    return s.translate(str.maketrans("ACGTN","TGCAN"))[::-1]

# -----------------------------
# EditPredict model I/O
# -----------------------------
def load_editpredict_model(repo_dir: Path):
    json_path = repo_dir / "editPredict_weight_alu.json"          # architecture
    h5_path   = repo_dir / "editPredict_construction_alu.h5"      # weights
    if not json_path.exists(): raise FileNotFoundError(f"Missing model JSON: {json_path}")
    if not h5_path.exists():   raise FileNotFoundError(f"Missing weights H5: {h5_path}")
    with json_path.open("r") as f:
        model = model_from_json(f.read())
    model.load_weights(str(h5_path))
    try: model.trainable = False
    except Exception: pass
    return model

def required_input_len(model) -> int:
    shp = getattr(model, "input_shape", None)
    if isinstance(shp, (list, tuple)) and isinstance(shp[0], (list, tuple)):
        shp = shp[0]
    if not isinstance(shp, (list, tuple)) or len(shp) < 4:
        raise ValueError(f"Unrecognized model input_shape: {shp}")
    return int(shp[1])

def predict_probs(model, seqs, batch_size=128):
    probs = []
    n = len(seqs)
    for i in range(0, n, batch_size):
        batch = seqs[i:i+batch_size]
        x = onehot_encode_batch(batch)  # (B,L,4,1)
        p = model.predict(x, verbose=0)
        p = np.array(p)
        if p.ndim == 2 and p.shape[1] == 1:
            probs.extend(p[:,0].tolist())
        elif p.ndim == 2 and p.shape[1] >= 2:
            probs.extend(p[:,-1].tolist())  # last logit as positive
        elif p.ndim == 1:
            probs.extend(p.tolist())
        else:
            probs.extend(np.max(p, axis=1).tolist())
    return np.asarray(probs, dtype=np.float32)

# -----------------------------
# Metrics & sweeps
# -----------------------------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    out = {
        "Accuracy":    float(metrics.accuracy_score(y_true, y_pred)),
        "Precision":   float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "Recall":      float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp) if (tn + fp) else 0.0),
        "F1":          float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Threshold":   float(threshold), "N": int(len(y_true)),
    }
    # curves
    try:
        out["AUROC"] = float(metrics.roc_auc_score(y_true, y_prob))
    except Exception:
        out["AUROC"] = float("nan")
    try:
        out["AUPRC"] = float(metrics.average_precision_score(y_true, y_prob))
    except Exception:
        out["AUPRC"] = float("nan")
    return out

def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray, steps=101):
    ts = np.linspace(0, 1, steps)
    rows = []
    for t in ts:
        m = compute_metrics(y_true, y_prob, threshold=float(t))
        rows.append(m)
    df = pd.DataFrame(rows).sort_values("Threshold").reset_index(drop=True)
    # best F1 (tie-break by higher specificity)
    best = df.loc[df["F1"].idxmax()]
    return df, best.to_dict()

def dump_curves(y_true: np.ndarray, y_prob: np.ndarray, outdir: Path):
    # ROC
    fpr, tpr, roc_thr = metrics.roc_curve(y_true, y_prob)
    thr_pad = np.full(fpr.shape, np.nan, dtype=float)   # same length as fpr/tpr
    if len(roc_thr) == len(fpr) - 1:
        thr_pad[:-1] = roc_thr
    else:
        # fallback: truncate or pad as needed
        k = min(len(roc_thr), len(fpr))
        thr_pad[:k] = roc_thr[:k]
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thr_pad}).to_csv(outdir / "roc_curve.csv", index=False)

    # PR
    precision, recall, pr_thr = metrics.precision_recall_curve(y_true, y_prob)
    thr_pad_pr = np.full(precision.shape, np.nan, dtype=float)  # same length as precision/recall
    if len(pr_thr) == len(precision) - 1:
        thr_pad_pr[:-1] = pr_thr
    else:
        k = min(len(pr_thr), len(precision))
        thr_pad_pr[:k] = pr_thr[:k]
    pd.DataFrame({"precision": precision, "recall": recall, "threshold": thr_pad_pr}).to_csv(outdir / "pr_curve.csv", index=False)

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="EditPredict baseline with sweeps + both strands")
    ap.add_argument("--train_jsonl", type=Path, required=True)
    ap.add_argument("--valid_jsonl", type=Path, required=True)
    ap.add_argument("--editpredict_dir", type=Path, required=True)
    ap.add_argument("--both_strands", action="store_true", help="Score reverse-complement too and take max")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--outdir", type=Path, default=Path("editpredict_plus_out"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Parse JSONLs")
    df_train = parse_adargpt_jsonl(args.train_jsonl)  # parsed for parity
    df_valid = parse_adargpt_jsonl(args.valid_jsonl)

    print("[2/6] Load EditPredict model")
    model = load_editpredict_model(args.editpredict_dir)
    L_required = required_input_len(model)
    print(f"      Detected input length: {L_required}")

    print("[3/6] Build centered windows")
    seqs = [make_centered_window(L, R, L_required) for L, R in zip(df_valid["L"], df_valid["R"])]
    assert len({len(s) for s in seqs}) == 1 and len(seqs[0]) == L_required

    print("[4/6] Inference")
    p_main = predict_probs(model, seqs, batch_size=args.batch_size)
    if args.both_strands:
        print("      Scoring reverse-complements (both_strands)")
        seqs_rc = [revcomp(s) for s in seqs]
        p_rc = predict_probs(model, seqs_rc, batch_size=args.batch_size)
        probs = np.maximum(p_main, p_rc)
    else:
        probs = p_main

    if len(probs) != len(df_valid):
        raise RuntimeError(f"Got {len(probs)} probs for {len(df_valid)} sequences")

    print("[5/6] Save raw probabilities")
    pd.DataFrame({"id": df_valid["id"], "prob": probs, "label": df_valid["label"]}).to_csv(args.outdir/"probs.csv", index=False)

    y_true = df_valid["label"].to_numpy(dtype=int)

    # Softmax sanity: mean positive prob should exceed mean negative
    pos_m = float(np.mean(probs[y_true==1])) if np.any(y_true==1) else float("nan")
    neg_m = float(np.mean(probs[y_true==0])) if np.any(y_true==0) else float("nan")
    if not np.isnan(pos_m) and not np.isnan(neg_m) and pos_m <= neg_m:
        print(f"[warn] Mean prob(pos)={pos_m:.4f} <= mean prob(neg)={neg_m:.4f}. Check class column!")

    print("[6/6] Metrics, sweep, curves")
    # Fixed 0.5 (paper’s main table)
    m05 = compute_metrics(y_true, probs, threshold=0.5)
    with (args.outdir/"metrics@0.5.json").open("w") as f: json.dump(m05, f, indent=2)

    # Sweep + best F1
    sweep_df, best = sweep_thresholds(y_true, probs, steps=101)
    sweep_df.to_csv(args.outdir/"threshold_sweep.csv", index=False)
    with (args.outdir/"metrics@bestF1.json").open("w") as f: json.dump(best, f, indent=2)

    # Curves
    dump_curves(y_true, probs, args.outdir)

    print("== Summary ==")
    print(f" Fixed-0.5  F1={m05['F1']:.4f}  Acc={m05['Accuracy']:.4f}  Rec={m05['Recall']:.4f}  Spec={m05['Specificity']:.4f}  AUROC={m05['AUROC']:.4f}  AUPRC={m05['AUPRC']:.4f}")
    print(f" Best-F1@t* F1={best['F1']:.4f}  t*={best['Threshold']:.2f}  Acc={best['Accuracy']:.4f}  Rec={best['Recall']:.4f}  Spec={best['Specificity']:.4f}")
    print(f" Mean prob: pos={pos_m:.4f}  neg={neg_m:.4f}")
    print(f"Saved:\n  {args.outdir/'probs.csv'}\n  {args.outdir/'metrics@0.5.json'}\n  {args.outdir/'metrics@bestF1.json'}\n  {args.outdir/'threshold_sweep.csv'}\n  {args.outdir/'roc_curve.csv'}\n  {args.outdir/'pr_curve.csv'}")

if __name__ == "__main__":
    main()

