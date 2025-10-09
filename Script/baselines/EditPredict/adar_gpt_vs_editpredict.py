#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics

# Silence TF logs a bit (CPU is fine)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from tensorflow.keras.models import model_from_json


# =========================
# Parsing ADAR-GPT JSONL
# =========================

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
    # uppercase, U->T, remove anything not A/C/G/T/N
    s = seq.upper().replace("U","T")
    return re.sub(r'[^ACGTN]', '', s)

def parse_adargpt_jsonl(p: Path) -> pd.DataFrame:
    """Return DataFrame with columns: id, L, R, seq (L+A+R), label (0/1)."""
    rows = []
    for i, rec in enumerate(read_jsonl(p)):
        try:
            user = rec["messages"][1]["content"]
            lab  = rec["messages"][2]["content"].strip().lower()
            label = 1 if lab == "yes" else 0 if lab == "no" else None
            m = L_AR_STRUCT_RE.match(user)
            if not m or label is None:
                continue
            L_raw, center, R_raw = m.group("L"), m.group("center"), m.group("R")
            if center.upper() != "A":
                continue
            Ls = sanitize_seq(L_raw)
            Rs = sanitize_seq(R_raw)
            seq = Ls + "A" + Rs
            rows.append({"id": i, "L": Ls, "R": Rs, "seq": seq, "label": label})
        except Exception:
            continue
    if not rows:
        raise RuntimeError(f"No usable rows parsed from {p}")
    return pd.DataFrame(rows).reset_index(drop=True)


# =========================
# Centered window & encoding
# =========================

def make_centered_window(L: str, R: str, target_len: int, pad_char: str = "N") -> str:
    """
    Build a fixed-length window centered on the edited A:
      [ left_len bases from end of L ] + 'A' + [ right_len bases from start of R ]
    with padding if needed.
    """
    left_len  = target_len // 2
    right_len = target_len - left_len - 1

    left_ctx  = L[-left_len:] if len(L) >= left_len else (pad_char * (left_len - len(L))) + L
    right_ctx = R[:right_len] if len(R) >= right_len else R + (pad_char * (right_len - len(R)))

    return left_ctx + "A" + right_ctx

def onehot_encode_batch(seqs, mapping={'A':0,'C':1,'G':2,'T':3}) -> np.ndarray:
    """Return (B, L, 4, 1) one-hot. N → all zeros."""
    B = len(seqs)
    L = len(seqs[0])
    x = np.zeros((B, L, 4, 1), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, ch in enumerate(s):
            k = mapping.get(ch, None)
            if k is not None:
                x[i, j, k, 0] = 1.0
    return x


# =========================
# EditPredict model
# =========================

def load_editpredict_model(repo_dir: Path):
    """
    Expects in repo_dir:
      - editPredict_weight_alu.json         (architecture)
      - editPredict_construction_alu.h5     (weights)
    """
    json_path = repo_dir / "editPredict_weight_alu.json"
    h5_path   = repo_dir / "editPredict_construction_alu.h5"
    if not json_path.exists():
        raise FileNotFoundError(f"Missing model JSON: {json_path}")
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing weights H5: {h5_path}")

    with json_path.open("r") as f:
        model = model_from_json(f.read())
    model.load_weights(str(h5_path))
    try: model.trainable = False
    except Exception: pass
    return model

def required_input_len(model) -> int:
    """
    Infer required window length from Keras input shape: (None, L, 4, 1).
    Works for Sequential and most Functional models with single input.
    """
    shp = getattr(model, "input_shape", None)
    if isinstance(shp, (list, tuple)) and isinstance(shp[0], (list, tuple)):
        shp = shp[0]  # multiple inputs: take the first
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
        # normalize to one prob per example
        if p.ndim == 2 and p.shape[1] == 1:
            probs.extend(p[:,0].tolist())
        elif p.ndim == 2 and p.shape[1] >= 2:
            probs.extend(p[:,-1].tolist())   # assume last column is positive class
        elif p.ndim == 1:
            probs.extend(p.tolist())
        else:
            probs.extend(np.max(p, axis=1).tolist())
    return np.asarray(probs, dtype=np.float32)


# =========================
# Metrics
# =========================

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    return {
        "Accuracy":    float(metrics.accuracy_score(y_true, y_pred)),
        "Precision":   float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "Recall":      float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp) if (tn + fp) else 0.0),
        "F1":          float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        "AUROC":       float(metrics.roc_auc_score(y_true, y_prob)) if len(np.unique(y_true))==2 else float("nan"),
        "AUPRC":       float(metrics.average_precision_score(y_true, y_prob)) if len(np.unique(y_true))==2 else float("nan"),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Threshold": float(threshold),
        "N": int(len(y_true)),
    }


# =========================
# Main
# =========================

def main():
    ap = argparse.ArgumentParser(description="ADAR-GPT → EditPredict baseline (centered window)")
    ap.add_argument("--train_jsonl", type=Path, required=True)
    ap.add_argument("--valid_jsonl", type=Path, required=True)
    ap.add_argument("--editpredict_dir", type=Path, required=True,
                    help="Directory containing editPredict_weight_alu.json & editPredict_construction_alu.h5")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--outdir", type=Path, default=Path("editpredict_baseline_out"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    print("[1/4] Parse JSONLs")
    df_train = parse_adargpt_jsonl(args.train_jsonl)   # not used by EditPredict; parsed for parity
    df_valid = parse_adargpt_jsonl(args.valid_jsonl)

    print("[2/4] Load EditPredict model")
    model = load_editpredict_model(args.editpredict_dir)
    L_required = required_input_len(model)
    print(f"      Detected EditPredict input length: {L_required}")

    print("[3/4] Build centered windows & infer")
    seqs = [make_centered_window(L, R, L_required) for L, R in zip(df_valid["L"], df_valid["R"])]
    # sanity: all sequences equal to L_required and A centered
    assert len({len(s) for s in seqs}) == 1 and len(seqs[0]) == L_required
    probs = predict_probs(model, seqs, batch_size=args.batch_size)

    if len(probs) != len(df_valid):
        raise RuntimeError(f"Got {len(probs)} probs for {len(df_valid)} sequences")

    # Save per-example probabilities
    per_path = args.outdir / "editpredict_probs_valid.csv"
    pd.DataFrame({"id": df_valid["id"], "prob": probs, "label": df_valid["label"]}).to_csv(per_path, index=False)

    print("[4/4] Metrics")
    y_true = df_valid["label"].to_numpy(dtype=int)
    stats = compute_metrics(y_true, probs, threshold=args.threshold)

    with (args.outdir / "editpredict_metrics_valid.json").open("w") as f:
        json.dump(stats, f, indent=2)

    print("== Baseline metrics (validation) ==")
    for k in ["Accuracy","Precision","Recall","Specificity","F1","AUROC","AUPRC","TN","FP","FN","TP","Threshold","N"]:
        print(f"{k:>12}: {stats[k]}")
    print(f"\nSaved:\n  {per_path}\n  {args.outdir / 'editpredict_metrics_valid.json'}")


if __name__ == "__main__":
    main()

