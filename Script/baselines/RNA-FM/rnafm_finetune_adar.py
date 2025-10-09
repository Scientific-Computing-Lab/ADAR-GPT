#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

# --------- JSONL parsing (same template as before) ----------
L_AR_STRUCT_RE = re.compile(
    r'^L:(?P<L>.*?),\s*A:(?P<center>[ACGTUacgtu]),\s*R:(?P<R>.*)Alu Vienna Structure:(?P<struct>[\.\(\)]+)$',
    re.DOTALL
)

def read_jsonl(p: Path):
    with p.open("r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def sanitize_seq(s: str) -> str:
    # Uppercase, map T->U for RNA-FM, keep N for unknowns, strip other chars to N
    s = s.upper().replace("T", "U")
    s = re.sub(r'[^ACGUN]', 'N', s)
    return s

def parse_adargpt_jsonl(p: Path):
    rows=[]
    for i, rec in enumerate(read_jsonl(p)):
        try:
            user = rec["messages"][1]["content"]
            lab  = rec["messages"][2]["content"].strip().lower()
            label = 1 if lab == "yes" else 0 if lab == "no" else None
            m = L_AR_STRUCT_RE.match(user)
            if (not m) or (label is None): 
                continue
            L = sanitize_seq(m.group("L"))
            center = m.group("center").upper().replace("T","U")
            R = sanitize_seq(m.group("R"))
            if center != "A":  # center must be A
                continue
            rows.append({"id": i, "L": L, "R": R, "label": label})
        except Exception:
            continue
    if not rows:
        raise RuntimeError(f"No usable rows parsed from {p}")
    return pd.DataFrame(rows).reset_index(drop=True)

# --------- Centered window & reverse-complement ----------
def make_centered_window(L: str, R: str, target_len: int, pad_char="N") -> str:
    left_len  = target_len // 2
    right_len = target_len - left_len - 1
    left_ctx  = L[-left_len:] if len(L) >= left_len else (pad_char*(left_len - len(L))) + L
    right_ctx = R[:right_len] if len(R) >= right_len else R + (pad_char*(right_len - len(R)))
    return left_ctx + "A" + right_ctx  # already T->U above; center is 'A'

# RNA alphabet for reverse-complement in U space
_RC = str.maketrans("ACGUN", "UGCAN")
def revcomp_rna(seq: str) -> str:
    return seq.translate(_RC)[::-1]

# --------- HuggingFace dataset ----------
class RNADataset(Dataset):
    def __init__(self, seqs, labels, tokenizer, max_length):
        self.seqs = seqs
        self.labels = labels
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        lab = int(self.labels[idx])
        enc = self.tok(
            s,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(lab, dtype=torch.long)
        return item

# --------- Metrics helpers ----------
def compute_metrics_at_threshold(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    out = {
        "Accuracy":    float(metrics.accuracy_score(y_true, y_pred)),
        "Precision":   float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "Recall":      float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp) if (tn+fp) else 0.0),
        "F1":          float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Threshold":   float(thr), "N": int(len(y_true)),
    }
    try: out["AUROC"] = float(metrics.roc_auc_score(y_true, y_prob))
    except Exception: out["AUROC"] = float("nan")
    try: out["AUPRC"] = float(metrics.average_precision_score(y_true, y_prob))
    except Exception: out["AUPRC"] = float("nan")
    return out

def sweep_thresholds(y_true, y_prob, steps=101):
    ts = np.linspace(0,1,steps)
    rows=[]
    for t in ts:
        rows.append(compute_metrics_at_threshold(y_true, y_prob, thr=float(t)))
    df = pd.DataFrame(rows).sort_values("Threshold").reset_index(drop=True)
    best = df.loc[df["F1"].idxmax()].to_dict()
    return df, best

def dump_curves(y_true, y_prob, outdir: Path):
    fpr, tpr, roc_thr = metrics.roc_curve(y_true, y_prob)
    thr_pad = np.full_like(fpr, np.nan, dtype=float)
    if len(roc_thr) == len(fpr) - 1:
        thr_pad[:-1] = roc_thr
    pd.DataFrame({"fpr":fpr, "tpr":tpr, "threshold":thr_pad}).to_csv(outdir/"roc_curve.csv", index=False)

    prec, rec, pr_thr = metrics.precision_recall_curve(y_true, y_prob)
    thr_pad_pr = np.full_like(prec, np.nan, dtype=float)
    if len(pr_thr) == len(prec) - 1:
        thr_pad_pr[:-1] = pr_thr
    pd.DataFrame({"precision":prec, "recall":rec, "threshold":thr_pad_pr}).to_csv(outdir/"pr_curve.csv", index=False)

# --------- Main pipeline ----------
def main():
    ap = argparse.ArgumentParser(description="Fine-tune RNA-FM for A-to-I editing classification and evaluate.")
    ap.add_argument("--train_jsonl", type=Path, required=True)
    ap.add_argument("--valid_jsonl", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--model_id", type=str, default="multimolecule/rnafm",
                    help="HF model id for RNA-FM.")
    ap.add_argument("--window_len", type=int, default=201)
    ap.add_argument("--max_length", type=int, default=256, help="tokenizer max length (>= window_len)")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--both_strands", action="store_true", help="At eval: also score reverse-complement and take max")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    print("[1/6] Parse JSONLs")
    df_tr = parse_adargpt_jsonl(args.train_jsonl)
    df_va = parse_adargpt_jsonl(args.valid_jsonl)

    # Build windows
    print("[2/6] Build centered windows")
    seqs_tr = [make_centered_window(L,R,args.window_len) for L,R in zip(df_tr["L"], df_tr["R"])]
    seqs_va = [make_centered_window(L,R,args.window_len) for L,R in zip(df_va["L"], df_va["R"])]

    labels_tr = df_tr["label"].tolist()
    labels_va = df_va["label"].tolist()

    # Tokenizer / model
    print("[3/6] Load tokenizer/model:", args.model_id)

    # RNA-FM tokenizer + model (from the 'multimolecule' package)
    from multimolecule.tokenisers.rna import RnaTokenizer
    from multimolecule.models.rnafm import RnaFmForSequencePrediction
    from transformers import AutoConfig

    # Tokenizer
    tok = RnaTokenizer.from_pretrained(args.model_id)

    # Config: IMPORTANT → problem_type must be one of: ['regression','binary','multiclass','multilabel']
    config = AutoConfig.from_pretrained(args.model_id)
    config.problem_type = "binary"
    config.num_labels = 1

    # Model with sequence head; ignore_mismatched_sizes attaches a fresh head cleanly
    model = RnaFmForSequencePrediction.from_pretrained(
        args.model_id,
        config=config,
        ignore_mismatched_sizes=True,
    )

    # Datasets
    dset_tr = RNADataset(seqs_tr, labels_tr, tok, args.max_length)
    dset_va = RNADataset(seqs_va, labels_va, tok, args.max_length)

    # Training
    print("[4/6] Fine-tuning")
    targs = TrainingArguments(
        output_dir=str(args.outdir / "hf_runs"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        seed=args.seed,
        report_to=[],
    )

    def compute_eval_metrics(eval_pred):
        # During training we’ll still log loss; final metrics come later
        return {}

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=dset_tr,
        eval_dataset=dset_va,
        tokenizer=tok,
        compute_metrics=compute_eval_metrics,
    )
    trainer.train()

    # Save fine-tuned model (for reuse)
    ft_dir = args.outdir / "rnafm_finetuned_model"
    trainer.save_model(ft_dir)
    tok.save_pretrained(ft_dir)

    # Evaluation (fixed model, probability outputs)
    print("[5/6] Inference on validation")

    def predict_logits(seqs):
        logits = []
        for i in tqdm(range(0, len(seqs), args.batch_size), desc="infer"):
            batch = seqs[i:i + args.batch_size]
            enc = tok(
                batch,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt"
            )
            with torch.no_grad():
                out = model(**{k: v.to(model.device) for k, v in enc.items()})
                lg = out.logits.detach().cpu().numpy()
            logits.append(lg)
        return np.concatenate(logits, axis=0)

    # Main strand inference
    logits_main = predict_logits(seqs_va)
    probs_main = torch.sigmoid(torch.tensor(logits_main)).numpy().squeeze(-1)

    if args.both_strands:
        print("      Scoring reverse-complements (both_strands)")
        seqs_rc = [revcomp_rna(s) for s in seqs_va]
        logits_rc = predict_logits(seqs_rc)
        probs_rc = torch.sigmoid(torch.tensor(logits_rc)).numpy().squeeze(-1)
        probs = np.maximum(probs_main, probs_rc)
    else:
        probs = probs_main

    y_true = np.asarray(labels_va, dtype=int)

    # Save raw probabilities
    print("[6/6] Metrics, sweeps, curves")
    pd.DataFrame({"id": df_va["id"].values, "prob": probs, "label": y_true}).to_csv(args.outdir/"probs.csv", index=False)

    # Fixed 0.5
    m05 = compute_metrics_at_threshold(y_true, probs, thr=0.5)
    with (args.outdir/"metrics@0.5.json").open("w") as f: json.dump(m05, f, indent=2)

    # Sweep & best-F1
    sweep_df, best = sweep_thresholds(y_true, probs, steps=101)
    sweep_df.to_csv(args.outdir/"threshold_sweep.csv", index=False)
    with (args.outdir/"metrics@bestF1.json").open("w") as f: json.dump(best, f, indent=2)

    # Curves
    dump_curves(y_true, probs, args.outdir)

    # Summary print
    pos_m = float(np.mean(probs[y_true==1])) if np.any(y_true==1) else float("nan")
    neg_m = float(np.mean(probs[y_true==0])) if np.any(y_true==0) else float("nan")
    print("== Summary ==")
    print(f" Fixed-0.5  F1={m05['F1']:.4f}  Acc={m05['Accuracy']:.4f}  Rec={m05['Recall']:.4f}  Spec={m05['Specificity']:.4f}  AUROC={m05['AUROC']:.4f}  AUPRC={m05['AUPRC']:.4f}")
    print(f" Best-F1@t* F1={best['F1']:.4f}  t*={best['Threshold']:.2f}  Acc={best['Accuracy']:.4f}  Rec={best['Recall']:.4f}  Spec={best['Specificity']:.4f}")
    print(f" Mean prob: pos={pos_m:.4f}  neg={neg_m:.4f}")
    print("Saved:\n ", args.outdir/"probs.csv",
          "\n ", args.outdir/"metrics@0.5.json",
          "\n ", args.outdir/"metrics@bestF1.json",
          "\n ", args.outdir/"threshold_sweep.csv",
          "\n ", args.outdir/"roc_curve.csv",
          "\n ", args.outdir/"pr_curve.csv")

if __name__ == "__main__":
    main()

