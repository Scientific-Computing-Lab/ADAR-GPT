#!/usr/bin/env python3
"""Generate ROC and PR curves for the 15% SFT vs. CFT comparison.

The script is self-contained: it reads the JSONL prediction dumps that live in
this directory, computes the operating curves, and writes two PNG files
(`liver_roc_curves.png` and `liver_pr_curves.png`) alongside the script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve


DATA_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = DATA_DIR

MODELS: Dict[str, Dict[str, str]] = {
    "Adar-GPT (Static)": {
        "kind": "jsonl",
        "path": "model_outputs_15_SFT_NoStructureonly100.jsonl",
    },
    "Adar-GPT (Continual)": {
        "kind": "jsonl",
        "path": "model_outputs_15_CFT_NoStructureonly100_FT15.jsonl",
    },
    "EditPredict (Pre-trained)": {
        "kind": "csv",
        "path": "baseline_probs/editpredict_pretrained_probs.csv",
    },
    "EditPredict (Fine-tuned)": {
        "kind": "csv",
        "path": "baseline_probs/editpredict_finetuned_probs.csv",
    },
    "RNA-FM (Fine-tuned)": {
        "kind": "csv",
        "path": "baseline_probs/rnafm_finetuned_probs.csv",
    },
}

COLORS = {
    "Adar-GPT (Static)": "#2ca02c",
    "Adar-GPT (Continual)": "#d62728",
    "EditPredict (Pre-trained)": "#808080",
    "EditPredict (Fine-tuned)": "#1f77b4",
    "RNA-FM (Fine-tuned)": "#ff7f0e",
}


def _load_jsonl(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    labels = []
    probs = []
    with path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            label = 1 if record["ground_truth"].lower() == "yes" else 0
            prob_yes = record["label_probabilities"]["yes"]
            labels.append(label)
            probs.append(prob_yes)
    return np.asarray(labels, dtype=int), np.asarray(probs, dtype=float)


def _load_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path)
    labels = df["label"].to_numpy(dtype=int)
    scores = df["prob"].to_numpy(dtype=float)
    return labels, scores


def _plot_curves(
    curves: Dict[str, Dict[str, np.ndarray]],
    kind: str,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    plt.figure(figsize=(10, 7))
    for name, data in curves.items():
        plt.plot(
            data["x"],
            data["y"],
            label=f"{name} (AUC={data['auc']:.3f})",
            color=COLORS.get(name, None),
            linewidth=2.5,
        )
    if kind == "roc":
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.2)
    plt.legend(loc="lower right" if kind == "roc" else "lower left")
    plt.tight_layout()
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=300)
    print(f"Wrote {out_path}")


def main() -> None:
    roc_curves: Dict[str, Dict[str, np.ndarray]] = {}
    pr_curves: Dict[str, Dict[str, np.ndarray]] = {}

    for label, spec in MODELS.items():
        path = DATA_DIR / spec["path"]
        if spec["kind"] == "jsonl":
            y_true, scores = _load_jsonl(path)
        elif spec["kind"] == "csv":
            y_true, scores = _load_csv(path)
        else:
            raise ValueError(f"Unknown data kind: {spec['kind']}")

        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_curves[label] = {"x": fpr, "y": tpr, "auc": auc(fpr, tpr)}

        precision, recall, _ = precision_recall_curve(y_true, scores)
        pr_curves[label] = {"x": recall, "y": precision, "auc": auc(recall, precision)}

    _plot_curves(
        roc_curves,
        kind="roc",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="ROC Curves (15% liver validation)",
        filename="liver_roc_curves.png",
    )
    _plot_curves(
        pr_curves,
        kind="pr",
        xlabel="Recall",
        ylabel="Precision",
        title="Precision–Recall Curves (15% liver validation)",
        filename="liver_pr_curves.png",
    )


if __name__ == "__main__":
    main()
