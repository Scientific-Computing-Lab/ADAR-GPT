#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, os, re, subprocess, sys, textwrap, shutil
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------------
# ADAR-GPT JSONL parsing
# -------------------------
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
    s = s.upper().replace("U","T")
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
            Lraw, center, Rraw = m.group("L"), m.group("center"), m.group("R")
            if center.upper() != "A":
                skipped += 1; continue
            rows.append({"id": i, "L": sanitize_seq(Lraw), "R": sanitize_seq(Rraw), "label": label})
        except Exception:
            skipped += 1
    if not rows:
        raise RuntimeError(f"No usable rows parsed from {p}")
    df = pd.DataFrame(rows)
    if skipped:
        print(f"[parse] Skipped {skipped} lines in {p}")
    return df.reset_index(drop=True)

# -------------------------
# Centered window
# -------------------------
def make_centered_window(L: str, R: str, target_len: int, pad_char: str="N") -> str:
    left_len  = target_len // 2
    right_len = target_len - left_len - 1
    left_ctx  = L[-left_len:] if len(L) >= left_len else (pad_char*(left_len - len(L))) + L
    right_ctx = R[:right_len] if len(R) >= right_len else R + (pad_char*(right_len - len(R)))
    return left_ctx + "A" + right_ctx

# -------------------------
# Build EditPredict training text
# Format expected by their trainer: "SEQUENCE,LABEL"
# (Comma-separated, label is last field)
# -------------------------
def write_editpredict_train_txt(df: pd.DataFrame, out_txt: Path, input_len: int):
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    with out_txt.open("w") as f:
        for L,R,y in zip(df["L"], df["R"], df["label"]):
            seq = make_centered_window(L,R,input_len)
            f.write(f"{seq},{int(y)}\n")

# -------------------------
# Patch their training script (minimally)
# Adds argparse and saves weights after training
# -------------------------
PATCHED_TRAIN = r'''
#!/usr/bin/env python3
# Patched from original editPredict_train.py to accept CLI args and save weights
import argparse, numpy as np, json
from sklearn.model_selection import train_test_split
from tensorflow import keras

def onehot_seq(s):
    m = {'A':0,'C':1,'G':2,'T':3}
    L = len(s)
    x = np.zeros((L,4,1), dtype='float32')
    for i,ch in enumerate(s):
        k = m.get(ch, None)
        if k is not None: x[i,k,0]=1.0
    return x

def load_dataset(txt_path):
    seqs, labs = [], []
    with open(txt_path,"r") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            parts=line.split(',')
            seq = parts[0].strip().upper()
            lab = int(parts[-1].strip())
            seqs.append(seq); labs.append(lab)
    X = np.stack([onehot_seq(s) for s in seqs], axis=0)   # (N, L, 4, 1)
    y = np.array(labs, dtype='int64')
    return X, y

def build_model(input_shape, num_classes=2):
    # input_shape = (L, 4, 1). Use length-wise convs so width=4 never collapses.
    model = keras.models.Sequential(name="EditPredict_lenwise")
    model.add(keras.layers.Conv2D(32, (3,1), padding='same', activation='relu', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,1)))
    model.add(keras.layers.Conv2D(64, (3,1), padding='same', activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,1)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                  metrics=['accuracy'])
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_txt", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--val_split", type=float, default=0.2)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_h5", required=True)
    args = ap.parse_args()

    X, y = load_dataset(args.train_txt)
    img_rows, img_cols = X.shape[1], X.shape[2]
    X = X.reshape(X.shape[0], img_rows, img_cols, 1)  # (N, L, 4, 1)
    num_classes = 2
    y_cat = keras.utils.to_categorical(y, num_classes)

    Xtr, Xte, ytr, yte = train_test_split(X, y_cat, test_size=args.val_split,
                                          stratify=y, random_state=42)
    print("Train:", Xtr.shape, "Valid:", Xte.shape)

    model = build_model((img_rows,img_cols,1), num_classes)
    model.summary()

    hist = model.fit(Xtr, ytr, epochs=args.epochs, batch_size=args.batch_size,
                     validation_data=(Xte,yte), verbose=2)

    score = model.evaluate(Xte, yte, verbose=0)
    print("Valid loss:", score[0], "Valid acc:", score[1])

    # Save architecture + weights
    json_str = model.to_json()
    with open(args.out_json, "w") as f: f.write(json_str)
    model.save_weights(args.out_h5)
    print("Saved:", args.out_json, args.out_h5)

if __name__ == "__main__":
    main()
'''.lstrip()

def write_patched_trainer(editpredict_dir: Path, out_path: Path):
    if out_path.exists():
        print(f"[patch] Using existing patched trainer → {out_path}")
        return
    out_path.write_text(PATCHED_TRAIN, encoding="utf-8")
    os.chmod(out_path, 0o755)
    print(f"[patch] Wrote patched trainer → {out_path}")

# -------------------------
# Run helpers
# -------------------------
def run_cmd(cmd, cwd=None):
    print(">>", " ".join(map(str,cmd)))
    completed = subprocess.run(cmd, cwd=cwd, check=True)
    return completed.returncode

def main():
    ap = argparse.ArgumentParser(description="End-to-end: build EP training files, train, and evaluate.")
    ap.add_argument("--train_jsonl", required=True, type=Path)
    ap.add_argument("--valid_jsonl", required=True, type=Path)
    ap.add_argument("--editpredict_dir", required=True, type=Path, help="Path to original EditPredict repo dir")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--input_len", type=int, default=201, help="Centered window length (EditPredict expects 201)")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--val_split", type=float, default=0.2, help="Internal split inside EP trainer (kept for parity)")
    ap.add_argument("--evaluate_with_plus", action="store_true", help="Run adar_gpt_vs_editpredict_plus.py on the new weights")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    data_dir = args.outdir / "data"
    model_dir = args.outdir / "model_ep_retrained"
    data_dir.mkdir(exist_ok=True, parents=True)
    model_dir.mkdir(exist_ok=True, parents=True)

    # 1) Build training txt from your ADAR-GPT TRAIN split
    df_train = parse_adargpt_jsonl(args.train_jsonl)
    train_txt = data_dir / "ep_train.txt"
    write_editpredict_train_txt(df_train, train_txt, args.input_len)
    print(f"[data] Wrote {train_txt} with {len(df_train)} examples")

    # (Optional) Keep a matching valid file for downstream evaluation
    df_valid = parse_adargpt_jsonl(args.valid_jsonl)
    valid_txt = data_dir / "ep_valid.txt"
    write_editpredict_train_txt(df_valid, valid_txt, args.input_len)
    print(f"[data] Wrote {valid_txt} with {len(df_valid)} examples")

    # 2) Patch trainer (side-by-side; original remains untouched)
    patched = args.outdir / "editPredict_train_patched.py"
    write_patched_trainer(args.editpredict_dir, patched)

    # 3) Train (will save JSON + H5)
    json_out = model_dir / "editPredict_weight_alu.json"
    h5_out   = model_dir / "editPredict_construction_alu.h5"
    cmd = [
        sys.executable, str(patched),
        "--train_txt", str(train_txt),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
        "--val_split", str(args.val_split),
        "--out_json", str(json_out),
        "--out_h5",   str(h5_out),
    ]
    run_cmd(cmd)

    # 4) Optionally evaluate with your existing plus script
    if args.evaluate_with_plus:
        # We re-use your evaluator by pointing --editpredict_dir to the new model_dir
        eval_out = args.outdir / "evaluation"
        eval_out.mkdir(exist_ok=True, parents=True)
        eval_cmd = [
            sys.executable, "adar_gpt_vs_editpredict_plus.py",
            "--train_jsonl", str(args.train_jsonl),
            "--valid_jsonl", str(args.valid_jsonl),
            "--editpredict_dir", str(model_dir),
            "--outdir", str(eval_out),
        ]
        run_cmd(eval_cmd)

    print("\n[done] Retraining pipeline complete.")
    print("Model saved under:", model_dir)
    print("Data files under:", data_dir)

if __name__ == "__main__":
    main()

