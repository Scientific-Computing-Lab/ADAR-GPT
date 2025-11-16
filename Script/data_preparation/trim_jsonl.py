#!/usr/bin/env python3
import json, re, sys
from pathlib import Path

PAT = re.compile(r"""
    \bL:(?P<L>[^,]*),\s*
    A:(?P<A>[ACGTNacgtn]),\s*
    R:(?P<R>[^,]*)
""", re.VERBOSE)

def trim_and_pad(left, right, flank=100, pad="N"):
    """Uppercase and pad to fixed flanks around the central A (default: 100 on each side)."""
    L = (left or "").upper()
    R = (right or "").upper()
    L100 = L[-flank:]
    if len(L100) < flank:
        L100 = pad * (flank - len(L100)) + L100
    R100 = R[:flank]
    if len(R100) < flank:
        R100 = R100 + pad * (flank - len(R100))
    return L100, R100

def process_line(line):
    """Parse one JSONL line, extract L/A/R from the first user message, and rewrite with fixed flanks."""
    obj = json.loads(line)
    msgs = obj.get("messages", [])
    # Find the first user message
    for m in msgs:
        if m.get("role") == "user" and "content" in m:
            c = m["content"]
            mobj = PAT.search(c)
            if not mobj:
                # If L/A/R pattern not found — return as-is
                return json.dumps(obj, ensure_ascii=False)
            Lseq = mobj.group("L").strip()
            A = mobj.group("A").strip().upper()
            Rseq = mobj.group("R").strip()
            L100, R100 = trim_and_pad(Lseq, Rseq, flank=100, pad="N")
            # Rebuild content to include only normalized L/A/R with fixed flanks
            m["content"] = f"L:{L100}, A:{A}, R:{R100}"
            break
    return json.dumps(obj, ensure_ascii=False)

def main(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            fout.write(process_line(line) + "\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python trim_jsonl.py <input.jsonl> <output.jsonl>")
        sys.exit(1)
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    main(in_path, out_path)
