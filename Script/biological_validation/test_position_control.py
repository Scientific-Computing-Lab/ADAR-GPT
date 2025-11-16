#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import copy
import re
from typing import Optional, Tuple


def extract_L_R_struct(user_content: str) -> Tuple[str, str, str]:
    """
    Expecting format:
      'L:<...>, A:A, R:<...>, Alu Vienna Structure:<...>'
    Returns (L, R, structure).
    Falls back to a regex if the simple split fails.
    """
    try:
        parts = [p.strip() for p in user_content.split(", ")]
        L = parts[0].split("L:")[-1].strip()
        R = parts[2].split("R:")[-1].strip()
        return L, R, 
    except Exception:
        m = re.search(
            r"L:([ACGTN]+).*?A:A.*?R:([ACGTN]+)",
            user_content
        )
        if not m:
            raise ValueError(f"Failed to parse user content: {user_content[:120]}...")
        return m.group(1), m.group(2)


def rebuild_user_content(L: str, R: str) -> str:
    """Rebuild the user message content string after sequence modifications."""
    return f"L:{L}, A:A, R:{R}"


def get_first_user_message_index(messages: list) -> Optional[int]:
    """Return the index of the first message with role == 'user'."""
    for i, m in enumerate(messages):
        if isinstance(m, dict) and m.get("role") == "user":
            return i
    return None


def process_file(input_path: str, out_original_path: str, out_mutated_path: str, offset: int):
    """
    - Read JSONL with chat-style 'messages' objects.
    - Central A index is len(L) (0-based).
    - Target index is central_idx + offset (e.g., offset=5 -> the 5th base after A, i.e., R[4]).
    - Select records where:
        * target index exists within L + 'A' + R, and
        * base at target index != 'G'.
      Write those originals to out_original_path and also write a mutated copy to out_mutated_path
      where base at target index is set to 'G' (length preserved).
    """
    total = 0
    selected = 0
    mutated = 0
    skipped_parse = 0
    skipped_len_mismatch = 0
    skipped_short = 0
    skipped_alreadyG = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(out_original_path, "w", encoding="utf-8") as f_orig, \
         open(out_mutated_path, "w", encoding="utf-8") as f_mut:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1

            # Parse JSON line
            try:
                obj = json.loads(line)
            except Exception:
                skipped_parse += 1
                continue

            # Find user message
            msgs = obj.get("messages", [])
            uidx = get_first_user_message_index(msgs)
            if uidx is None:
                skipped_parse += 1
                continue

            content = msgs[uidx].get("content", "")
            try:
                L, R = extract_L_R_struct(content)
            except Exception:
                skipped_parse += 1
                continue

            full_seq = L + "A" + R
            

            central_idx = len(L)
            target_idx = central_idx + offset

            if target_idx >= len(full_seq):
                # Not enough downstream bases to reach +offset
                skipped_short += 1
                continue

            if full_seq[target_idx] == 'G':
                skipped_alreadyG += 1
                continue

            # (1) Write original unchanged
            f_orig.write(json.dumps(obj, ensure_ascii=False) + "\n")
            selected += 1

            # (2) Mutate: set base at +offset to 'G'
            full_list = list(full_seq)
            full_list[target_idx] = 'G'

            # Re-split back to L and R
            new_L = ''.join(full_list[:len(L)])
            new_R = ''.join(full_list[len(L) + 1:])

            mutated_obj = copy.deepcopy(obj)
            mutated_content = rebuild_user_content(new_L, new_R)
            mutated_obj["messages"][uidx]["content"] = mutated_content
            f_mut.write(json.dumps(mutated_obj, ensure_ascii=False) + "\n")
            mutated += 1

    print("Done.")
    print(f"Total lines read: {total}")
    print(f"Selected with base at +{offset} != 'G': {selected}")
    print(f"Mutated (set +{offset} to 'G'): {mutated}")
    print("--- Skips ---")
    print(f"Parse failures: {skipped_parse}")
    print(f"Length mismatch (L+A+R vs structure): {skipped_len_mismatch}")
    print(f"Insufficient length to reach +{offset}: {skipped_short}")
    print(f"Already 'G' at +{offset}: {skipped_alreadyG}")
    print(f"Outputs:\n - {out_original_path}\n - {out_mutated_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Select records where base at +offset (after central A) is not 'G', and write mutated copies with 'G' at that position."
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("out_original", help="Output JSONL for originals with non-G at +offset")
    parser.add_argument("out_mutated", help="Output JSONL for mutated copies (set +offset to 'G')")
    parser.add_argument("--offset", type=int, default=5,
                        help="Downstream offset after central A (default: 5). +1 means first base after A.")
    args = parser.parse_args()
    process_file(args.input, args.out_original, args.out_mutated, offset=args.offset)


if __name__ == "__main__":
    main()
